# -*- coding: utf-8 -*-
"""
Cok buyuk / karmasik ham CSV dosyalarini COKMEDEN, TAM VERI uzerinde
calisacak sekilde iceri alma katmani.

Mimari (v2):
  - Dosya asla pandas'a tamamen yuklenmez. DuckDB, CSV'yi diskten okuyup bir
    kez KALICI bir .duckdb dosyasina (tabloya) yazar ("ingest"). Bu, pahali
    CSV ayristirma isini bir kere yapar; sonraki her analiz sorgusu artik
    hizli, sikistirilmis, kolonsal bir tablo uzerinde calisir.
  - Ingest sonucu dosya boyutu+degisim zamanina gore diskte onbelleklenir;
    ayni dosya tekrar yuklendiginde yeniden ayristirma YAPILMAZ.
  - Tum analiz fonksiyonlari (bkz. analytics.py) bu tabloyu SQL ile sorgular;
    boylece sonuclar HER ZAMAN dosyadaki TAM veriye gore hesaplanir - kucuk
    bir ornekleme uzerinden degil. Ornekleme sadece bellekte calismasi
    gereken ML modelleri (anomali tespiti, ozellik onemi) icin, acikca
    belirtilerek kullanilir.
  - Karmasik / bozuk / Turkce bicimli CSV'lere (';' ayrac, '12,5' ondalik
    virgul, bozuk satirlar, farkli kodlamalar) karsi cok asamali bir
    "dene, olmazsa bir sonrakine gec" (fallback) zinciri uygulanir; hicbiri
    calismazsa kullaniciya HAM PYTHON HATASI degil, anlasilir bir Turkce
    mesaj gosterilir.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

CACHE_DIR = Path(tempfile.gettempdir()) / "uap2040_dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TABLE = "raw_data"

# CSV ayristirma sirasinda denenecek ayraclar (sirayla denenir)
_DELIMS = [None, ",", ";", "\t", "|"]  # None -> duckdb otomatik tespit etsin

# ONEMLI: DuckDB'nin CSV okuyucusundaki 'encoding' parametresi yalnizca
# utf-8 / utf-16 / latin-1 destekler; Turkce Windows dosyalarda cok yaygin
# olan 'cp1254' gibi kod sayfalarini SESSIZCE (hata vermeden) yanlis
# cozumler ve karakterleri bozar (ör. "Çiğli" -> "?i?li"). Bu nedenle dosya
# DuckDB'ye verilmeden once Python ile (dogru kod sayfasi tespit edilerek)
# guvenli sekilde UTF-8'e cevrilir - bkz. _ensure_utf8().
_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "cp1252", "latin1"]


class IngestError(RuntimeError):
    """Kullaniciya gosterilecek, anlasilir (Turkce) veri yukleme hatasi."""


@dataclass
class DataSource:
    db_path: str
    total_rows: int
    columns: list[str]
    dtypes: dict[str, str]
    source_path: str
    ingest_seconds: float
    from_cache: bool
    warnings: list[str]


def _cache_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _detect_safe_memory_limit_mb() -> int | None:
    """Konteynerlestirilmis (orn. Streamlit Community Cloud, Linux/cgroup)
    ortamlarda GERCEK bellek sinirini okuyup DuckDB'ye guvenli, ACIK bir
    ust sinir tanimlamak icin kullanilir.

    DOGRULUK/KARARLILIK ICIN KRITIK (gercek bir bulut dagitiminda olculup
    dogrulandi): bu sinir OLMADAN DuckDB, konteynerin GERCEK (kucuk)
    sinirini GOREMEYIP host makinenin bildirdigi TAM bellegi kullanmaya
    calisabilir. Sonuc, Python'a hicbir yakalanabilir hata firlatilmadan
    isletim sistemi tarafindan SESSIZCE "OOM-kill" edilmek olur - bu da
    o an baglı TUM kullanicilar icin TUM uygulamanin (paylasilan tek bir
    konteyner oldugundan) cokmesine yol acar. Ornek: 245MB, 5700 sutunlu
    bir dosya Streamlit Community Cloud'un ucretsiz (1GB) katmaninda
    yuklenmeye calisildiginda konteyner tamamen coktu ("Connection failed
    with status 503", uygulama TUM kullanicilar icin yeniden baslatilana
    kadar erisilemez oldu).

    Bu fonksiyon cgroup dosyalarindan GERCEK sinirini okuyup DuckDB'ye
    acikca bildirir - boylece asilirsa DuckDB KENDISI (OS'ten once)
    yakalanabilir bir `OutOfMemoryException` firlatir; bu, app.py'deki
    mevcut `except Exception` blogu tarafindan zaten yakalanip kullaniciya
    Turkce, anlasilir bir hata olarak gosterilir - TUM uygulama COKMEZ,
    sadece o TEK islem basarisiz olur.

    Yerel Windows/Mac gelistirme ortaminda bu cgroup dosyalari mevcut
    DEGILDIR - bu durumda None donulur ve DuckDB'nin kendi varsayilan
    (mevcut, onceden dogrulanmis) davranisi HICBIR SEKILDE degismez."""
    for cgroup_path in ("/sys/fs/cgroup/memory.max",  # cgroup v2
                        "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            with open(cgroup_path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit_bytes = int(raw)
        except ValueError:
            continue
        # Pratikte "sinirsiz" sayilan cok buyuk degerleri (bazi cgroup v1
        # kurulumlarinda gorulur) yok say - gercek bir konteyner siniri
        # degildir, DuckDB'yi gereksiz yere kisitlar.
        if 0 < limit_bytes < 64 * 1024 ** 3:
            # %65'i kullan - Python yorumlayicisi, Streamlit, pandas vb.
            # icin de pay birakilir; DuckDB TEK basina konteynerin TAMAMINI
            # kullanamaz.
            mb = int(limit_bytes / 1024 / 1024 * 0.65)
            return max(256, mb)
    return None


def _detect_encoding(path: str, sample_bytes: int = 1_000_000) -> str:
    """Dosyanin gercek kodlamasini, DuckDB'ye sormadan, dogrudan Python'un
    tam kod-sayfasi destegiyle tespit eder (DuckDB yalnizca utf-8/utf-16/
    latin-1 tanir; cp1254 gibi Turkce kod sayfalarini SESSIZCE yanlis
    cozer). Sirasiyla dener: gecerli utf-8 mi? degilse Turkce Windows
    dosyalarda en yaygin olan cp1254'u dogrular (yuksek-bit karakter
    oranina bakarak), o da olmazsa iso-8859-9, en sonda latin1 (hep basarili
    olur, veri kaybi riski en yuksek secenektir)."""
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in ("cp1254", "iso-8859-9", "cp1252"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"  # her zaman decode edilir; veri kaybi olabilir ama coke sebep olmaz


def _ensure_utf8(path: str, cache_key: str) -> tuple[str, str | None]:
    """Dosya zaten UTF-8 ise oldugu gibi kullanilir (kopyalanmaz - hizli).
    Degilse, dogru kod sayfasiyla satir satir okunup guvenli/kalici bir
    UTF-8 kopyasi olusturulur (bu maliyet, aynen ingest gibi, dosya
    degismedigi surece bir daha odenmez). DuckDB'ye HER ZAMAN saf UTF-8
    metin verilerek sessiz karakter bozulmasi tamamen onlenir."""
    enc = _detect_encoding(path)
    if enc in ("utf-8", "utf-8-sig"):
        return path, None

    utf8_path = CACHE_DIR / f"{cache_key}_utf8.csv"
    if utf8_path.exists():
        return str(utf8_path), enc
    with open(path, "r", encoding=enc, errors="replace", newline="") as fin, \
         open(utf8_path, "w", encoding="utf-8", newline="") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return str(utf8_path), enc


WIDE_FILE_COL_THRESHOLD = 300  # bu sutun sayisini asan dosyalarda tip algilama atlanir


def _peek_column_count(path: str) -> int:
    """Sadece BASLIK satirini okuyup yaklasik sutun sayisini tahmin eder -
    tam bir CSV ayristirmasi yapmadan, cok ucuz (milisaniyeler) bir on-kontrol.
    Ayrac henuz bilinmedigi icin en yaygin adaylardan en cok bolen kazanir."""
    try:
        with open(path, "rb") as f:
            first_line = f.readline()
        return max((first_line.count(d) for d in (b";", b",", b"\t", b"|")), default=0) + 1
    except OSError:
        return 0


def _try_ingest(con: duckdb.DuckDBPyConnection, path: str) -> list[str]:
    """CSV'yi (UTF-8 oldugu garanti edilmis 'path' icin) TABLE icine
    ayristirir. Basarili olana kadar farkli ayrac/all_varchar
    kombinasyonlarini dener. Hicbiri calismazsa IngestError firlatir."""
    warnings: list[str] = []
    last_err = None

    # PERFORMANS ICIN KRITIK: DuckDB'nin otomatik TIP ALGILAMA adimi (her
    # sutun icin int/float/tarih/... turlerini deneyip en uygununu secmesi)
    # sutun sayisiyla birlikte KATLANARAK yavaslar. Gercek bir 5699 sutunlu
    # anket dosyasinda bu adim TEK BASINA ~2 SAAT surebiliyor (olculup
    # dogrulandi); ayni dosya tip algilama atlanip dogrudan metin (VARCHAR)
    # olarak okundugunda ~30 SANIYEDE bitiyor. Sayisal/kategorik donusum
    # zaten SQL katmaninda (bkz. analytics.numeric_expr/cat_expr) icerige
    # bakarak dogru sekilde yapildigindan bu kisayoldan hicbir dogruluk
    # kaybi olmaz - sadece DuckDB'nin gereksiz on-analizi atlanir.
    wide = _peek_column_count(path) > WIDE_FILE_COL_THRESHOLD

    attempts = []
    if not wide:
        attempts.append(dict(auto_detect=True, ignore_errors=True, null_padding=True))
    attempts.append(dict(auto_detect=True, ignore_errors=True, null_padding=True, all_varchar=True))
    for opts in attempts:
        try:
            con.execute("DROP TABLE IF EXISTS " + TABLE)
            opt_str = ", ".join(f"{k}={v}" for k, v in opts.items())
            con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM read_csv_auto(?, {opt_str})", [path])
            n = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            ncol = len(con.execute(f"DESCRIBE {TABLE}").fetchdf())
            if n > 0 and ncol > 1:
                if opts.get("all_varchar"):
                    if wide:
                        warnings.append(
                            f"Dosya {ncol:,} sutun icerdigi icin hiz amaciyla tip algilama "
                            f"adimi atlandi (tum sutunlar metin olarak okundu); sayisal/kategorik "
                            f"donusum analiz sirasinda icerige bakilarak otomatik yapilir."
                        )
                    else:
                        warnings.append(
                            "Bazi sutunlarda tip tespiti guvenilir yapilamadigi icin tum sutunlar "
                            "metin olarak okundu; sayisal analizlerde otomatik donusum uygulanir."
                        )
                return warnings
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    # 2) farkli ayraclari acikca dene (otomatik tespit yanildiysa)
    for delim in _DELIMS:
        try:
            con.execute("DROP TABLE IF EXISTS " + TABLE)
            kwargs = ["ignore_errors=true", "null_padding=true", "all_varchar=true"]
            kwargs.append(f"delim='{delim}'" if delim is not None else "auto_detect=true")
            opt_str = ", ".join(kwargs)
            con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM read_csv(?, {opt_str})", [path])
            n = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            ncol = len(con.execute(f"DESCRIBE {TABLE}").fetchdf())
            if n > 0 and ncol > 1:
                warnings.append(
                    f"Dosya '{delim or 'otomatik'}' ayraciyla, tum sutunlar metin olarak "
                    f"okunarak yuklendi (guvenli mod)."
                )
                return warnings
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    # 3) Son care: pandas ile parca parca (chunk) okuyup DuckDB tablosuna aktar
    try:
        con.execute("DROP TABLE IF EXISTS " + TABLE)
        first = True
        reader = pd.read_csv(path, encoding="utf-8", sep=None, engine="python",
                              dtype=str, chunksize=100_000, on_bad_lines="skip")
        for chunk in reader:
            if first:
                con.register("_chunk", chunk)
                con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM _chunk")
                first = False
            else:
                con.register("_chunk", chunk)
                con.execute(f"INSERT INTO {TABLE} SELECT * FROM _chunk")
        if not first:
            warnings.append(
                "Dosya standart disi bir bicimde oldugu icin parca parca (guvenli) okundu; "
                "tum sutunlar metin olarak alindi."
            )
            return warnings
    except Exception as e:  # noqa: BLE001
        last_err = e

    raise IngestError(
        f"Dosya okunamadi. Cok karmasik/bozuk bir CSV bicimi olabilir. "
        f"Teknik detay: {last_err}"
    )


def get_or_build(path: str, force_rebuild: bool = False) -> DataSource:
    """Verilen CSV yolu icin (onbellekten ya da sifirdan) bir DuckDB veri
    kaynagi hazirlar. Ayni dosya (boyut+degisim zamani ayni oldugu surece)
    bir daha yeniden ayristirilmaz."""
    if not os.path.isfile(path):
        raise IngestError(f"Dosya bulunamadi: {path}")
    if os.path.getsize(path) == 0:
        raise IngestError("Dosya bos (0 byte).")

    key = _cache_key(path)
    db_path = CACHE_DIR / f"{key}.duckdb"
    meta_path = CACHE_DIR / f"{key}.meta.json"

    if not force_rebuild and db_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return DataSource(
                db_path=str(db_path), total_rows=meta["total_rows"],
                columns=meta["columns"], dtypes=meta["dtypes"], source_path=path,
                ingest_seconds=0.0, from_cache=True, warnings=meta.get("warnings", []),
            )
        except Exception:  # noqa: BLE001
            pass  # onbellek bozuksa yeniden olustur

    for p in (db_path, meta_path):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    t0 = time.time()
    utf8_path, detected_enc = _ensure_utf8(path, key)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA threads=4")
        _mem_limit_mb = _detect_safe_memory_limit_mb()
        if _mem_limit_mb:
            con.execute(f"PRAGMA memory_limit='{_mem_limit_mb}MB'")
        warnings = _try_ingest(con, utf8_path)
        if detected_enc:
            warnings.insert(0, f"Dosya '{detected_enc}' kodlamasi tespit edildi ve UTF-8'e cevrildi.")
        total = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        desc = con.execute(f"DESCRIBE {TABLE}").fetchdf()
        columns = desc["column_name"].tolist()
        dtypes = dict(zip(desc["column_name"], desc["column_type"]))
        con.execute("CHECKPOINT")
    finally:
        con.close()
    elapsed = time.time() - t0

    meta = {"total_rows": total, "columns": columns, "dtypes": dtypes, "warnings": warnings}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    return DataSource(
        db_path=str(db_path), total_rows=total, columns=columns, dtypes=dtypes,
        source_path=path, ingest_seconds=elapsed, from_cache=False, warnings=warnings,
    )


def materialize_upload(uploaded_file) -> str:
    """Streamlit file_uploader nesnesini diske yazip yolunu dondurur
    (boylece geri kalan tum boru hatti dosya-yolu tabanli calisabilir)."""
    safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._-") or "upload.csv"
    tmp_path = CACHE_DIR / f"upload_{hashlib.sha1(uploaded_file.getvalue()[:2048]).hexdigest()[:12]}_{safe_name}"
    if not tmp_path.exists():
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return str(tmp_path)


def open_readonly(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def open_work_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Kalici (salt-okunur) veri dosyasina DOKUNMADAN, uzerine gecici
    goruntuler (VIEW) kurabilecegimiz, bellek-ici yazilabilir bir oturum
    baglantisi acar. Boylece 'aracadet1+aracadet2' gibi birlestirmeler ya da
    'kisi1, kisi2, ...' gibi tekrarlayan alanlarin uzun formata cevrilmesi
    icin CREATE VIEW kullanabiliriz; asil onbellek dosyasi hep salt-okunur
    ve bozulmaz kalir."""
    con = duckdb.connect()
    _mem_limit_mb = _detect_safe_memory_limit_mb()
    if _mem_limit_mb:
        con.execute(f"PRAGMA memory_limit='{_mem_limit_mb}MB'")
    safe_path = str(db_path).replace("'", "''")
    con.execute(f"ATTACH '{safe_path}' AS src (READ_ONLY)")
    con.execute(f"CREATE VIEW {TABLE} AS SELECT * FROM src.{TABLE}")
    return con


def clear_cache():
    for p in CACHE_DIR.glob("*"):
        try:
            p.unlink()
        except OSError:
            pass
