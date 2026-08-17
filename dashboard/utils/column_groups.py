# -*- coding: utf-8 -*-
"""
Siralanmis / numaralandirilmis sutun gruplarini otomatik tespit eden ve
kullanilabilir hale getiren yardimcilar.

Gercek hanehalki ulasim anketi ham verilerinde (bkz. UAP 2040 rapor yapisi)
UC seviyeli ic ice bir desen gorulur:

  1) TEK indeksli, TEK tabanli diziler (orn. "aracadet_1".."aracadet_8"):
     "hanedeki N. arac turunun adedi" gibi bir seyi ifade eder - TOPLA/
     ORTALAMA gibi bir birlestirme ile TEK bir sutuna indirgenebilir.

  2) TEK indeksli, AYNI index kumesini paylasan COK tabanli gruplar
     (orn. "cins_1".."cins_11", "yas_1".."yas_11", "meslek_1".."meslek_11"):
     "hanedeki N. kisi" kaydini ifade eder - UZUN FORMATA (1 satir = 1 kisi)
     cevrilerek Demografik analiz mantigina uygun hale getirilebilir.

  3) CIFT indeksli gruplar (orn. "yol_bas_1_1".."yol_bas_11_10"):
     "N. kisinin M. yolculugu" gibi ic ice tekrarlayan bir kaydi ifade eder -
     UZUN FORMATA (1 satir = 1 yolculuk) cevrilip istenirse kisi
     demografisiyle (JOIN) zenginlestirilerek Yolculuk analiz mantigina
     uygun hale getirilebilir.
"""
from __future__ import annotations
import re
from collections import defaultdict

_DOUBLE_RE = re.compile(r"^(.*?)[_\-\s](\d{1,3})[_\-\s](\d{1,3})$")
_SINGLE_RE = re.compile(r"^(.*?)[_\-\s]?(\d{1,3})$")


def ident(col: str) -> str:
    return '"' + str(col).replace('"', '""') + '"'


def _not_blank(col_ident: str) -> str:
    """Bir sutunun (zaten tirnaklanmis SQL kimligi) GERCEKTEN dolu olup
    olmadigini kontrol eder.

    DOGRULUK VE PERFORMANS ICIN KRITIK: CSV/Excel'lerde eksik bir hucre her
    zaman gercek NULL olarak okunmaz - cok genis dosyalarda hiz icin
    kullanilan 'tum sutunlar metin' modunda (bkz. data_io) bos hucreler
    genelde '' (bos metin) olarak gelir. Sadece 'col IS NOT NULL' kontrolu
    bunu YAKALAYAMAZ. Bunun sonucu: 'boş kisi/yolculuk yuvasi' (orn.
    hanede kayitli olmayan 4. kisi ya da hic yapilmamis 7. yolculuk) da
    'dolu' sayilip uzun formatta GERCEK bir satir gibi uretilir - gercek
    bir dosyada bu, 11 kisi x 10 yolculuk = 110 olasi yuvanin TAMAMININ
    (aslinda sadece birkacinin dolu oldugu halde) satira donusmesine, yani
    1.7 MILYON satirlik dev/curuk bir tablo olusmasina yol acmisti (olculup
    dogrulandi). Bu fonksiyon olmadan hem sonuclar yanlis (var olmayan
    yolculuklar) hem de performans feci (devasa tablo) olur."""
    return f"NULLIF(TRIM(CAST({col_ident} AS VARCHAR)), '') IS NOT NULL"


# ---------------------------------------------------------------- tespit
def detect_all_groups(columns: list[str]) -> dict:
    """Sutunlari uc kategoriye ayirir:
      - double: base -> {(i1, i2): colname}   (kisi x yolculuk gibi ic ice)
      - single: base -> {idx: colname}         (double'a dahil olmayanlar)
      - plain:  numarasiz sutunlar
    """
    double: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
    consumed: set[str] = set()
    for col in columns:
        m2 = _DOUBLE_RE.match(col.strip())
        if m2:
            base = m2.group(1).strip("_ -")
            if len(base) < 2:
                continue
            i1, i2 = int(m2.group(2)), int(m2.group(3))
            double[base][(i1, i2)] = col
            consumed.add(col)

    single: dict[str, dict[int, str]] = defaultdict(dict)
    for col in columns:
        if col in consumed:
            continue
        m1 = _SINGLE_RE.match(col.strip())
        if m1:
            base = m1.group(1).strip("_ -")
            if len(base) < 2:
                continue
            single[base][int(m1.group(2))] = col
            consumed.add(col)

    double = {b: v for b, v in double.items() if len(v) >= 2}
    single = {b: v for b, v in single.items() if len(v) >= 2}
    plain = [c for c in columns if c not in consumed]
    return {"double": double, "single": single, "plain": plain}


def cluster_single_blocks(single: dict[str, dict[int, str]]) -> list[dict]:
    """Ayni index imzasina sahip (orn. 1..11) coklu tabanlari (2+ farkli
    taban - orn. cins+yas+egit) tek bir 'tekrarlayan kayit blogu' olarak
    gruplar (kisi N gibi).

    ONEMLI: filtre BASE SAYISINA gore yapilir (len(bases) < 2), index
    sayisina gore DEGIL. Eski hali (len(sig) < 2) tek bir tabani bile (orn.
    sadece 'aracadet_1..3') yanlislikla 'coklu blok' sayiyor, bu da onu
    yanlislikla melt-only isaretleyip TOPLA/ORTALAMA listesinden disliyordu
    - halbuki tek-tabanli diziler icin toplama/ortalama tam olarak istenen
    davranistir."""
    by_sig: dict[tuple, list[str]] = defaultdict(list)
    for base, idxmap in single.items():
        by_sig[tuple(sorted(idxmap.keys()))].append(base)
    blocks = []
    for sig, bases in by_sig.items():
        if len(bases) < 2:
            continue
        blocks.append({
            "indices": list(sig), "bases": sorted(bases),
            "columns": {b: single[b] for b in bases}, "kind": "multi",
        })
    return sorted(blocks, key=lambda x: -len(x["bases"]))


def cluster_double_blocks(double: dict[str, dict[tuple[int, int], str]]) -> list[dict]:
    """Ayni (kisi-araligi, yolculuk-araligi) imzasina sahip cift-indeksli
    tabanlari tek bir 'yolculuk blogu' olarak gruplar."""
    by_sig: dict[tuple, list[str]] = defaultdict(list)
    for base, pairmap in double.items():
        i1s = tuple(sorted(set(p[0] for p in pairmap)))
        i2s = tuple(sorted(set(p[1] for p in pairmap)))
        by_sig[(i1s, i2s)].append(base)
    blocks = []
    for (i1s, i2s), bases in by_sig.items():
        blocks.append({
            "outer_indices": list(i1s), "inner_indices": list(i2s),
            "bases": sorted(bases), "columns": {b: double[b] for b in bases},
            "kind": "double",
        })
    return sorted(blocks, key=lambda x: -len(x["bases"]))


def block_columns(block: dict) -> list[str]:
    """Bir bloktaki (single ya da double) TUM ham sutun adlarini duz bir
    liste olarak dondurur (melt oncesi 'dar ara tablo' olusturmak icin)."""
    cols: list[str] = []
    for base in block["bases"]:
        cols.extend(block["columns"][base].values())
    return cols


def suggest_unique_key(con, table: str, candidates: list[str], total_rows: int) -> tuple[list[str], float]:
    """Kimlik sutunu (household_id/id_cols) icin, sadece anahtar kelimeye
    degil GERCEK TEKILLIGE bakarak en iyi aday(lar)i onerir.

    DOGRULUK ICIN KRITIK: gercek anket dosyalarinda 'haneno' gibi 'hane'
    kelimesini iceren bir sutun her zaman GLOBAL olarak tekil olmayabilir
    (orn. kume/blok bazinda sifirlanan YEREL bir sira numarasi olabilir).
    Boyle tekil olmayan bir sutunu 'kisi/yolculuk tablosu' birlestirmesinde
    kimlik anahtari olarak kullanmak, JOIN'de devasa bir carpma (fan-out)
    yaratir - gercek bir dosyada bu, 15.610 satirlik veriden 30 MILYON
    satirlik curuk bir sonuc uretmisti (olculup dogrulandi). Bu fonksiyon
    once TEK sutun adaylarini (en tekil olan basa), tek sutun yeterli
    degilse EN IYI IKI sutunun BIRLESIMINI dener ve (secilen sutunlar,
    tekillik orani) dondurur - tekillik orani 1.0 ise anahtar guvenlidir."""
    if not candidates or total_rows == 0:
        return (candidates[:1] if candidates else []), 0.0

    scores: list[tuple[float, str]] = []
    for col in candidates[:15]:  # cok fazla adayi taramamak icin sinirla
        try:
            n = con.execute(f"SELECT COUNT(DISTINCT {ident(col)}) FROM {table}").fetchone()[0]
        except Exception:  # noqa: BLE001
            continue
        scores.append((n / total_rows, col))
    if not scores:
        return candidates[:1], 0.0
    scores.sort(key=lambda x: -x[0])
    best_ratio, best_col = scores[0]
    if best_ratio >= 0.999:
        return [best_col], best_ratio

    # tek sutun tekil degil - en iyi iki adayin BIRLESIMINI dene (orn.
    # 'kumeno' + 'haneno' birlikte tekil olabilir, ayri ayri olmasalar bile)
    top2 = [c for _, c in scores[:2]]
    if len(top2) == 2:
        try:
            combo_expr = " || '|' || ".join(f"CAST({ident(c)} AS VARCHAR)" for c in top2)
            n_combo = con.execute(f"SELECT COUNT(DISTINCT {combo_expr}) FROM {table}").fetchone()[0]
            combo_ratio = n_combo / total_rows
            if combo_ratio > best_ratio:
                return top2, combo_ratio
        except Exception:  # noqa: BLE001
            pass
    return [best_col], best_ratio


def count_key_duplicates(con, table: str, key_cols: list[str]) -> int:
    """Verilen sutun(lar)in TABLODA gercekten tekil bir anahtar olusturup
    olusturmadigini dogrudan olcer (tekrar eden anahtar sayisini dondurur;
    0 = tam tekil, guvenli). JOIN'den ONCE bu kontrolun yapilmasi, tekil
    olmayan bir anahtarin (bkz. suggest_unique_key) devasa bir fan-out'a
    (orn. 15 bin satirdan 30 milyon satira) yol acmasini onler."""
    if not key_cols:
        return 0
    combo = " || '|' || ".join(f"CAST({ident(c)} AS VARCHAR)" for c in key_cols)
    row = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {combo}) FROM {table}").fetchone()
    return max(0, row[0] - row[1])


def build_narrow_staging_sql(source_table: str, needed_cols: list[str], staging_name: str) -> str:
    """PERFORMANS ICIN KRITIK: cok genis (binlerce sutunlu) bir kaynaktan,
    asagida yapilacak agir bir islem (orn. 100+ dalli UNION ALL melt) icin
    sadece gercekten gereken sutunlari TEK SEFERDE yerel/kucuk bir tabloya
    kopyalar. 110 dalli bir melt sorgusu DOGRUDAN 5699 sutunlu genis
    kaynaga karsi calistirildiginda, her dal ayri ayri o genis semaya karsi
    baglanip planlanmak zorunda kaliyor - gercek bir dosyada (15.6K satir,
    5699 sutun) bu YAKLASIK 4 DAKIKA suruyordu. Once ihtiyac duyulan ~50
    sutunu kucuk bir ara tabloya alip melt'i ONA karsi calistirmak (bu ara
    adimla birlikte) toplam sureyi ~20 SANIYEYE indirdi (olculup
    dogrulandi) - hicbir sonuc/dogruluk degisikligi olmadan, sadece
    DuckDB'nin tekrarlanan baglama/planlama maliyetini ortadan kaldirarak."""
    seen = list(dict.fromkeys(needed_cols))  # sirali + tekrarsiz
    sel = ", ".join(ident(c) for c in seen)
    return f"CREATE OR REPLACE TABLE {ident(staging_name)} AS SELECT {sel} FROM {source_table}"


# ---------------------------------------------------------------- sayisal-gibi mi?
def numeric_like_bases(con, table: str, groups: dict[str, dict[int, str]],
                        numeric_expr_fn, min_ratio: float = 0.5, sample_limit: int = 3000) -> set[str]:
    """Her taban icin (grubun ILK sutununa bakarak), degerlerin YETERINCE
    BUYUK bir kismi (varsayilan >= %50) sayisal olarak ayristirilabiliyor
    mu diye kontrol eder - AYNI numeric_expr_fn parse mantigiyla (TOPLA/
    ORTALAMA/MAKS gibi islemlerin FIILEN kullanacagi parse ile BIREBIR
    ayni, boylece "gorunuste sayisal" ile "gercekte parse edilebilir"
    arasinda fark olmaz).

    NEDEN: 'Tekrarlayan sutunlari duzenle' panelinin '1️⃣ Topla/Ortalama/
    Maks' bolumu, tespit edilen HER sirali/numarali grubu (isim, adres,
    saat gibi ac1kca metin-bazli olanlar dahil) gosteriyordu - byle bir
    grup secilip 'Topla' denildiginde, numeric_expr_fn hicbirini sayiya
    ceviremedigi icin COALESCE(...,0) devreye giriyor ve SESSIZCE her
    satirda '0' URETIYORDU (crash yok, ama tamamen ANLAMSIZ/YANLIS
    gorunen bir sonuc - gercek bir dosyada 'adi' (isim) sutunuyla olculup
    dogrulandi). Bu fonksiyon, byle gruplari '1️⃣ Topla' listesinden
    ONCEDEN elemek icin kullanilir - '3️⃣ Kac kisi bu cevabi verdi'
    (kategorik sayim) bolumunde bu gruplar YINE DE kullanilabilir kalir,
    cunku o islem metin degerler icin de anlamlidir."""
    result: set[str] = set()
    for base, idxmap in groups.items():
        if not idxmap:
            continue
        first_col = idxmap[min(idxmap)]
        e = numeric_expr_fn(first_col)
        c = ident(first_col)
        try:
            row = con.execute(f"""
                WITH s AS (SELECT {c} AS v FROM {table} WHERE {_not_blank(c)} LIMIT {sample_limit})
                SELECT COUNT(*), COUNT({numeric_expr_fn('v')}) FROM s
            """).fetchone()
        except Exception:  # noqa: BLE001
            continue
        if not row or not row[0]:
            continue
        total, ok = row
        if (ok / total) >= min_ratio:
            result.add(base)
    return result


# ---------------------------------------------------------------- birlestirme (tek taban)
def build_combine_expr(idxmap: dict[int, str], agg: str, numeric_expr_fn) -> str:
    cols = [idxmap[i] for i in sorted(idxmap)]
    exprs = [f"COALESCE({numeric_expr_fn(c)}, 0)" for c in cols]
    if agg == "sum":
        return "(" + " + ".join(exprs) + ")"
    if agg == "avg":
        return "(" + " + ".join(exprs) + f") / {len(exprs)}.0"
    if agg == "max":
        return "GREATEST(" + ", ".join(exprs) + ")"
    if agg == "count_nonzero":
        return "(" + " + ".join(f"CASE WHEN {e} > 0 THEN 1 ELSE 0 END" for e in exprs) + ")"
    raise ValueError(f"bilinmeyen agregasyon: {agg}")


# ---------------------------------------------------------------- kisi bazli sayim (kategorik bayrak)
def build_flag_count_expr(idxmap: dict[int, str], positive_values: list[str]) -> str:
    """"cins", "yas", "ogrenci" gibi COKLU-TABANLI bir kisi/kayit blogunun
    tabani icin, HER SATIRDA (hanede/ankette) kac kisinin/kaydin secilen
    'pozitif' deger(ler)e sahip oldugunu SAYAN bir SQL ifadesi uretir (orn.
    "ogrenci_1..ogrenci_11" + pozitif=['Evet'] -> hane basina kac kisinin
    ogrenci oldugu).

    build_combine_expr'den farki: bu fonksiyon KATEGORIK (metin/kod) alanlar
    icindir - ham degerleri TOPLAMAK/ORTALAMASINI ALMAK anlamsiz olur (orn.
    cinsiyet kodu 1+2=3 gibi), ama 'kac kisi belirli bir degere sahip'
    SAYMAK anlamlidir.

    - KUMULATIF DEGILDIR: her satir, sadece KENDI sutunlarina bakarak
      bagimsiz hesaplanir; onceki/sonraki satirlara ya da calistirma
      sirasina bagli bir "running total" DEGILDIR.
    - BOS/doldurulmamis kisi yuvalari (NULL ya da bos metin - orn. 3 kisilik
      bir hanede kayitli olmayan 4. kisi) ne pozitif ne negatif sayilir;
      NULL IN (...) daima NULL/FALSE dondugunden toplam sayima hic
      KATILMAZLAR (bkz. _not_blank ile ayni mantik)."""
    cols = [idxmap[i] for i in sorted(idxmap)]
    vals_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in positive_values)
    parts = []
    for c in cols:
        norm = f"NULLIF(TRIM(CAST({ident(c)} AS VARCHAR)), '')"
        parts.append(f"(CASE WHEN {norm} IN ({vals_sql}) THEN 1 ELSE 0 END)")
    return "(" + " + ".join(parts) + ")"


# ---------------------------------------------------------------- tek seviyeli uzun format
def build_melt_view_sql(source_table: str, block: dict, id_columns: list[str],
                          view_name: str, index_col_name: str = "tekrar_no",
                          extra_cols: list[str] | None = None) -> tuple[str, list[str]]:
    """'single' (tek-indeksli) bir blogu (orn. yas_1..11, cins_1..11) UNION
    ALL ile uzun formata cevirir. Doner: (SQL, yeni tablodaki sutunlar).

    DOGRULUK ICIN KRITIK: `extra_cols` olmadan, kimlik sutunu(lar) DISINDAKI
    TUM "duz" (tekrarlamayan) hane/anket sutunlari - ilce, gelir, hanehalki
    buyuklugu gibi - uzun formata cevrildikten sonra KAYBOLURDU (cunku bu
    fonksiyon SADECE id_columns + blogun kendi tabanlarini tasir). Bu,
    kullanici 'Kisi/Yolculuk Tablosu'nu olusturur olusturmaz Ilce/Gelir gibi
    TEMEL kavramlarin sessizce "eslestirilemez" hale gelmesine yol aciyordu -
    gercek bir dosyada olculup dogrulandi (gelirtext/ilce kisi_uzun'da hic
    yoktu). `extra_cols` ile bu "duz" sutunlar da HER satirda (hanenin o
    satirdaki kisisi/yolculugu ne olursa olsun sabit degerleriyle) tasinir."""
    bases = block["bases"]
    indices = block["indices"]
    _base_set = set(bases)
    prefix_cols = list(id_columns or [])
    for c in (extra_cols or []):
        if c not in prefix_cols and c not in _base_set:
            prefix_cols.append(c)
    id_sel = ", ".join(ident(c) for c in prefix_cols) if prefix_cols else ""
    parts = []
    for idx in indices:
        cols_sel, filt_cols = [], []
        for base in bases:
            src_col = block["columns"][base].get(idx)
            if src_col is None:
                cols_sel.append(f"NULL AS {ident(base)}")
            else:
                cols_sel.append(f"{ident(src_col)} AS {ident(base)}")
                filt_cols.append(ident(src_col))
        prefix = (id_sel + ", ") if id_sel else ""
        where = (" WHERE " + " OR ".join(_not_blank(c) for c in filt_cols)) if filt_cols else ""
        parts.append(
            f"SELECT {prefix}{idx} AS {ident(index_col_name)}, {', '.join(cols_sel)} "
            f"FROM {source_table}{where}"
        )
    sql = f"CREATE OR REPLACE TABLE {ident(view_name)} AS " + " UNION ALL ".join(parts)
    new_cols = prefix_cols + [index_col_name] + bases
    return sql, new_cols


# ---------------------------------------------------------------- iki seviyeli uzun format (kisi x yolculuk)
def build_double_melt_view_sql(source_table: str, block: dict, id_columns: list[str],
                                 view_name: str, outer_name: str = "kisi_no",
                                 inner_name: str = "yolculuk_no",
                                 extra_cols: list[str] | None = None) -> tuple[str, list[str]]:
    """'double' (kisi x yolculuk gibi cift-indeksli) bir blogu, HER (i1, i2)
    kombinasyonu icin bir satir uretecek sekilde UNION ALL ile uzun formata
    cevirir. Cok sayida kombinasyon oldugundan (orn. 11 kisi x 10 yolculuk =
    110 SELECT) sorgu buyuk olabilir ama DuckDB bunu rahatlikla kaldirir;
    tamamen bos (tum alanlari NULL) kombinasyonlar satira donusturulmez.

    `extra_cols` icin bkz. build_melt_view_sql'deki ayni notu - ilce, gelir
    gibi "duz" hane sutunlarinin uzun formatta kaybolmamasini saglar."""
    bases = block["bases"]
    _base_set = set(bases)
    prefix_cols = list(id_columns or [])
    for c in (extra_cols or []):
        if c not in prefix_cols and c not in _base_set:
            prefix_cols.append(c)
    id_sel = ", ".join(ident(c) for c in prefix_cols) if prefix_cols else ""
    parts = []
    for i1 in block["outer_indices"]:
        for i2 in block["inner_indices"]:
            cols_sel, filt_cols = [], []
            for base in bases:
                src_col = block["columns"][base].get((i1, i2))
                if src_col is None:
                    cols_sel.append(f"NULL AS {ident(base)}")
                else:
                    cols_sel.append(f"{ident(src_col)} AS {ident(base)}")
                    filt_cols.append(ident(src_col))
            if not filt_cols:
                continue
            prefix = (id_sel + ", ") if id_sel else ""
            where = " WHERE " + " OR ".join(_not_blank(c) for c in filt_cols)
            parts.append(
                f"SELECT {prefix}{i1} AS {ident(outer_name)}, {i2} AS {ident(inner_name)}, "
                f"{', '.join(cols_sel)} FROM {source_table}{where}"
            )
    sql = f"CREATE OR REPLACE TABLE {ident(view_name)} AS " + " UNION ALL ".join(parts)
    new_cols = prefix_cols + [outer_name, inner_name] + bases
    return sql, new_cols


def build_join_sql(trip_view: str, person_view: str, id_columns: list[str],
                     outer_name: str, view_name: str, person_index_col: str = "kisi_no",
                     extra_person_exclude: list[str] | None = None) -> str:
    """Yolculuk (trip) uzun-format tablosunu, ayni kisiye ait demografik
    bilgilerle (kisi uzun-format tablosu) birlestirir - boylece her
    yolculuk satirinda o yolculugu yapan kisinin yas/cinsiyet/meslek gibi
    bilgileri de bulunur. extra_person_exclude: kisi tablosunda da AYNI
    isimle/degerle zaten var olan (orn. "_kisi_id" gibi turetilmis) ek
    sutunlari, cakisan ("duplicate column name") hata vermemesi icin kisi
    tarafindan disarida birakir - trip tarafindaki kopyasi (t.*) zaten
    yeterlidir."""
    join_cond = " AND ".join(
        f"t.{ident(c)} = p.{ident(c)}" for c in id_columns
    ) + f" AND t.{ident(outer_name)} = p.{ident(person_index_col)}"
    # DOGRULUK ICIN KRITIK: extra_person_exclude artik (plain_cols eklenmesiyle)
    # id_columns/person_index_col ile CAKISABILIR (orn. "id" hem id_columns'ta
    # hem plain_cols'ta) - DuckDB'nin EXCLUDE listesi AYNI sutunun IKI KEZ
    # gecmesine izin vermez ("Duplicate entry" hatasi verir). Sira KORUNARAK
    # (ilk gorulen kazanir) tekillestirilir.
    exclude_cols = list(dict.fromkeys(list(id_columns) + [person_index_col] + list(extra_person_exclude or [])))
    return (
        f"CREATE OR REPLACE TABLE {ident(view_name)} AS "
        f"SELECT t.*, p.* EXCLUDE ({', '.join(ident(c) for c in exclude_cols)}) "
        f"FROM {ident(trip_view)} AS t LEFT JOIN {ident(person_view)} AS p ON {join_cond}"
    )


def humanize_single_label(block: dict) -> str:
    idx_range = f"{min(block['indices'])}-{max(block['indices'])}"
    if len(block["bases"]) == 1:
        return f"{block['bases'][0]}[{idx_range}]"
    return ", ".join(block["bases"][:4]) + (" ..." if len(block["bases"]) > 4 else "") + f"  ({idx_range})"


def humanize_double_label(block: dict) -> str:
    o = f"{min(block['outer_indices'])}-{max(block['outer_indices'])}"
    i = f"{min(block['inner_indices'])}-{max(block['inner_indices'])}"
    bases_preview = ", ".join(block["bases"][:3]) + (" ..." if len(block["bases"]) > 3 else "")
    return f"{bases_preview}  (kisi {o} x yolculuk {i})"
