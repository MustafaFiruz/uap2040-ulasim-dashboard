# -*- coding: utf-8 -*-
"""
UAP 2040 rapor mantigini SQL (DuckDB) uzerinden uygulayan hesaplama katmani.

ONEMLI: Bu modulde HICBIR fonksiyon tum veriyi pandas'a cekmez. Her fonksiyon
DuckDB'ye bir agregasyon sorgusu gonderir; DuckDB dosyadaki (kalici .duckdb
tablosundaki) TUM SATIRLARI tarar, yalnizca KUCUK ozet sonucu (ornegin
20 ilce x birkac sutun) pandas'a doner. Bu sayede sonuclar veri boyutundan
bagimsiz olarak her zaman TAM VERIYE gore dogrudur ve buyuk veri de hizli
calisir (bellege yuk binmez).
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

AGE_BIN_SIZE = 5


def ident(col: str) -> str:
    """SQL tanimlayici (sutun adi) guvenli sekilde tirnaklanir."""
    return '"' + str(col).replace('"', '""') + '"'


def cat_expr(col: str) -> str:
    """Kategorik (metin) bir sutunu, BOS ya da SADECE-BOSLUKLU degerleri de
    NULL sayacak sekilde normallestirir.

    DOGRULUK ICIN KRITIK: CSV/Excel'lerde eksik bir hucre her zaman gercek
    NULL olarak okunmaz - bazen '' (bos metin) ya da ' ' gibi bosluk(lar)
    olarak okunur. Bu fonksiyon olmadan 'WHERE col IS NOT NULL' bu tur
    degerleri YAKALAYAMAZ ve bos deger, pasta/çubuk grafiklerde kendi basina
    ayri, anlamsiz bir 'kategori' olarak gorunur (orn. cinsiyet grafiginde
    3. bir dilim cikmasi). Tum kategorik SELECT/WHERE/GROUP BY ifadelerinde
    ham ident(col) yerine bu fonksiyon kullanilmalidir."""
    c = f"CAST({ident(col)} AS VARCHAR)"
    return f"NULLIF(TRIM({c}), '')"


def numeric_expr(col: str) -> str:
    """Sutunu sayisala cevirir; Turkce bicimli sayilari da ('1.234,56' ya da
    '12,5') dogru sekilde ayristirir. Herhangi bir Python tarafinda tum
    veriyi okumaya gerek kalmadan, tamamen SQL icinde calisir.

    DOGRULUK ICIN KRITIK (bir dosyada olculup dogrulandi): ESKI kural, TEK
    bir ".XXX" (3 haneli) grubu goren HER degeri "Turkce binlik ayiraci"
    sanip yanlislikla noktayi siliyordu - orn. "0.523" (gercekte GERCEKTEN
    ondalik bir kesir, orn. bir anket agirlik/genisletme katsayisi) yanlislikla
    "523" olarak okunuyordu (0.523 -> 523, ~900 kat buyuk bir hata). Bu,
    "12.345.678" (IKI ayri nokta grubu - Turkce binlik oldugu SUPHESIZ) ya
    da "1.234,56" (nokta grubu + virgullu ondalik - yine supheye yer yok)
    gibi GERCEKTEN BELIRSIZ OLMAYAN durumlarla karistirilamaz. Artik SADECE
    bu iki supheye yer birakmayan durumda "binlik ayiraci" varsayilir; TEK
    bir nokta grubu (virgul olmadan, orn. "1.234" ya da "0.523") ARTIK
    normal bir ondalik sayi olarak (noktasi SILINMEDEN) okunur - bu, agirlik
    katsayisi gibi 2-4 ondalikli GERCEK kesirli degerler icin sarttir."""
    c = f"CAST({ident(col)} AS VARCHAR)"
    return (
        f"TRY_CAST(CASE "
        f"WHEN regexp_matches({c}, '^-?[0-9]{{1,3}}(\\.[0-9]{{3}}){{2,}}(,[0-9]+)?$') "
        f"OR regexp_matches({c}, '^-?[0-9]{{1,3}}(\\.[0-9]{{3}})+,[0-9]+$') "
        f"THEN replace(replace({c}, '.', ''), ',', '.') "
        f"WHEN regexp_matches({c}, '^-?[0-9]+,[0-9]+$') "
        f"THEN replace({c}, ',', '.') "
        f"ELSE {c} END AS DOUBLE)"
    )


def has(mapping: dict, *concepts: str) -> bool:
    return all(mapping.get(c) for c in concepts)


def _with_weight(mapping: dict, cols: list[str]) -> list[str]:
    """Bir _dedup_source cagrisinin extra_cols listesine, mapping'de bir
    'weight' (anket agirligi/genisletme katsayisi) sutunu tanimliysa onu da
    ekler. KRITIK: agirlik sutunu _dedup_source SONRASI (src tablosunda) da
    bulunmazsa, agirlikli hesaplama 'sutun bulunamadi' hatasi verir - bu
    yuzden agirlik kullanilacak HER yerde ayni anda dedup'a da eklenmelidir.
    weight tanimli degilse listeyi degistirmeden dondurur (geriye donuk
    uyumlu - agirliksiz akista HICBIR SEY degismez)."""
    w = mapping.get("weight")
    return cols + [w] if w and w not in cols else cols


def _dedup_source(table: str, key_col: str | None, extra_cols: list[str], alias: str = "_dedup") -> str:
    """DOGRULUK ICIN KRITIK: kisi/hane bazli bir ozelligi (cinsiyet, yas,
    gelir, arac sayisi gibi) TEKRARLAYAN kayitlar uzerinden (orn. her
    yolculugun ayri bir satir oldugu bir tabloda) dogrudan saymak, cok
    yolculuk yapan kisi/haneleri istatistige fazla agirlikla katar ve
    dagilimi carpitir. key_col (orn. person_id/household_id) eslenmisse,
    bu fonksiyon o anahtara gore TEKRARSIZ bir alt sorgu dondurerek her
    kisi/hane yalnizca BIR kez sayilmasini garanti eder.

    ONEMLI: SELECT DISTINCT degil, GROUP BY key_col + ANY_VALUE(...) kullanilir.
    Karmasik/gercek dunya verilerinde ayni kisinin farkli yolculuk satirlarinda
    'ilce' gibi bir alan (orn. yolculuk baslangic ilcesi ev disi bir yolculukta
    farkli kaydedilmisse) TUTARSIZ olabilir - SELECT DISTINCT boyle durumlarda
    o kisi icin BIRDEN FAZLA satir uretip kisiyi fazladan sayardi. GROUP BY +
    ANY_VALUE, tutarsizlik olsa bile HER ZAMAN tam olarak bir satir garanti
    eder (rasgele ama tutarli bir deger secerek), boylece sayim asla
    sismez/carpitilmaz. Anahtar eslenmemisse tablo oldugu gibi kullanilir
    (eski davranisla ayni, geriye donuk uyumlu)."""
    if not key_col:
        return table
    if not extra_cols:
        return f"(SELECT DISTINCT {ident(key_col)} FROM {table}) AS {alias}"
    agg_cols = ", ".join(f"ANY_VALUE({ident(c)}) AS {ident(c)}" for c in extra_cols)
    return f"(SELECT {ident(key_col)}, {agg_cols} FROM {table} GROUP BY {ident(key_col)}) AS {alias}"


def _fetch(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).fetchdf()


# ---------------------------------------------------------------- genel profil
MAX_PROFILE_COLS = 300  # cok genis (binlerce sutunlu) dosyalarda profillemeyi sinirla


def data_profile(con: duckdb.DuckDBPyConnection, table: str, total_rows: int,
                  columns: list[str], dtypes: dict[str, str],
                  dup_check_limit: int = 3_000_000, dup_check_col_limit: int = 500,
                  numeric_sample_n: int = 2000) -> dict:
    # Cok genis tablolarda (orn. 5000+ sutun) TEK sorguda binlerce SUM(CASE..)
    # ifadesi olusturmak hem SQL derlemesini hem agi ciddi sekilde yavaslatir.
    # Bu yuzden profilleme en fazla MAX_PROFILE_COLS sutunla sinirlanir.
    profiled_cols = columns[:MAX_PROFILE_COLS]
    truncated = len(columns) > MAX_PROFILE_COLS

    # "Sayisal sutun" tespiti DuckDB'nin sakladigi tipe DEGIL, sutunun
    # GERCEK ICERIGINE bakilarak yapilir: genis (>WIDE_FILE_COL_THRESHOLD
    # sutunlu) dosyalar hiz icin TAMAMEN METIN (VARCHAR) olarak okunuyor
    # (bkz. data_io._try_ingest), yani DuckDB'nin dtype bilgisi bu durumda
    # her sutun icin 'VARCHAR' doner ve gercekte sayisal olan sutunlari da
    # kategorik gosterir. Kucuk bir ornek (LIMIT) uzerinde numeric_expr ile
    # gercek doluluk oranina bakmak, dosyanin nasil okundugundan bagimsiz
    # olarak dogru sonuc verir.
    numeric_types = {"BIGINT", "DOUBLE", "INTEGER", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT"}
    dtype_numeric = {c for c in profiled_cols if any(t in dtypes.get(c, "") for t in numeric_types)}
    content_check_cols = [c for c in profiled_cols if c not in dtype_numeric]
    numeric_cols = set(dtype_numeric)
    if content_check_cols:
        exprs = ", ".join(
            f"COUNT({ident(c)}) AS {ident('t_'+str(i))}, COUNT({numeric_expr(c)}) AS {ident('n_'+str(i))}"
            for i, c in enumerate(content_check_cols)
        )
        row0 = con.execute(f"SELECT {exprs} FROM (SELECT * FROM {table} LIMIT {numeric_sample_n})").fetchone()
        for i, c in enumerate(content_check_cols):
            tot, numok = row0[2 * i], row0[2 * i + 1]
            if tot and numok / tot >= 0.9:
                numeric_cols.add(c)
    numeric_cols = [c for c in profiled_cols if c in numeric_cols]
    cat_cols = [c for c in columns if c not in numeric_cols]

    exprs = ", ".join(
        f"SUM(CASE WHEN {ident(c)} IS NULL THEN 1 ELSE 0 END) AS {ident('m_'+str(i))}"
        for i, c in enumerate(profiled_cols)
    )
    row = con.execute(f"SELECT {exprs} FROM {table}").fetchone()
    missing_pct = pd.Series(
        {c: (row[i] / total_rows * 100 if total_rows else 0) for i, c in enumerate(profiled_cols)}
    ).sort_values(ascending=False)

    # Mukerrer satir kontrolu de sutun sayisi cok fazlaysa (DISTINCT * pahali
    # hale gelir) atlanir - satir sayisina gore olan mevcut sinir yeterli degil.
    dup_rows = None
    if total_rows <= dup_check_limit and len(columns) <= dup_check_col_limit:
        distinct_n = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {table})").fetchone()[0]
        dup_rows = total_rows - distinct_n

    return {
        "profiled_cols_truncated": truncated,
        "profiled_col_count": len(profiled_cols),
        "n_rows": total_rows, "n_cols": len(columns), "missing_pct": missing_pct,
        "numeric_cols": numeric_cols, "cat_cols": cat_cols, "dup_rows": dup_rows,
    }


# ---------------------------------------------------------------- genel agregasyon yardimcilari
def value_counts_sql(con, table, col: str, limit: int = 60, weight_col: str | None = None) -> pd.Series | None:
    """weight_col verilirse (anket agirligi/genisletme katsayisi), her satir
    KENDI agirligiyla sayilir (COUNT(*) yerine SUM(agirlik)) - boylece
    orneklemde kasitli olarak fazla/az temsil edilen gruplar duzeltilerek
    GERCEK nufusu yansitan bir yuzde elde edilir. weight_col=None ise
    (varsayilan - agirlik hic eslenmemisse) davranis ONCEKI ile BIREBIR
    AYNIDIR - hicbir mevcut sonuc degismez."""
    if not col:
        return None
    e = cat_expr(col)
    if weight_col:
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            SELECT {e} AS cat, SUM({we}) AS n
            FROM {table} WHERE {e} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT {limit}
        """)
    else:
        df = _fetch(con, f"""
            SELECT {e} AS cat, COUNT(*) AS n
            FROM {table} WHERE {e} IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT {limit}
        """)
    if df.empty:
        return None
    total = df["n"].sum()
    return pd.Series((df["n"] / total * 100).values, index=df["cat"].astype(str), name="pct")


def raw_count_sql(con, table, col: str, limit: int = 60, weight_col: str | None = None) -> pd.Series | None:
    """value_counts_sql ile ayni ama YUZDE degil HAM SAYI (ya da weight_col
    verilmisse AGIRLIKLI TOPLAM - gercek nufusa genisletilmis tahmini sayi)
    dondurur (orn. 'Tablo 5. Ilcelere Gore Ogrenci Nufusu' gibi rapor
    tablolarinda oran degil dogrudan kisi/kayit sayisi gosterilir)."""
    if not col:
        return None
    e = cat_expr(col)
    if weight_col:
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            SELECT {e} AS cat, SUM({we}) AS n
            FROM {table} WHERE {e} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT {limit}
        """)
    else:
        df = _fetch(con, f"""
            SELECT {e} AS cat, COUNT(*) AS n
            FROM {table} WHERE {e} IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT {limit}
        """)
    if df.empty:
        return None
    return pd.Series(df["n"].values, index=df["cat"].astype(str), name="adet")


def crosstab_sql(con, table, row_col: str, col_col: str, normalize: str = "index",
                  row_limit: int = 60, col_limit: int = 30, weight_col: str | None = None) -> pd.DataFrame | None:
    if not row_col or not col_col:
        return None
    re_, ce_ = cat_expr(row_col), cat_expr(col_col)
    if weight_col:
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            SELECT {re_} AS r, {ce_} AS c, SUM({we}) AS n
            FROM {table}
            WHERE {re_} IS NOT NULL AND {ce_} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1, 2
        """)
    else:
        df = _fetch(con, f"""
            SELECT {re_} AS r, {ce_} AS c, COUNT(*) AS n
            FROM {table}
            WHERE {re_} IS NOT NULL AND {ce_} IS NOT NULL
            GROUP BY 1, 2
        """)
    if df.empty:
        return None
    # DOGRULUK ICIN KRITIK (bir stres testinde bulundu/dogrulandi): satir/
    # sutun sayisi row_limit/col_limit'i asarsa (orn. 30'dan fazla ulasim
    # turu kategorisi), yuzdeyi ONCE row_limit/col_limit'e KESIP SONRA
    # normalize etmek, sadece GOSTERILEN alt kumenin toplamini payda
    # yapardi - bu da yuzdeleri YANLISLIKLA sisirirdi (confidence_
    # interval_proportion'da bulunan/duzeltilen ayni hata deseni - bkz. o
    # fonksiyonun docstring'i). Bu yuzden normalize islemi HER ZAMAN
    # TAM (kesilmemis) tabloda yapilir; row_limit/col_limit SADECE en
    # SONDA, normal hesaplanmis degerlerden hangi satir/sutunlarin
    # GORUNTULENECEGINI secmek icin kullanilir - degerlerin kendisini
    # DEGISTIRMEZ.
    wide = df.pivot_table(index="r", columns="c", values="n", fill_value=0)
    if normalize == "index":
        wide = wide.div(wide.sum(axis=1), axis=0) * 100
    elif normalize == "columns":
        wide = wide.div(wide.sum(axis=0), axis=1) * 100
    top_rows = df.groupby("r")["n"].sum().sort_values(ascending=False).head(row_limit).index
    top_cols = df.groupby("c")["n"].sum().sort_values(ascending=False).head(col_limit).index
    wide = wide.loc[wide.index.isin(top_rows), wide.columns.isin(top_cols)]
    wide.index.name = None
    wide.columns.name = None
    return wide


def numeric_summary_sql(con, table, col: str, weight_col: str | None = None) -> dict | None:
    """weight_col verilirse AGIRLIKLI ortalama/medyan/std hesaplar (bkz.
    value_counts_sql docstring'indeki ayni mantik). Agirlikli ortalama:
    SUM(deger*agirlik)/SUM(agirlik). Agirlikli medyan: degerler agirliga
    gore siralanip KUMULATIF agirligin toplam agirligin yarisini gectigi
    ilk deger alinir (standart 'weighted median' tanimi). weight_col=None
    ise (varsayilan) SQL ONCEKI ile BIREBIR AYNIDIR - hicbir mevcut sonuc
    degismez."""
    if not col:
        return None
    e = numeric_expr(col)
    if not weight_col:
        row = con.execute(f"""
            SELECT AVG({e}), MIN({e}), MAX({e}), MEDIAN({e}), STDDEV_POP({e}), COUNT({e})
            FROM {table}
        """).fetchone()
        if row is None or row[5] == 0:
            return None
        keys = ["ortalama", "min", "maks", "medyan", "std", "n"]
        return dict(zip(keys, row))
    we = numeric_expr(weight_col)
    row = con.execute(f"""
        WITH v AS (
            SELECT {e} AS x, {we} AS w FROM {table}
            WHERE {e} IS NOT NULL AND {we} IS NOT NULL AND {we} > 0
        ),
        agg AS (
            SELECT SUM(x * w) / SUM(w) AS wmean, MIN(x) AS mn, MAX(x) AS mx,
                   SUM(w) AS totw, COUNT(*) AS n
            FROM v
        ),
        var_calc AS (
            SELECT SUM(v.w * POWER(v.x - agg.wmean, 2)) / ANY_VALUE(agg.totw) AS wvar
            FROM v, agg
        ),
        ranked AS (
            SELECT x, SUM(w) OVER (ORDER BY x) AS cum_w FROM v
        ),
        med AS (
            SELECT MIN(x) AS wmedian FROM ranked, agg WHERE ranked.cum_w >= agg.totw / 2.0
        )
        SELECT agg.wmean, agg.mn, agg.mx, med.wmedian, SQRT(var_calc.wvar), agg.n
        FROM agg, var_calc, med
    """).fetchone()
    if row is None or row[5] == 0:
        return None
    keys = ["ortalama", "min", "maks", "medyan", "std", "n"]
    return dict(zip(keys, row))


def numeric_by_group_sql(con, table, group_col: str, value_col: str,
                          label_group: str = "Grup", weight_col: str | None = None) -> pd.DataFrame | None:
    if not group_col or not value_col:
        return None
    e = numeric_expr(value_col)
    ge = cat_expr(group_col)
    if not weight_col:
        df = _fetch(con, f"""
            SELECT {ge} AS grp, AVG({e}) AS ortalama, MEDIAN({e}) AS medyan,
                   STDDEV_POP({e}) AS std, COUNT({e}) AS n
            FROM {table}
            WHERE {ge} IS NOT NULL AND {e} IS NOT NULL
            GROUP BY 1 ORDER BY ortalama DESC
        """)
    else:
        # Ayni agirlikli ortalama/medyan/std mantigi (bkz. numeric_summary_sql),
        # burada HER GRUP (orn. ilce) icin AYRI AYRI (PARTITION BY grp) hesaplanir.
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            WITH v AS (
                SELECT {ge} AS grp, {e} AS x, {we} AS w FROM {table}
                WHERE {ge} IS NOT NULL AND {e} IS NOT NULL AND {we} IS NOT NULL AND {we} > 0
            ),
            gagg AS (
                SELECT grp, SUM(x * w) / SUM(w) AS ortalama, SUM(w) AS totw, COUNT(*) AS n
                FROM v GROUP BY grp
            ),
            gvar AS (
                SELECT v.grp, SUM(v.w * POWER(v.x - gagg.ortalama, 2)) / ANY_VALUE(gagg.totw) AS wvar
                FROM v JOIN gagg ON v.grp = gagg.grp
                GROUP BY v.grp
            ),
            granked AS (
                SELECT grp, x, SUM(w) OVER (PARTITION BY grp ORDER BY x) AS cum_w FROM v
            ),
            gmed AS (
                SELECT granked.grp, MIN(granked.x) AS medyan
                FROM granked JOIN gagg ON granked.grp = gagg.grp
                WHERE granked.cum_w >= gagg.totw / 2.0
                GROUP BY granked.grp
            )
            SELECT gagg.grp AS grp, gagg.ortalama, gmed.medyan, SQRT(gvar.wvar) AS std, gagg.n
            FROM gagg JOIN gmed ON gagg.grp = gmed.grp JOIN gvar ON gagg.grp = gvar.grp
            ORDER BY gagg.ortalama DESC
        """)
    if df.empty:
        return None
    df = df.rename(columns={
        "grp": label_group, "ortalama": "Ortalama", "medyan": "Medyan",
        "std": "Std. Sapma", "n": "Adet",
    })
    return df


# ---------------------------------------------------------------- hanehalki buyuklugu
def household_size_stats(con, table, mapping: dict) -> pd.DataFrame | None:
    if mapping.get("household_size"):
        # hanehalki_buyuklugu HANE bazli bir deger - household_id eslenmisse
        # tekrarsiz sayilir (aksi halde yolculuk-bazli bir tabloda ayni
        # hanenin degeri her yolculukta bir kez daha sayilirdi).
        src = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [mapping["household_size"]]))
        s = numeric_summary_sql(con, src, mapping["household_size"], weight_col=mapping.get("weight"))
        if s is None:
            return None
        return pd.DataFrame({
            "Ortalama": [s["ortalama"]], "Minimum": [s["min"]], "Maksimum": [s["maks"]],
            "Medyan": [s["medyan"]], "Std. Sapma": [s["std"]], "Gozlem": [s["n"]],
        })
    if has(mapping, "household_id", "person_id"):
        hh, pid = ident(mapping["household_id"]), ident(mapping["person_id"])
        row = con.execute(f"""
            WITH hh_sizes AS (
                SELECT {hh} AS h, COUNT(DISTINCT {pid}) AS sz
                FROM {table} WHERE {hh} IS NOT NULL GROUP BY 1
            )
            SELECT AVG(sz), MIN(sz), MAX(sz), MEDIAN(sz), STDDEV_POP(sz), COUNT(*)
            FROM hh_sizes
        """).fetchone()
        if row is None or row[5] == 0:
            return None
        return pd.DataFrame({
            "Ortalama": [row[0]], "Minimum": [row[1]], "Maksimum": [row[2]],
            "Medyan": [row[3]], "Std. Sapma": [row[4]], "Gozlem (Hane)": [row[5]],
        })
    return None


def household_size_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    if not mapping.get("district"):
        return None
    dist = cat_expr(mapping["district"])
    if mapping.get("household_size"):
        src = _dedup_source(table, mapping.get("household_id"),
                            _with_weight(mapping, [mapping["district"], mapping["household_size"]]))
        df = numeric_by_group_sql(con, src, mapping["district"], mapping["household_size"],
                                   "Ilce", weight_col=mapping.get("weight"))
        if df is None:
            return None
        return df.rename(columns={"Ortalama": "Ortalama Hanehalki Buyuklugu"})[
            ["Ilce", "Ortalama Hanehalki Buyuklugu"]].sort_values(
            "Ortalama Hanehalki Buyuklugu", ascending=False)
    if has(mapping, "household_id", "person_id"):
        hh, pid = ident(mapping["household_id"]), ident(mapping["person_id"])
        df = _fetch(con, f"""
            WITH hh_sizes AS (
                SELECT {dist} AS Ilce, {hh} AS h, COUNT(DISTINCT {pid}) AS sz
                FROM {table} WHERE {dist} IS NOT NULL AND {hh} IS NOT NULL GROUP BY 1, 2
            )
            SELECT Ilce, AVG(sz) AS "Ortalama Hanehalki Buyuklugu"
            FROM hh_sizes GROUP BY 1 ORDER BY 2 DESC
        """)
        return df if not df.empty else None
    return None


# ---------------------------------------------------------------- cinsiyet & yas
# NOT: cinsiyet/yas/egitim gibi ozellikler KISI bazlidir. Aktif tablo
# yolculuk-bazli (1 satir = 1 yolculuk) olabildigi icin, person_id
# eslenmisse asagidaki fonksiyonlar otomatik olarak kisi bazinda TEKRARSIZ
# hesaplama yapar - aksi halde cok yolculuk yapan kisiler dagilimi yanlis
# agirliklandirirdi (bkz. _dedup_source).
def gender_distribution(con, table, mapping: dict) -> pd.Series | None:
    col = mapping.get("gender")
    if not col:
        return None
    src = _dedup_source(table, mapping.get("person_id"), _with_weight(mapping, [col]))
    return value_counts_sql(con, src, col, weight_col=mapping.get("weight"))


def gender_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    if not has(mapping, "gender", "district"):
        return None
    src = _dedup_source(table, mapping.get("person_id"),
                        _with_weight(mapping, [mapping["district"], mapping["gender"]]))
    ct = crosstab_sql(con, src, mapping["district"], mapping["gender"], weight_col=mapping.get("weight"))
    if ct is None:
        return None
    return ct.reset_index().rename(columns={"index": "Ilce"})


def age_pyramid(con, table, mapping: dict) -> pd.DataFrame | None:
    if not has(mapping, "age", "gender"):
        return None
    weight_col = mapping.get("weight")
    src = _dedup_source(table, mapping.get("person_id"), _with_weight(mapping, [mapping["age"], mapping["gender"]]))
    age_e = numeric_expr(mapping["age"])
    gender = cat_expr(mapping["gender"])
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    where_extra = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT CAST(FLOOR({age_e} / {AGE_BIN_SIZE}) * {AGE_BIN_SIZE} AS INTEGER) AS age_bin,
               {gender} AS gender, {n_expr} AS n
        FROM {src}
        WHERE {age_e} IS NOT NULL AND {age_e} BETWEEN 0 AND 120 AND {gender} IS NOT NULL{where_extra}
        GROUP BY 1, 2
    """)
    if df.empty:
        return None
    df["age_bin"] = df["age_bin"].clip(upper=85)
    df["age_group"] = df["age_bin"].apply(lambda a: "85+" if a >= 85 else f"{a}-{a+4}")
    pivot = df.groupby(["age_group", "gender"])["n"].sum().unstack(fill_value=0)
    order = [f"{a}-{a+4}" for a in range(0, 85, AGE_BIN_SIZE)] + ["85+"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    pivot_pct = pivot / pivot.values.sum() * 100
    return pivot_pct.reset_index()


def avg_age(con, table, mapping: dict) -> float | None:
    col = mapping.get("age")
    if not col:
        return None
    src = _dedup_source(table, mapping.get("person_id"), _with_weight(mapping, [col]))
    s = numeric_summary_sql(con, src, col, weight_col=mapping.get("weight"))
    return s["ortalama"] if s else None


def age_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    """Ilce bazinda ortalama yas (Rapor Grafik 7)."""
    if not has(mapping, "age", "district"):
        return None
    src = _dedup_source(table, mapping.get("person_id"),
                        _with_weight(mapping, [mapping["age"], mapping["district"]]))
    df = numeric_by_group_sql(con, src, mapping["district"], mapping["age"], "Ilce", weight_col=mapping.get("weight"))
    if df is None:
        return None
    return df.rename(columns={"Ortalama": "Ortalama Yas"})[["Ilce", "Ortalama Yas"]].sort_values(
        "Ortalama Yas", ascending=False)


def age_group_distribution(con, table, mapping: dict, bin_edges: tuple = (0, 15, 25, 45, 65, 120),
                            bin_labels: tuple = ("0-14", "15-24", "25-44", "45-64", "65+"),
                            filter_concept: str | None = None,
                            filter_positive_values: list[str] | None = None) -> pd.Series | None:
    """Yas gruplarina gore kisi dagilimi (%). filter_concept/filter_positive_values
    verilirse (orn. concept='employment', values=['Calisan']), sadece o
    kategoriye uyan kisiler sayilir (Rapor Grafik 21: 'Calisan Nufusun Yas
    Gruplarina Dagilimi')."""
    if not mapping.get("age"):
        return None
    weight_col = mapping.get("weight")
    extra_cols = _with_weight(mapping, [mapping["age"]])
    if filter_concept and mapping.get(filter_concept):
        extra_cols.append(mapping[filter_concept])
    src = _dedup_source(table, mapping.get("person_id"), extra_cols)
    age_e = numeric_expr(mapping["age"])
    where_extra = ""
    if filter_concept and mapping.get(filter_concept) and filter_positive_values:
        fcol = cat_expr(mapping[filter_concept])
        vals_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in filter_positive_values)
        where_extra = f" AND {fcol} IN ({vals_sql})"
    if weight_col:
        where_extra += f" AND {numeric_expr(weight_col)} IS NOT NULL"
    edges_sql = ", ".join(str(e) for e in bin_edges)
    case_parts = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        case_parts.append(f"WHEN {age_e} >= {lo} AND {age_e} < {hi} THEN '{bin_labels[i]}'")
    case_sql = "CASE " + " ".join(case_parts) + " END"
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    df = _fetch(con, f"""
        SELECT {case_sql} AS grp, {n_expr} AS n
        FROM {src} WHERE {age_e} IS NOT NULL{where_extra}
        GROUP BY 1
    """)
    if df.empty:
        return None
    df = df.dropna(subset=["grp"])
    total = df["n"].sum()
    if not total:
        return None
    s = pd.Series(df.set_index("grp")["n"] / total * 100)
    return s.reindex([b for b in bin_labels if b in s.index])


# ---------------------------------------------------------------- kategorik dagilimlar
def category_distribution(con, table, mapping: dict, concept: str, dedup_by: str | None = None,
                           group_siblings: dict[str, list[str]] | None = None) -> pd.Series | None:
    """dedup_by: 'person_id' ya da 'household_id' verilirse (ve mapping'de
    o kavram eslenmisse), o anahtara gore tekrarsiz sayilir. Amac/mod gibi
    dogal olarak yolculuk-bazli kavramlar icin dedup_by verilmemelidir.

    group_siblings: DOGRULUK ICIN KRITIK (gercek bir dosyada olculup
    dogrulandi - "Arac Park Yeri" icin ~%11 eksik cevap sayimi). Eslenen
    sutun ('col') henuz melt EDILMEMIS, coklu-sutunlu bir tekrar grubunun
    (orn. arac basina bir sutun tutan 'parkyeri_1..parkyeri_8') sadece TEK
    bir indeksiyse, o tek sutunu kullanmak diger araclarin/kayitlarin
    (2., 3., ... arac) cevaplarini SESSIZCE gormezden gelir - kullaniciya
    hicbir uyari verilmeden dagilim yanlis/eksik hesaplanir. Bu sozluk
    (app.py'de single_groups'tan turetilir) o sutunun TUM kardeslerini
    verirse, hepsi UNION ALL ile havuzlanip TEK bir dagilim olarak
    sayilir. Grup tespit edilmezse (col bu sozlukte yoksa ya da grup 1
    sutunluysa) davranis ONCEKI ile BIREBIR AYNIDIR - mevcut hicbir sonuc
    degismez."""
    col = mapping.get(concept)
    if not col:
        return None
    weight_col = mapping.get("weight")
    dedup_col = mapping.get(dedup_by) if dedup_by else None
    siblings = (group_siblings or {}).get(col)
    # GUVENLIK ICIN KRITIK: dedup_col'un KENDISI de (grup_siblings'e gore)
    # coklu-sutunlu bir tekrar grubunun bir parcasiysa (orn. "kisi_no_1"),
    # onu HER UNION dalinda AYNEN tekrar kullanmak yanlis olurdu - her dal
    # FARKLI bir kisiyi/kaydi (egit_2, egit_3, ...) temsil ederken, dedup
    # anahtari HEP AYNI (orn. hep "kisi_no_1") kalir ve bu, farkli kisileri
    # ayni kimlige sahipmis gibi BIRBIRINE KARISTIRIP sessizce yanlis
    # tekillestirmeye yol acar. Boyle bir uyusmazlik riski varsa havuzlama
    # ATLANIR (guvenli, ONCEKI davranisla BIREBIR AYNI kalir) - bu concept
    # icin dogru havuzlama, index'e hizali (zip) bir eslestirme gerektirir
    # ve bu fonksiyonun kapsami disindadir.
    if dedup_col and dedup_col in (group_siblings or {}):
        siblings = None
    if siblings and len(siblings) > 1:
        extra_sel = (f", {ident(weight_col)}" if weight_col else "") + \
                    (f", {ident(dedup_col)}" if dedup_col else "")
        union_sql = " UNION ALL ".join(
            f"SELECT {cat_expr(sib)} AS {ident(col)}{extra_sel} FROM {table}" for sib in siblings
        )
        table = f"({union_sql}) AS _pooled"
    src = _dedup_source(table, dedup_col, _with_weight(mapping, [col]))
    return value_counts_sql(con, src, col, weight_col=weight_col)


def category_by_district(con, table, mapping: dict, concept: str, dedup_by: str | None = None,
                          group_siblings: dict[str, list[str]] | None = None) -> pd.DataFrame | None:
    """group_siblings: bkz. category_distribution'daki ayni parametrenin
    commit notu - eslenen sutun ('mapping[concept]') melt edilmemis coklu-
    sutunlu bir tekrar grubunun tek bir indeksiyse, TUM kardesler (ilce
    bilgisiyle birlikte) havuzlanir; aksi halde davranis ONCEKI ile
    BIREBIR AYNIDIR."""
    if not has(mapping, concept, "district"):
        return None
    col = mapping[concept]
    dist_col = mapping["district"]
    weight_col = mapping.get("weight")
    dedup_col = mapping.get(dedup_by) if dedup_by else None
    siblings = (group_siblings or {}).get(col)
    # GUVENLIK ICIN KRITIK: bkz. category_distribution'daki ayni notu -
    # dedup_col YA DA dist_col (ilce) kendisi de bir tekrar grubunun
    # parcasiysa, o TEK sutunu her UNION dalinda aynen tekrar kullanmak
    # yanlis hizalamaya yol acar; boyle bir durumda havuzlama guvenli
    # sekilde ATLANIR (ONCEKI davranis korunur).
    if (dedup_col and dedup_col in (group_siblings or {})) or \
       (dist_col in (group_siblings or {})):
        siblings = None
    if siblings and len(siblings) > 1:
        extra_sel = (f", {ident(weight_col)}" if weight_col else "") + \
                    (f", {ident(dedup_col)}" if dedup_col else "")
        union_sql = " UNION ALL ".join(
            f"SELECT {cat_expr(dist_col)} AS {ident(dist_col)}, {cat_expr(sib)} AS {ident(col)}{extra_sel} "
            f"FROM {table}" for sib in siblings
        )
        table = f"({union_sql}) AS _pooled"
    src = _dedup_source(table, dedup_col, _with_weight(mapping, [dist_col, col]))
    ct = crosstab_sql(con, src, dist_col, col, weight_col=weight_col)
    if ct is None:
        return None
    return ct.reset_index().rename(columns={"index": "Ilce"})


# ---------------------------------------------------------------- gelir / esitsizlik
# NOT: gelir HANE bazli bir degerdir. household_id eslenmisse, asagidaki
# fonksiyonlar hane bazinda TEKRARSIZ hesaplama yapar - aksi halde (orn.
# yolculuk-bazli bir tabloda) cok yolculuk yapan hanelerin geliri fazla
# sayilir ve ozellikle Gini katsayisi ciddi sekilde yanlis cikar.
def income_summary(con, table, mapping: dict) -> dict | None:
    col = mapping.get("income")
    if not col:
        return None
    src = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [col]))
    return numeric_summary_sql(con, src, col, weight_col=mapping.get("weight"))


def gini_and_lorenz(con, table, mapping: dict, n_buckets: int = 200) -> tuple[float, pd.DataFrame] | None:
    """Gini katsayisini TAM veri uzerinde (SQL ile, satirlari Python'a
    cekmeden) hesaplar; Lorenz egrisi icin ise gorsellestirmeye yetecek
    kadar (n_buckets) noktaya indirgenmis kumulatif dagilim doner.

    weight_col (mapping['weight']) verilirse AGIRLIKLI Gini/Lorenz hesaplanir
    - her hane KENDI agirligiyla nufusa/gelire katkida bulunur. Kullanilan
    agirlikli Gini formulu, w_i=1 icin ASAGIDAKI agirliksiz kapali-form
    formule CEBIRSEL OLARAK INDIRGENIR (elle turetilip dogrulandi) - yani
    weight_col=None oldugunda (varsayilan) sonuc ONCEKI ile BIREBIR AYNIDIR."""
    if not mapping.get("income"):
        return None
    weight_col = mapping.get("weight")
    table = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [mapping["income"]]))
    e = numeric_expr(mapping["income"])

    if not weight_col:
        gini_row = con.execute(f"""
            WITH v AS (SELECT {e} AS x FROM {table} WHERE {e} IS NOT NULL AND {e} >= 0),
            r AS (SELECT x, ROW_NUMBER() OVER (ORDER BY x) AS rn, COUNT(*) OVER () AS n FROM v)
            SELECT (2.0 * SUM(rn * x) - (MAX(n) + 1) * SUM(x)) / (MAX(n) * SUM(x)) AS gini
            FROM r
        """).fetchone()
    else:
        we = numeric_expr(weight_col)
        gini_row = con.execute(f"""
            WITH v AS (
                SELECT {e} AS x, {we} AS w FROM {table}
                WHERE {e} IS NOT NULL AND {e} >= 0 AND {we} IS NOT NULL AND {we} > 0
            ),
            r AS (SELECT x, w, SUM(w * x) OVER (ORDER BY x) AS cum_wx FROM v),
            totals AS (SELECT SUM(w) AS total_w, SUM(w * x) AS total_wx FROM v)
            SELECT 1.0 - SUM(r.w * (2 * r.cum_wx - r.w * r.x)) / (ANY_VALUE(t.total_w) * ANY_VALUE(t.total_wx)) AS gini
            FROM r CROSS JOIN totals t
        """).fetchone()
    if gini_row is None or gini_row[0] is None:
        return None
    gini = float(gini_row[0])

    if not weight_col:
        buckets = _fetch(con, f"""
            WITH v AS (SELECT {e} AS x FROM {table} WHERE {e} IS NOT NULL AND {e} >= 0),
            b AS (SELECT x, NTILE({n_buckets}) OVER (ORDER BY x) AS bucket FROM v)
            SELECT bucket, SUM(x) AS bucket_sum, COUNT(*) AS bucket_n
            FROM b GROUP BY 1 ORDER BY 1
        """)
    else:
        we = numeric_expr(weight_col)
        # Lorenz egrisi icin de "nufus payi" artik SATIR SAYISI degil,
        # AGIRLIK TOPLAMI ile olculur - NTILE hala satir siralamasina gore
        # dilimler (deger sirasi degismez), ama her dilimin "agirligi" ve
        # "gelir payi" agirlikli toplanir.
        buckets = _fetch(con, f"""
            WITH v AS (
                SELECT {e} AS x, {we} AS w FROM {table}
                WHERE {e} IS NOT NULL AND {e} >= 0 AND {we} IS NOT NULL AND {we} > 0
            ),
            b AS (SELECT x, w, NTILE({n_buckets}) OVER (ORDER BY x) AS bucket FROM v)
            SELECT bucket, SUM(x * w) AS bucket_sum, SUM(w) AS bucket_n
            FROM b GROUP BY 1 ORDER BY 1
        """)
    if buckets.empty:
        return gini, pd.DataFrame({"Nufus Payi": [0, 1], "Gelir Payi": [0, 1]})
    buckets["cum_income"] = buckets["bucket_sum"].cumsum()
    buckets["cum_pop"] = buckets["bucket_n"].cumsum()
    total_income = buckets["bucket_sum"].sum()
    total_pop = buckets["bucket_n"].sum()
    lorenz = pd.DataFrame({
        "Nufus Payi": [0] + (buckets["cum_pop"] / total_pop).tolist(),
        "Gelir Payi": [0] + (buckets["cum_income"] / total_income).tolist(),
    })
    return gini, lorenz


def income_bucket_expr(mapping: dict, n: int = 5) -> str:
    e = numeric_expr(mapping["income"])
    return f"NTILE({n}) OVER (ORDER BY {e})"


def income_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    """Ilce bazinda ortalama hanehalki geliri (Rapor Grafik 24). Gelir HANE
    bazli bir deger oldugundan household_id eslenmisse hane bazinda
    tekrarsiz hesaplanir (bkz. income_summary docstring'i)."""
    if not has(mapping, "income", "district"):
        return None
    src = _dedup_source(table, mapping.get("household_id"),
                        _with_weight(mapping, [mapping["income"], mapping["district"]]))
    df = numeric_by_group_sql(con, src, mapping["district"], mapping["income"], "Ilce", weight_col=mapping.get("weight"))
    if df is None:
        return None
    return df.rename(columns={"Ortalama": "Ortalama Gelir"})[["Ilce", "Ortalama Gelir"]].sort_values(
        "Ortalama Gelir", ascending=False)


def gini_by_district(con, table, mapping: dict, min_households: int = 30) -> pd.DataFrame | None:
    """Ilce bazinda Gini katsayisi (Rapor Grafik 27: 'Bolgeler Arasi Gini
    Katsayisi Karsilastirmasi'). Guvenilir bir Gini hesabi icin en az
    min_households gozlem gerektiren ilceler dahil edilir - cok az gozlemli
    bir ilcede Gini degeri anlamsiz derecede oynak (gurultu) olabilir."""
    if not has(mapping, "income", "district"):
        return None
    weight_col = mapping.get("weight")
    src = _dedup_source(table, mapping.get("household_id"),
                        _with_weight(mapping, [mapping["income"], mapping["district"]]))
    e = numeric_expr(mapping["income"])
    de = cat_expr(mapping["district"])
    if not weight_col:
        rows = _fetch(con, f"""
            WITH v AS (SELECT {de} AS ilce, {e} AS x FROM {src} WHERE {de} IS NOT NULL AND {e} IS NOT NULL AND {e} >= 0),
            r AS (SELECT ilce, x, ROW_NUMBER() OVER (PARTITION BY ilce ORDER BY x) AS rn,
                         COUNT(*) OVER (PARTITION BY ilce) AS n FROM v)
            SELECT ilce AS Ilce, MAX(n) AS Hane_Sayisi,
                   (2.0 * SUM(rn * x) - (MAX(n) + 1) * SUM(x)) / (MAX(n) * SUM(x)) AS Gini
            FROM r GROUP BY ilce HAVING MAX(n) >= {min_households}
            ORDER BY Gini DESC
        """)
    else:
        # Ayni agirlikli Gini formulu (bkz. gini_and_lorenz), burada HER ILCE
        # icin AYRI AYRI (PARTITION BY ilce) hesaplanir. min_households artik
        # AGIRLIKLI toplam (tahmini genisletilmis hane sayisi) ile kontrol
        # edilir - guvenilirlik esigi HALA orneklem BUYUKLUGUNE (satir
        # sayisina) gore uygulanir, yoksa tek bir cok agirlikli satir
        # yanlislikla 'yeterli orneklem' sanilabilir.
        we = numeric_expr(weight_col)
        rows = _fetch(con, f"""
            WITH v AS (
                SELECT {de} AS ilce, {e} AS x, {we} AS w FROM {src}
                WHERE {de} IS NOT NULL AND {e} IS NOT NULL AND {e} >= 0 AND {we} IS NOT NULL AND {we} > 0
            ),
            r AS (
                SELECT ilce, x, w, SUM(w * x) OVER (PARTITION BY ilce ORDER BY x) AS cum_wx,
                       COUNT(*) OVER (PARTITION BY ilce) AS satir_sayisi
                FROM v
            ),
            totals AS (
                SELECT ilce, SUM(w) AS total_w, SUM(w * x) AS total_wx, COUNT(*) AS n
                FROM v GROUP BY ilce
            )
            SELECT r.ilce AS Ilce, ANY_VALUE(t.n) AS Hane_Sayisi,
                   1.0 - SUM(r.w * (2 * r.cum_wx - r.w * r.x)) / (ANY_VALUE(t.total_w) * ANY_VALUE(t.total_wx)) AS Gini
            FROM r JOIN totals t ON r.ilce = t.ilce
            GROUP BY r.ilce HAVING ANY_VALUE(t.n) >= {min_households}
            ORDER BY Gini DESC
        """)
    return rows if not rows.empty else None


_QUINTILE_LABELS = {1: "1. %20 (Alt Gelir)", 2: "2. %20", 3: "3. %20", 4: "4. %20", 5: "5. %20 (Ust Gelir)"}


def income_quintile_distribution(con, table, mapping: dict) -> pd.Series | None:
    if not mapping.get("income"):
        return None
    weight_col = mapping.get("weight")
    table = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [mapping["income"]]))
    e = numeric_expr(mapping["income"])
    if weight_col:
        # Dilim SINIRLARI (NTILE) satir siralamasina gore belirlenir - ama
        # her dilimin "buyuklugu" artik satir sayisi degil AGIRLIK TOPLAMI
        # ile raporlanir (gercek nufus payi).
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            WITH v AS (SELECT {e} AS x, {we} AS w FROM {table} WHERE {e} IS NOT NULL AND {we} IS NOT NULL),
            q AS (SELECT x, w, NTILE(5) OVER (ORDER BY x) AS grp FROM v)
            SELECT grp, SUM(w) AS n FROM q GROUP BY 1 ORDER BY 1
        """)
    else:
        df = _fetch(con, f"""
            WITH v AS (SELECT {e} AS x FROM {table} WHERE {e} IS NOT NULL),
            q AS (SELECT x, NTILE(5) OVER (ORDER BY x) AS grp FROM v)
            SELECT grp, COUNT(*) AS n FROM q GROUP BY 1 ORDER BY 1
        """)
    if df.empty:
        return None
    total = df["n"].sum()
    idx = [_QUINTILE_LABELS[g] for g in df["grp"]]
    return pd.Series((df["n"] / total * 100).values, index=idx, name="pct")


def income_group_by_category(con, table, mapping: dict, concept: str) -> pd.DataFrame | None:
    """Gelir grubu (5'li dilim) x baska bir kategorik degisken (amac/mod) capraz tablosu.
    Gelir ceyreklik SINIRLARI HANE bazinda (household_id eslenmisse tekrarsiz)
    hesaplanir - aksi halde cok yolculuk yapan (dolayisiyla tabloda cok kez
    tekrar eden) haneler gelir siralamasini yanlis carpitirdi. Her yolculuk
    daha sonra kendi hanesinin dogru gelir grubuna atanir."""
    if not has(mapping, "income", concept):
        return None
    weight_col = mapping.get("weight")
    inc_e = numeric_expr(mapping["income"])
    cat = cat_expr(mapping[concept])
    hh_col = mapping.get("household_id")
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""

    if hh_col:
        hh = ident(hh_col)
        df = _fetch(con, f"""
            WITH hh_income AS (
                SELECT {hh} AS hh, ANY_VALUE({inc_e}) AS inc
                FROM {table} WHERE {hh} IS NOT NULL AND {inc_e} IS NOT NULL
                GROUP BY 1
            ),
            hh_grp AS (SELECT hh, NTILE(5) OVER (ORDER BY inc) AS grp FROM hh_income)
            SELECT g.grp AS grp, {cat} AS cat, {n_expr} AS n
            FROM {table} t JOIN hh_grp g ON t.{hh} = g.hh
            WHERE {cat} IS NOT NULL{w_where}
            GROUP BY 1, 2
        """)
    else:
        # household_id eslenmemis - eski (satir-bazli, potansiyel carpitmali)
        # yontemle devam edilir; en azindan hicbir sonuc uretmemekten iyidir.
        weight_sel = f", {numeric_expr(weight_col)} AS w" if weight_col else ""
        n_expr2 = "SUM(w)" if weight_col else "COUNT(*)"
        df = _fetch(con, f"""
            WITH base AS (
                SELECT {inc_e} AS inc, {cat} AS cat{weight_sel}
                FROM {table} WHERE {inc_e} IS NOT NULL AND {cat} IS NOT NULL{w_where}
            ),
            q AS (SELECT cat, NTILE(5) OVER (ORDER BY inc) AS grp{', w' if weight_col else ''} FROM base)
            SELECT grp, cat, {n_expr2} AS n FROM q GROUP BY 1, 2
        """)
    if df.empty:
        return None
    df["grp"] = df["grp"].map(_QUINTILE_LABELS)
    wide = df.pivot_table(index="grp", columns="cat", values="n", fill_value=0)
    wide = wide.div(wide.sum(axis=1), axis=0) * 100
    wide = wide.reindex([v for v in _QUINTILE_LABELS.values() if v in wide.index])
    wide.index.name = None
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------- arac sahipligi
# NOT: arac sayisi HANE bazli bir degerdir. household_id eslenmisse hane
# bazinda TEKRARSIZ sayilir (aksi halde yolculuk-bazli bir tabloda ayni
# hanenin araci, o hanenin yaptigi her yolculukta bir kez daha sayilirdi).
def vehicle_ownership_distribution(con, table, mapping: dict) -> pd.Series | None:
    col = mapping.get("vehicle_count")
    if not col:
        return None
    weight_col = mapping.get("weight")
    src = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [col]))
    e = numeric_expr(col)
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    where_extra = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT CASE WHEN {e} <= 0 THEN '0 Arac' WHEN {e} = 1 THEN '1 Arac'
                    WHEN {e} = 2 THEN '2 Arac' ELSE '3+ Arac' END AS cat, {n_expr} AS n
        FROM {src} WHERE {e} IS NOT NULL{where_extra}
        GROUP BY 1
    """)
    if df.empty:
        return None
    order = ["0 Arac", "1 Arac", "2 Arac", "3+ Arac"]
    total = df["n"].sum()
    s = pd.Series(df.set_index("cat")["n"] / total * 100)
    return s.reindex([o for o in order if o in s.index])


def vehicles_per_1000_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    """1000 KISIYE dusen arac sayisi: pay (toplam arac) HANE bazinda,
    payda (nufus) KISI bazinda tekrarsiz hesaplanir - ikisi de tekrarsiz
    olmadan (orn. yolculuk-bazli bir tabloda) oran anlamsizlasirdi.

    NUFUS (payda) icin ONCELIK SIRASI - DOGRULUK ICIN KRITIK:
      1) person_id eslenmisse: KISI bazinda tekrarsiz gercek nufus sayimi.
      2) person_id yoksa ama household_size eslenmisse: SUM(hanehalki
         buyuklugu) - hane genis (kisi bazli olmayan, orn. ham genis) bir
         tabloda calisirken bu, gercek nufusa en yakin tahmindir.
      3) hicbiri yoksa: None DONER - once bu fonksiyon person_id
         eslenmemisse SESSIZCE SATIR (hane) sayisini "kisi sayisi" olarak
         kullanip GERCEKTEN OLDUGUNDAN COK DAHA YUKSEK, yanlis ama makul
         gorunen bir oran uretiyordu (bir gercek dosyada 1000 kisiye dusen
         arac sayisi ~330 yerine ~955 gibi cikip olculup dogrulandi) - bu
         artik mumkun degil, yanlis bir sayi uretmek yerine hesaplanamaz."""
    if not has(mapping, "district", "vehicle_count"):
        return None
    weight_col = mapping.get("weight")
    dist_col, veh_col = mapping["district"], mapping["vehicle_count"]
    veh_src = _dedup_source(table, mapping.get("household_id"), _with_weight(mapping, [dist_col, veh_col]))
    e = numeric_expr(veh_col)
    de = cat_expr(dist_col)
    # Agirlik varsa arac ADEDI de agirlikli (SUM(arac*agirlik)) toplanir -
    # boylece "1000 kisiye dusen arac" tahmini de dogru genisletilmis olur.
    veh_sum_expr = f"SUM({e} * {numeric_expr(weight_col)})" if weight_col else f"SUM({e})"
    veh_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    veh_by_district = _fetch(con, f"""
        SELECT {de} AS Ilce, {veh_sum_expr} AS toplam_arac
        FROM {veh_src} WHERE {de} IS NOT NULL AND {e} IS NOT NULL{veh_where}
        GROUP BY 1
    """)
    if veh_by_district.empty:
        return None

    if mapping.get("person_id"):
        pop_src = _dedup_source(table, mapping["person_id"], _with_weight(mapping, [dist_col]))
        pop_n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
        pop_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
        pop_by_district = _fetch(con, f"""
            SELECT {de} AS Ilce, {pop_n_expr} AS Kisi_Sayisi
            FROM {pop_src} WHERE {de} IS NOT NULL{pop_where}
            GROUP BY 1
        """)
    elif mapping.get("household_size"):
        pop_src = _dedup_source(table, mapping.get("household_id"),
                                _with_weight(mapping, [dist_col, mapping["household_size"]]))
        hs_e = numeric_expr(mapping["household_size"])
        hs_sum_expr = f"SUM({hs_e} * {numeric_expr(weight_col)})" if weight_col else f"SUM({hs_e})"
        hs_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
        pop_by_district = _fetch(con, f"""
            SELECT {de} AS Ilce, {hs_sum_expr} AS Kisi_Sayisi
            FROM {pop_src} WHERE {de} IS NOT NULL AND {hs_e} IS NOT NULL{hs_where}
            GROUP BY 1
        """)
    else:
        return None
    if pop_by_district.empty:
        return None
    merged = veh_by_district.merge(pop_by_district, on="Ilce", how="inner")
    if merged.empty:
        return None
    merged["1000_Kisiye_Dusen_Arac"] = merged["toplam_arac"] / merged["Kisi_Sayisi"] * 1000
    return merged[["Ilce", "1000_Kisiye_Dusen_Arac", "Kisi_Sayisi"]].sort_values(
        "1000_Kisiye_Dusen_Arac", ascending=False)


def vehicle_ownership_rate_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    """Ilce bazinda EN AZ 1 aracin sahip oldugu hanelerin yuzdesi (Rapor
    Grafik 29: 'Ilce Bazinda Arac Sahipligi'). vehicles_per_1000_by_district'ten
    farki: bu bir ORAN (kac hanenin arabasi VAR), o ise bir YOGUNLUK (toplam
    arac / nufus) olcusudur - rapor ikisini de ayri grafiklerde gosterir."""
    if not has(mapping, "district", "vehicle_count"):
        return None
    weight_col = mapping.get("weight")
    src = _dedup_source(table, mapping.get("household_id"),
                        _with_weight(mapping, [mapping["district"], mapping["vehicle_count"]]))
    e = numeric_expr(mapping["vehicle_count"])
    de = cat_expr(mapping["district"])
    if not weight_col:
        df = _fetch(con, f"""
            SELECT {de} AS Ilce, AVG(CASE WHEN {e} > 0 THEN 100.0 ELSE 0 END) AS Arac_Sahip_Yuzde,
                   COUNT(*) AS Hane_Sayisi
            FROM {src} WHERE {de} IS NOT NULL AND {e} IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """)
    else:
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            SELECT {de} AS Ilce,
                   SUM(CASE WHEN {e} > 0 THEN {we} ELSE 0 END) / SUM({we}) * 100 AS Arac_Sahip_Yuzde,
                   COUNT(*) AS Hane_Sayisi
            FROM {src} WHERE {de} IS NOT NULL AND {e} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """)
    return df if not df.empty else None


# ---------------------------------------------------------------- yolculuk / hareketlilik
def mobility_rate(con, table, mapping: dict) -> float | None:
    if not mapping.get("person_id"):
        return None
    weight_col = mapping.get("weight")
    pid = ident(mapping["person_id"])
    if not weight_col:
        row = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {pid}) FROM {table}").fetchone()
        if not row or not row[1]:
            return None
        return row[0] / row[1]
    # Agirlikli hareketlilik orani: pay = TUM yolculuk satirlarinin agirlik
    # toplami; payda = HER KISININ (tekrarsiz) KENDI agirligi (bir kisinin
    # birden fazla yolculugu varsa agirligi ANY_VALUE ile BIR KEZ sayilir -
    # aksi halde cok yolculuk yapan kisiler paydayi da yanlislikla sisirirdi).
    we = numeric_expr(weight_col)
    row = con.execute(f"""
        WITH persons AS (
            SELECT {pid} AS p, ANY_VALUE({we}) AS w FROM {table} WHERE {we} IS NOT NULL GROUP BY {pid}
        )
        SELECT (SELECT SUM({we}) FROM {table} WHERE {we} IS NOT NULL) AS toplam_yolculuk_agirlik,
               (SELECT SUM(w) FROM persons) AS toplam_kisi_agirlik
    """).fetchone()
    if not row or not row[1]:
        return None
    return row[0] / row[1]


def mobility_rate_by_district(con, table, mapping: dict) -> pd.DataFrame | None:
    if not has(mapping, "district", "person_id"):
        return None
    weight_col = mapping.get("weight")
    dist, pid = cat_expr(mapping["district"]), ident(mapping["person_id"])
    if not weight_col:
        df = _fetch(con, f"""
            SELECT {dist} AS Ilce, COUNT(*) AS Yolculuk_Sayisi, COUNT(DISTINCT {pid}) AS Kisi_Sayisi,
                   COUNT(*) * 1.0 / COUNT(DISTINCT {pid}) AS Hareketlilik_Orani
            FROM {table} WHERE {dist} IS NOT NULL
            GROUP BY 1 ORDER BY 4 DESC
        """)
        return df if not df.empty else None
    # Agirlikli surum: her ilce icin pay=yolculuk agirlik toplami, payda=o
    # ilcedeki (tekrarsiz) kisilerin agirlik toplami.
    we = numeric_expr(weight_col)
    df = _fetch(con, f"""
        WITH trip_w AS (
            SELECT {dist} AS Ilce, SUM({we}) AS Yolculuk_Sayisi
            FROM {table} WHERE {dist} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1
        ),
        person_w AS (
            SELECT {dist} AS Ilce, {pid} AS p, ANY_VALUE({we}) AS w
            FROM {table} WHERE {dist} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1, 2
        ),
        person_agg AS (
            SELECT Ilce, SUM(w) AS Kisi_Sayisi FROM person_w GROUP BY 1
        )
        SELECT trip_w.Ilce AS Ilce, trip_w.Yolculuk_Sayisi, person_agg.Kisi_Sayisi,
               trip_w.Yolculuk_Sayisi / person_agg.Kisi_Sayisi AS Hareketlilik_Orani
        FROM trip_w JOIN person_agg ON trip_w.Ilce = person_agg.Ilce
        ORDER BY 4 DESC
    """)
    return df if not df.empty else None


def home_work_district_comparison(con, table, mapping: dict, work_positive_values: list[str] | None = None,
                                   employment_col: str | None = None) -> pd.DataFrame | None:
    """Ikamet (district) ile istihdam/varis (district_dest) ilcesini
    karsilastirir (Rapor Grafik 19-20: 'Calisanlarin Istihdam Edildikleri
    Ilceler' ve 'Istihdamin Calisanlari Karsilama Orani'). work_positive_values
    verilirse (employment_col ile birlikte), sadece o kategorilere (orn.
    'Calisan') uyan kayitlar 'istihdam' tarafinda sayilir; verilmezse TUM
    kayitlar (orn. dogal olarak yolculuk-bazli bir tabloda tum varis
    noktalari) kullanilir."""
    if not has(mapping, "district", "district_dest"):
        return None
    home_e = cat_expr(mapping["district"])
    work_e = cat_expr(mapping["district_dest"])
    where_extra = ""
    if work_positive_values and employment_col:
        ecol = cat_expr(employment_col)
        vals_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in work_positive_values)
        where_extra = f" AND {ecol} IN ({vals_sql})"
    home_counts = _fetch(con, f"""
        SELECT {home_e} AS Ilce, COUNT(*) AS Ikamet FROM {table}
        WHERE {home_e} IS NOT NULL GROUP BY 1
    """)
    work_counts = _fetch(con, f"""
        SELECT {work_e} AS Ilce, COUNT(*) AS Istihdam FROM {table}
        WHERE {work_e} IS NOT NULL{where_extra} GROUP BY 1
    """)
    if home_counts.empty and work_counts.empty:
        return None
    merged = home_counts.merge(work_counts, on="Ilce", how="outer").fillna(0)
    merged["Karsilama_Orani_%"] = np.where(
        merged["Ikamet"] > 0, merged["Istihdam"] / merged["Ikamet"] * 100, np.nan
    )
    return merged.sort_values("Istihdam", ascending=False)


def purpose_distribution(con, table, mapping: dict,
                          group_siblings: dict[str, list[str]] | None = None) -> pd.Series | None:
    return category_distribution(con, table, mapping, "purpose", group_siblings=group_siblings)


def mode_distribution(con, table, mapping: dict,
                       group_siblings: dict[str, list[str]] | None = None) -> pd.Series | None:
    return category_distribution(con, table, mapping, "mode", group_siblings=group_siblings)


def purpose_mode_crosstab(con, table, mapping: dict,
                           group_siblings: dict[str, list[str]] | None = None) -> pd.DataFrame | None:
    """group_siblings: bkz. mode_flag_split'teki ayni parametrenin commit
    notu - "mode" (Ulasim Turu) coklu-bacakli/transfer sutunlarinin (orn.
    "yol_arac1/2/3...") sadece birine eslenmisse, diger bacaklar TUM
    kardesler havuzlanarak (amac o bacaklarda AYNI kalip tekrarlanarak)
    dahil edilir; "purpose" (amac) kendisi de gruplu bir sutunsa (nadir
    ama olasi), guvenlik icin havuzlama atlanir."""
    if not has(mapping, "purpose", "mode"):
        return None
    purpose_col = mapping["purpose"]
    mode_col = mapping["mode"]
    weight_col = mapping.get("weight")
    siblings = (group_siblings or {}).get(mode_col)
    if purpose_col in (group_siblings or {}):
        siblings = None
    if siblings and len(siblings) > 1:
        extra_sel = f", {ident(weight_col)}" if weight_col else ""
        union_sql = " UNION ALL ".join(
            f"SELECT {cat_expr(purpose_col)} AS {ident(purpose_col)}, "
            f"{cat_expr(sib)} AS {ident(mode_col)}{extra_sel} FROM {table}" for sib in siblings
        )
        table = f"({union_sql}) AS _pooled"
    return crosstab_sql(con, table, purpose_col, mode_col, weight_col=weight_col)


def mode_flag_split(con, table, mapping: dict, vehicle_mode_values: list[str],
                     by_district: bool = False,
                     group_siblings: dict[str, list[str]] | None = None) -> pd.Series | pd.DataFrame | None:
    """Ulasim turunu 'Aracli' / 'Aracsiz' (yurume haric tum motorlu/motorsuz
    araclar vs. yaya) iki gruba ayirir (Rapor Grafik 34-45'te tekrarlanan
    'aracli-aracsiz yolculuk' ayrimi). vehicle_mode_values: kullanicinin
    'aracli' saydigi ulasim turu deger(ler)i (orn. ['Otomobil','Otobus',...]);
    rapor tek bir sabit sozluk kullanmadigindan bu secim kullaniciya
    birakilir (bkz. app.py 'Sutun Gruplari' tarzi pozitif-deger secimi).
    by_district=True ise ilce bazinda %100 istiflenmis tablo doner.

    group_siblings: DOGRULUK ICIN KRITIK (gercek bir dosyada olculup
    dogrulandi - "Ulasim Turu" transfer/bacak sutunlari icin, orn.
    "yol_arac1/2/3..."). Eslenen sutun ('mapping["mode"]') aslinda
    coklu-bacakli bir yolculugun SADECE ILK bacagini (transfer oncesi)
    tutuyorsa, 2. ve 3. bacaklardaki (aktarma sonrasi) ulasim turleri
    SESSIZCE sayilmiyordu - kullanicinin kendi gozlemiyle yakalandi. TUM
    bacak sutunlari tespit edilirse (havuzlanabilirse - ilce de ayrica
    gruplu degilse) UNION ALL ile havuzlanir; aksi halde davranis ONCEKI
    ile BIREBIR AYNIDIR."""
    if not mapping.get("mode") or not vehicle_mode_values:
        return None
    weight_col = mapping.get("weight")
    mode_col = mapping["mode"]
    dist_col = mapping.get("district") if by_district else None
    siblings = (group_siblings or {}).get(mode_col)
    if dist_col and dist_col in (group_siblings or {}):
        siblings = None  # guvenlik: ilce de gruplu ise havuzlama atlanir (bkz. docstring)
    if siblings and len(siblings) > 1:
        extra_sel = (f", {ident(weight_col)}" if weight_col else "") + \
                    (f", {ident(dist_col)}" if dist_col else "")
        union_sql = " UNION ALL ".join(
            f"SELECT {cat_expr(sib)} AS {ident(mode_col)}{extra_sel} FROM {table}" for sib in siblings
        )
        table = f"({union_sql}) AS _pooled"
    mode_e = cat_expr(mode_col)
    vals_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in vehicle_mode_values)
    flag_expr = f"CASE WHEN {mode_e} IN ({vals_sql}) THEN 'Aracli' ELSE 'Aracsiz' END"
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    if by_district:
        if not mapping.get("district"):
            return None
        de = cat_expr(mapping["district"])
        df = _fetch(con, f"""
            SELECT {de} AS Ilce, {flag_expr} AS grp, {n_expr} AS n
            FROM {table} WHERE {de} IS NOT NULL AND {mode_e} IS NOT NULL{w_where}
            GROUP BY 1, 2
        """)
        if df.empty:
            return None
        wide = df.pivot_table(index="Ilce", columns="grp", values="n", fill_value=0)
        wide = wide.div(wide.sum(axis=1), axis=0) * 100
        wide.columns.name = None
        return wide.reset_index()
    df = _fetch(con, f"""
        SELECT {flag_expr} AS grp, {n_expr} AS n FROM {table} WHERE {mode_e} IS NOT NULL{w_where} GROUP BY 1
    """)
    if df.empty:
        return None
    total = df["n"].sum()
    return pd.Series((df["n"] / total * 100).values, index=df["grp"], name="pct")


def duration_stats_by(con, table, mapping: dict, by_concept: str) -> pd.DataFrame | None:
    if not has(mapping, "duration", by_concept):
        return None
    df = numeric_by_group_sql(con, table, mapping[by_concept], mapping["duration"], "Grup",
                               weight_col=mapping.get("weight"))
    if df is None:
        return None
    return df.rename(columns={"Ortalama": "Ortalama Sure (dk)", "Medyan": "Medyan Sure (dk)"})


def duration_frequency_distribution(con, table, mapping: dict,
                                     bin_edges: tuple = (0, 10, 20, 30, 45, 60, 90, 1000),
                                     bin_labels: tuple = ("0-10", "10-20", "20-30", "30-45",
                                                           "45-60", "60-90", "90+")) -> pd.Series | None:
    """Yolculuk surelerinin araliklara gore frekans dagilimi (%) - Rapor
    Grafik 51/53 ve Tablo 18 'Ortalama Surelerinin Araliklara Gore Dagilimi'."""
    if not mapping.get("duration"):
        return None
    weight_col = mapping.get("weight")
    e = numeric_expr(mapping["duration"])
    case_parts = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        case_parts.append(f"WHEN {e} >= {lo} AND {e} < {hi} THEN '{bin_labels[i]}'")
    case_sql = "CASE " + " ".join(case_parts) + " END"
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"SELECT {case_sql} AS grp, {n_expr} AS n FROM {table} WHERE {e} IS NOT NULL{w_where} GROUP BY 1")
    if df.empty:
        return None
    df = df.dropna(subset=["grp"])
    total = df["n"].sum()
    if not total:
        return None
    s = pd.Series(df.set_index("grp")["n"] / total * 100)
    return s.reindex([b for b in bin_labels if b in s.index])


def range_quality_flags(con, table, col: str, low, high, label: str) -> dict | None:
    """duration_quality_flags ile AYNI mantigin GENEL (herhangi bir sayisal
    alan icin) surumu - CMAP My Daily Travel'in "gecerli aralik disi kayitlari
    ayri raporla" pratiginin dogrudan genellemesi (bkz. duration_quality_flags
    docstring'i - CMAP yas<5 ve mesafe>100 mil gibi ESIK DEGERLERLE calisir,
    burada da ayni fikir herhangi bir alt/ust sinir icin uygulanir).

    SADECE RAPORLAR - hicbir satiri silmez, mapping[col] uzerinden yapilan
    BASKA HICBIR hesabi degistirmez.

    Doner: {"toplam", "aralik_disi", "aralik_disi_yuzde", "low", "high",
    "label"} ya da col=None/veri yoksa None."""
    if not col:
        return None
    e = numeric_expr(col)
    df = _fetch(con, f"""
        SELECT COUNT(*) AS toplam,
               COUNT(*) FILTER (WHERE {e} < {low} OR {e} > {high}) AS aralik_disi
        FROM {table} WHERE {e} IS NOT NULL
    """)
    if df.empty or not df["toplam"].iloc[0]:
        return None
    row = df.iloc[0]
    total = int(row["toplam"])
    out_n = int(row["aralik_disi"])
    return {
        "toplam": total, "aralik_disi": out_n,
        "aralik_disi_yuzde": round(out_n / total * 100, 2) if total else 0.0,
        "low": low, "high": high, "label": label,
    }


def iqr_outlier_flags(con, table, col: str, label: str, k: float = 1.5) -> dict | None:
    """Sayisal bir alan icin standart IQR (ceyrekler-arasi aralik) yontemiyle
    istatistiksel aykiri deger tespiti - Q1 - k*IQR ile Q3 + k*IQR disindaki
    degerler 'aykiri' sayilir (k=1.5, kutu grafiginin/box-plot'un standart
    esigi). range_quality_flags'ten farki: SABIT bir esik yerine VERININ
    KENDI dagilimina gore ADAPTIF bir esik kullanir - gelir gibi, 'normal'
    araligi sehirden sehire cok degisen alanlar icin daha uygundur.

    SADECE RAPORLAR - hicbir satiri silmez/baska hesabi degistirmez.

    Doner: {"toplam","aykiri","aykiri_yuzde","q1","q3","iqr","alt_esik",
    "ust_esik","label"} ya da col=None/veri yoksa None."""
    if not col:
        return None
    e = numeric_expr(col)
    row = con.execute(f"""
        SELECT COUNT({e}), quantile_cont({e}, 0.25), quantile_cont({e}, 0.75)
        FROM {table} WHERE {e} IS NOT NULL
    """).fetchone()
    if not row or not row[0] or row[1] is None or row[2] is None:
        return None
    total, q1, q3 = row
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out_row = con.execute(f"""
        SELECT COUNT(*) FILTER (WHERE {e} < {lo} OR {e} > {hi})
        FROM {table} WHERE {e} IS NOT NULL
    """).fetchone()
    out_n = int(out_row[0]) if out_row else 0
    return {
        "toplam": int(total), "aykiri": out_n,
        "aykiri_yuzde": round(out_n / total * 100, 2) if total else 0.0,
        "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
        "alt_esik": round(lo, 2), "ust_esik": round(hi, 2), "label": label,
    }


def duration_quality_flags(con, table, mapping: dict, long_trip_minutes: int = 180) -> dict | None:
    """Yolculuk suresi icin TEMEL bir veri kalitesi/olagandisi-deger kontrolu.

    NEDEN: benzer acik kaynak anket projeleri (orn. CMAP "My Daily Travel"
    - github.com/CMAP-REPOS/mydailytravel) yolculuk kayitlarini analize
    sokmadan once fiziksel olarak imkansiz/supheli olanlari (orn. cok uzun
    ya da negatif sureli) ayri raporluyor - bu, hem veri girisi hatalarini
    yakalamaya hem de ortalama/medyan gibi ozet degerlerin birkac asiri
    deger yuzunden carpilmadigini gormeye yarar. Bu fonksiyon o mantigin
    SADE bir surumudur: SADECE RAPORLAR, hicbir satiri SESSIZCE silmez ya
    da baska hesaplari degistirmez (kullanicinin "degerler asla degismesin"
    kurali geregi) - sadece "su kadar kayit supheli, kontrol edin" der.

    Doner: {"toplam": int, "gecersiz_sure": int, "gecersiz_yuzde": float,
            "olagandisi_uzun": int, "olagandisi_uzun_yuzde": float,
            "p99": float, "long_trip_minutes": int} ya da 'duration'
    eslenmemisse/veri yoksa None."""
    if not mapping.get("duration"):
        return None
    e = numeric_expr(mapping["duration"])
    df = _fetch(con, f"""
        SELECT
            COUNT(*) AS toplam,
            COUNT(*) FILTER (WHERE {e} <= 0) AS gecersiz_sure,
            COUNT(*) FILTER (WHERE {e} > {long_trip_minutes}) AS olagandisi_uzun,
            quantile_cont({e}, 0.99) FILTER (WHERE {e} > 0) AS p99
        FROM {table} WHERE {e} IS NOT NULL
    """)
    if df.empty or not df["toplam"].iloc[0]:
        return None
    row = df.iloc[0]
    total = int(row["toplam"])
    invalid = int(row["gecersiz_sure"])
    long_n = int(row["olagandisi_uzun"])
    return {
        "toplam": total,
        "gecersiz_sure": invalid,
        "gecersiz_yuzde": round(invalid / total * 100, 2) if total else 0.0,
        "olagandisi_uzun": long_n,
        "olagandisi_uzun_yuzde": round(long_n / total * 100, 2) if total else 0.0,
        "p99": round(float(row["p99"]), 1) if pd.notna(row["p99"]) else None,
        "long_trip_minutes": long_trip_minutes,
    }


_HOUR_SQL_CACHE: dict[str, str] = {}


def hour_expr(col: str) -> str:
    c = ident(col)
    txt = f"CAST({c} AS VARCHAR)"
    return (
        "COALESCE("
        f"TRY_CAST(split_part({txt}, ':', 1) AS INTEGER), "
        f"TRY_CAST({txt} AS INTEGER), "
        f"TRY_CAST(EXTRACT(HOUR FROM TRY_CAST({c} AS TIME)) AS INTEGER), "
        f"TRY_CAST(EXTRACT(HOUR FROM TRY_CAST({c} AS TIMESTAMP)) AS INTEGER)"
        ")"
    )


def minute_of_day_expr(col: str) -> str:
    """'HH:MM' (ya da TIME/TIMESTAMP) bicimli bir saat sutununu, gece
    yarisindan itibaren gecen DAKIKA sayisina (0-1439) cevirir. hour_expr'den
    farki: sadece saati degil dakikayi da kullanarak, yolculuk suresini
    (bitis-baslangic) DAHA HASSAS hesaplamaya olanak tanir (bkz.
    'baslangic/bitis saatinden turetilmis sure' ozelligi, app.py)."""
    c = ident(col)
    txt = f"CAST({c} AS VARCHAR)"
    return (
        "COALESCE("
        f"TRY_CAST(split_part({txt}, ':', 1) AS INTEGER) * 60 + "
        f"COALESCE(TRY_CAST(split_part({txt}, ':', 2) AS INTEGER), 0), "
        f"EXTRACT(HOUR FROM TRY_CAST({c} AS TIME))::INTEGER * 60 + "
        f"EXTRACT(MINUTE FROM TRY_CAST({c} AS TIME))::INTEGER, "
        f"EXTRACT(HOUR FROM TRY_CAST({c} AS TIMESTAMP))::INTEGER * 60 + "
        f"EXTRACT(MINUTE FROM TRY_CAST({c} AS TIMESTAMP))::INTEGER"
        ")"
    )


def binned_time_distribution(con, table, mapping: dict, time_concept: str = "start_time",
                              bin_minutes: int = 60) -> pd.Series | None:
    """hourly_distribution ile AYNI mantik ama SERBEST bin genisligi ile
    (orn. 30 dk, 120 dk) - MTA-dash gibi projelerdeki 'zaman agregasyon
    secici' ozelliginden esinlenilmistir. bin_minutes=60 verildiginde
    hourly_distribution ile TAM AYNI degerleri uretir (saat 0-23 icin ayni
    gruplama) - bu, rapor-esli 'Grafik 54' gibi MEVCUT sabit-saatlik
    grafiklerin DEGISMEDEN kalmasini, bu fonksiyonun SADECE ek/istege bagli
    bir gorunum olarak eklenmesini saglar."""
    col = mapping.get(time_concept)
    if not col or bin_minutes <= 0 or bin_minutes > 1440:
        return None
    weight_col = mapping.get("weight")
    me = minute_of_day_expr(col)
    bin_expr = f"CAST(FLOOR(({me}) / {bin_minutes}) AS INTEGER)"
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT {bin_expr} AS b, {n_expr} AS n
        FROM {table} WHERE {me} IS NOT NULL AND {me} BETWEEN 0 AND 1439{w_where}
        GROUP BY 1
    """)
    if df.empty:
        return None
    n_bins = (1440 + bin_minutes - 1) // bin_minutes
    total = df["n"].sum()
    if not total:
        return None
    s = df.set_index("b")["n"].reindex(range(n_bins), fill_value=0) / total * 100

    def _label(b):
        start = b * bin_minutes
        h, m = divmod(start, 60)
        return f"{h:02d}:{m:02d}"

    s.index = [_label(b) for b in s.index]
    return s


def hourly_distribution(con, table, mapping: dict, time_concept: str = "start_time") -> pd.Series | None:
    col = mapping.get(time_concept)
    if not col:
        return None
    weight_col = mapping.get("weight")
    he = hour_expr(col)
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT {he} AS h, {n_expr} AS n
        FROM {table} WHERE {he} IS NOT NULL AND {he} BETWEEN 0 AND 23{w_where}
        GROUP BY 1
    """)
    if df.empty:
        return None
    total = df["n"].sum()
    s = df.set_index("h")["n"].reindex(range(24), fill_value=0) / total * 100
    s.index = [f"{h:02d}:00" for h in s.index]
    return s


def hourly_distribution_by_purpose(con, table, mapping: dict) -> pd.DataFrame | None:
    if not has(mapping, "start_time", "purpose"):
        return None
    weight_col = mapping.get("weight")
    he = hour_expr(mapping["start_time"])
    purpose = cat_expr(mapping["purpose"])
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT {he} AS h, {purpose} AS purpose, {n_expr} AS n
        FROM {table} WHERE {he} IS NOT NULL AND {he} BETWEEN 0 AND 23 AND {purpose} IS NOT NULL{w_where}
        GROUP BY 1, 2
    """)
    if df.empty:
        return None
    wide = df.pivot_table(index="h", columns="purpose", values="n", fill_value=0)
    wide = wide.div(wide.sum(axis=0), axis=1) * 100
    wide = wide.reindex(range(24), fill_value=0)
    wide.index = [f"{h:02d}:00" for h in wide.index]
    wide.columns.name = None
    return wide


def hourly_distribution_by_group(con, table, mapping: dict, group_concept: str,
                                  time_concept: str = "start_time") -> pd.DataFrame | None:
    """hourly_distribution_by_purpose ile AYNI mantik ama herhangi bir
    kategorik kavrama (orn. cinsiyet, egitim durumu) gore genellenmis -
    'hangi grup GUNUN HANGI SAATINDE yolculuk yapiyor' karsilastirmasi
    (ActivityViz gibi projelerdeki 'zaman kullanimi' analizine benzer).
    Her GRUP kendi icinde %100'e normalize edilir (satir bazinda) - boylece
    gruplarin BUYUKLUGU degil SEKLI (hangi saatlere yayildigi) karsilastirilir."""
    if not mapping.get(group_concept) or not mapping.get(time_concept):
        return None
    weight_col = mapping.get("weight")
    he = hour_expr(mapping[time_concept])
    ge = cat_expr(mapping[group_concept])
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT {he} AS h, {ge} AS grp, {n_expr} AS n
        FROM {table} WHERE {he} IS NOT NULL AND {he} BETWEEN 0 AND 23 AND {ge} IS NOT NULL{w_where}
        GROUP BY 1, 2
    """)
    if df.empty:
        return None
    wide = df.pivot_table(index="grp", columns="h", values="n", fill_value=0)
    wide = wide.div(wide.sum(axis=1), axis=0) * 100
    wide = wide.reindex(columns=range(24), fill_value=0)
    wide.columns = [f"{h:02d}:00" for h in wide.columns]
    wide.index.name = None
    return wide


def _effective_sample_size(con, table, weight_col: str, where_extra: str = "1=1") -> float | None:
    """Kish 'etkin ornek buyuklugu': n_eff = (sum w)^2 / sum(w^2).

    NEDEN: agirlikli bir ortalama/oranin GERCEK kesinligi, satir sayisiyla
    degil bu degerle olculur - agirliklar ne kadar ESITSIZSE (bazi satirlar
    diger satirlarin kat kat 'agir'i), ayni ham satir sayisinda GERCEKTE
    daha AZ bagimsiz bilgi vardir. Tum agirliklar esitse (w=1) n_eff TAM
    OLARAK ham satir sayisina esittir (matematiksel olarak: (n*w)^2/(n*w^2)
    = n) - yani agirliksiz durumda bu fonksiyon hicbir sey degistirmez."""
    we = numeric_expr(weight_col)
    row = con.execute(f"""
        SELECT SUM({we}), SUM({we} * {we}) FROM {table}
        WHERE {we} IS NOT NULL AND {we} > 0 AND {where_extra}
    """).fetchone()
    if not row or not row[0] or not row[1]:
        return None
    sw, sw2 = row
    return (sw ** 2) / sw2


def two_sample_z_test(mean1: float, se1: float, mean2: float, se2: float) -> dict:
    """Iki BAGIMSIZ ornek ortalamasi (ya da orani) arasindaki farkin
    istatistiksel olarak ANLAMLI olup olmadigini test eder (buyuk-ornek
    z-testi - NHTS/PSRC gibi resmi anket araclarinin 'bu fark gercek mi,
    yoksa orneklem hatasi mi' sorusuna verdigi standart cevap).

    NEDEN: guven araligi/karsilastirma ozelliklerimiz iki sayiyi yan yana
    gosteriyordu ama "%30 ile %35 arasindaki fark ANLAMLI mi" sorusuna
    cevap vermiyordu - kucuk orneklemli iki grupta buyuk gorunen bir fark
    aslinda sadece orneklem gurultusu olabilir. Formul: z = (mean1-mean2) /
    sqrt(se1^2 + se2^2); iki-kuyruklu p-degeri standart normal dagilimdan
    (math.erf ile, ek bagimlilik gerektirmeden) hesaplanir.

    Doner: {"fark","se_fark","z","p_value","anlamli_mi"} - anlamli_mi,
    p_value < 0.05 (%95 guvenle) ise True."""
    import math
    diff = mean1 - mean2
    se_diff = (se1 ** 2 + se2 ** 2) ** 0.5
    if se_diff == 0:
        z = float("inf") if diff != 0 else 0.0
    else:
        z = diff / se_diff
    # standart normal CDF: Phi(z) = 0.5*(1+erf(z/sqrt(2))); iki-kuyruklu p-deger
    phi = 0.5 * (1 + math.erf(abs(z) / (2 ** 0.5))) if z not in (float("inf"), float("-inf")) else 1.0
    p_value = max(0.0, 2 * (1 - phi))
    return {
        "fark": diff, "se_fark": se_diff, "z": z, "p_value": p_value,
        "anlamli_mi": p_value < 0.05,
    }


# HANGI kavramlar HANE-bazli (household_id'ye gore tekrarsizlastirilmali) ve
# HANGILERI KISI-bazli (person_id'ye gore) - avg_age/income_summary/
# household_size_stats/gender_distribution/category_distribution gibi
# BUTUN diger fonksiyonlarin ZATEN kullandigi kuralin AYNISI (bkz. bu
# fonksiyonlarin dedup_source cagrilari). confidence_interval_mean/
# _proportion eskiden bu dedup'i HIC yapmiyordu - bu, TRIP-bazli aktif bir
# tabloda (her satir bir yolculuk) hane/kisi-bazli bir alanin (orn. gelir)
# YOLCULUK SAYISINA gore agirliklanmis (yanlis) bir ortalama vermesine yol
# aciyordu (bir dosyada olculdu: 46.554 yerine 50.551 gibi, income_summary
# ile TUTARSIZ bir sonuc). Concept bu listelerde yoksa (orn. amac, sure,
# mod - dogal olarak yolculuk-bazli kavramlar) dedup uygulanmaz - tablo
# oldugu gibi kullanilir (diger fonksiyonlarla AYNI davranis).
_PERSON_LEVEL_CONCEPTS = {"age", "gender", "education", "student", "employment", "literacy"}
_HOUSEHOLD_LEVEL_CONCEPTS = {"income", "household_size", "vehicle_count"}


def _dedup_key_for_concept(mapping: dict, concept: str) -> str | None:
    if concept in _PERSON_LEVEL_CONCEPTS:
        return mapping.get("person_id")
    if concept in _HOUSEHOLD_LEVEL_CONCEPTS:
        return mapping.get("household_id")
    return None


def confidence_interval_mean(con, table, mapping: dict, concept: str) -> dict | None:
    """Sayisal bir alanin (orn. yas, gelir, yolculuk suresi) ortalamasi icin
    %95 GUVEN ARALIGI (margin of error).

    NEDEN: resmi anket toolkit'lerinin standart pratigi (ABD NHTS Data
    Explorer, PSRC Household Travel Survey Explorer vb.) - TEK BASINA
    'ortalama X' yaniltici olabilir: kucuk orneklemli bir ilcede/grupta
    gercek deger COK DAHA genis bir aralikta olabilir. Standart buyuk-
    ornek normal yaklasimi kullanilir: GA = ortalama +/- 1.96 * (std /
    sqrt(n)). Agirlik eslenmisse std zaten AGIRLIKLI hesaplanir
    (numeric_summary_sql), n yerine Kish etkin ornek buyuklugu kullanilir.
    Hane/kisi-bazli kavramlar icin _dedup_key_for_concept'e gore ONCE
    tekrarsizlastirilir (bkz. o fonksiyonun docstring'i - AYRINTILI neden).

    Doner: {"ortalama","std","n","n_eff","se","moe","alt","ust"} ya da
    veri/eslesme yoksa None."""
    if not concept or not mapping.get(concept):
        return None
    col = mapping[concept]
    weight_col = mapping.get("weight")
    dedup_key = _dedup_key_for_concept(mapping, concept)
    src = _dedup_source(table, dedup_key, _with_weight(mapping, [col]))
    summary = numeric_summary_sql(con, src, col, weight_col=weight_col)
    if not summary or not summary.get("std") or not summary.get("n"):
        return None
    n_raw = summary["n"]
    n_eff = float(n_raw)
    if weight_col:
        eff = _effective_sample_size(con, src, weight_col, where_extra=f"{numeric_expr(col)} IS NOT NULL")
        if eff:
            n_eff = eff
    if n_eff <= 1:
        return None
    se = summary["std"] / (n_eff ** 0.5)
    moe = 1.96 * se
    return {
        "ortalama": summary["ortalama"], "std": summary["std"], "n": int(n_raw),
        "n_eff": round(n_eff, 1), "se": se, "moe": moe,
        "alt": summary["ortalama"] - moe, "ust": summary["ortalama"] + moe,
    }


def confidence_interval_proportion(con, table, mapping: dict, concept: str, top_k: int = 10) -> pd.DataFrame | None:
    """Kategorik bir alanin (orn. cinsiyet, ulasim turu, amac) HER kategorisi
    icin oran (%) + %95 guven araligi (margin of error, +/- puan).

    Formul: SE = sqrt(p*(1-p)/n_eff), GA = p +/- 1.96*SE (buyuk-ornek
    normal yaklasimi - NHTS Data Explorer'in da resmi olarak kullandigi
    yontem). n_eff: agirlik yoksa ham gecerli satir sayisi, agirlik varsa
    Kish etkin ornek buyuklugu (bkz. confidence_interval_mean/
    _effective_sample_size docstring'i)."""
    if not concept or not mapping.get(concept):
        return None
    col = mapping[concept]
    weight_col = mapping.get("weight")
    # DOGRULUK ICIN KRITIK (bkz. confidence_interval_mean docstring'indeki
    # AYNI aciklama): hane/kisi-bazli bir kavram (orn. cinsiyet, egitim)
    # ONCE dedup edilir - aksi halde TRIP-bazli bir tabloda, cok yolculuk
    # yapan kisiler/haneler yanlislikla fazla sayilir.
    dedup_key = _dedup_key_for_concept(mapping, concept)
    src = _dedup_source(table, dedup_key, _with_weight(mapping, [col]))
    # DOGRULUK ICIN KRITIK: value_counts_sql, yuzdeyi SADECE getirdigi (limit
    # kadar) kategorinin toplamina gore hesaplar - eger top_k'dan fazla farkli
    # kategori varsa (orn. 16 farkli ulasim turu, top_k=10), limit=top_k ile
    # cagirmak GERCEK toplami degil sadece gosterilen ilk top_k kategorinin
    # toplamini payda yapar, bu da oranlari YANLISLIKLA sisirir (olculdu:
    # gercek %30.36 yerine %30.73 gibi). Bu yuzden burada COK BUYUK bir limit
    # ile (pratikte "hepsi") GERCEK yuzdeler alinir, sadece EN SONDA
    # goruntuleme icin ilk top_k satira kesilir.
    pct = value_counts_sql(con, src, col, limit=100_000, weight_col=weight_col)
    if pct is None or pct.empty:
        return None
    pct = pct.head(top_k)
    e = cat_expr(col)
    if weight_col:
        n_eff = _effective_sample_size(con, src, weight_col, where_extra=f"{e} IS NOT NULL")
    else:
        row = con.execute(f"SELECT COUNT(*) FROM {src} WHERE {e} IS NOT NULL").fetchone()
        n_eff = float(row[0]) if row and row[0] else None
    if not n_eff or n_eff <= 1:
        return None
    p = pct.values / 100.0
    se = np.sqrt(p * (1 - p) / n_eff)
    moe = 1.96 * se * 100
    # NOT: n_eff, DataFrame.attrs YERINE normal bir SUTUN olarak eklenir -
    # .attrs, st.cache_data'nin serialize/deserialize surecinde HER pandas
    # surumunde guvenilir sekilde korunmayabilir (deneysel bir ozellik);
    # sutun olarak eklemek bu riski tamamen ortadan kaldirir.
    out = pd.DataFrame({
        "Kategori": pct.index,
        "Oran (%)": pct.values.round(2),
        "Guven Araligi (+/- puan)": moe.round(2),
        "Alt Sinir (%)": np.clip(pct.values - moe, 0, 100).round(2),
        "Ust Sinir (%)": np.clip(pct.values + moe, 0, 100).round(2),
        "Etkin Ornek (n)": round(n_eff, 1),
    })
    return out


def income_group_purpose(con, table, mapping: dict) -> pd.DataFrame | None:
    return income_group_by_category(con, table, mapping, "purpose")


def income_group_mode(con, table, mapping: dict) -> pd.DataFrame | None:
    return income_group_by_category(con, table, mapping, "mode")


def zone_internal_external(con, table, mapping: dict) -> pd.DataFrame | None:
    if not has(mapping, "district", "district_dest"):
        return None
    weight_col = mapping.get("weight")
    o, d = cat_expr(mapping["district"]), cat_expr(mapping["district_dest"])
    if not weight_col:
        df = _fetch(con, f"""
            SELECT {o} AS Ilce,
                   AVG(CASE WHEN {o} = {d} THEN 100.0 ELSE 0 END) AS "Zon Ici Yuzde",
                   AVG(CASE WHEN {o} != {d} THEN 100.0 ELSE 0 END) AS "Zon Disi Yuzde"
            FROM {table} WHERE {o} IS NOT NULL AND {d} IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """)
    else:
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            SELECT {o} AS Ilce,
                   SUM(CASE WHEN {o} = {d} THEN {we} ELSE 0 END) / SUM({we}) * 100 AS "Zon Ici Yuzde",
                   SUM(CASE WHEN {o} != {d} THEN {we} ELSE 0 END) / SUM({we}) * 100 AS "Zon Disi Yuzde"
            FROM {table} WHERE {o} IS NOT NULL AND {d} IS NOT NULL AND {we} IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """)
    return df if not df.empty else None


def od_matrix(con, table, mapping: dict, top_n: int = 25) -> pd.DataFrame | None:
    if not has(mapping, "district", "district_dest"):
        return None
    return crosstab_sql(con, table, mapping["district"], mapping["district_dest"],
                         normalize="none", row_limit=top_n, col_limit=top_n, weight_col=mapping.get("weight"))


# ---------------------------------------------------------------- aktarma / maliyet
def transfer_stats(con, table, mapping: dict) -> dict | None:
    """Aktarmali yolculuklarin genel istatistikleri (Rapor 1.5.2.8 Aktarmali
    Yolculuklar bolumu: Grafik 70 + Tablo 28-30). Aktarma sayisi 0 olan
    yolculuklar 'aktarmasiz', 1+ olanlar 'aktarmali' sayilir."""
    if not mapping.get("transfer"):
        return None
    weight_col = mapping.get("weight")
    e = numeric_expr(mapping["transfer"])
    if not weight_col:
        row = con.execute(f"""
            SELECT COUNT(*) AS toplam,
                   SUM(CASE WHEN {e} > 0 THEN 1 ELSE 0 END) AS aktarmali,
                   AVG({e}) AS ort_aktarma,
                   AVG(CASE WHEN {e} > 0 THEN {e} END) AS ort_aktarma_aktarmalilar
            FROM {table} WHERE {e} IS NOT NULL
        """).fetchone()
    else:
        we = numeric_expr(weight_col)
        row = con.execute(f"""
            SELECT SUM({we}) AS toplam,
                   SUM(CASE WHEN {e} > 0 THEN {we} ELSE 0 END) AS aktarmali,
                   SUM({e} * {we}) / SUM({we}) AS ort_aktarma,
                   SUM(CASE WHEN {e} > 0 THEN {e} * {we} ELSE 0 END)
                       / NULLIF(SUM(CASE WHEN {e} > 0 THEN {we} ELSE 0 END), 0) AS ort_aktarma_aktarmalilar
            FROM {table} WHERE {e} IS NOT NULL AND {we} IS NOT NULL
        """).fetchone()
    if row is None or not row[0]:
        return None
    toplam, aktarmali, ort_aktarma, ort_aktarma_aktarmalilar = row
    return {
        "toplam_yolculuk": toplam, "aktarmali_yolculuk": (aktarmali if weight_col else int(aktarmali or 0)),
        "aktarmali_oran_pct": (aktarmali or 0) / toplam * 100,
        "ortalama_aktarma_tum": ort_aktarma,
        "ortalama_aktarma_aktarmalilarda": ort_aktarma_aktarmalilar,
    }


def mobility_by_vehicle_ownership(con, table, mapping: dict) -> pd.DataFrame | None:
    """Otomobil sahipligi (0/1/2/3+ arac) grubuna gore kisi bazli
    hareketlilik orani (Rapor Tablo 27 / Grafik 69: 'Arac Sahipligine Gore
    Kisi Bazli Yolculuk Oranlari')."""
    if not has(mapping, "vehicle_count", "person_id"):
        return None
    weight_col = mapping.get("weight")
    ve = numeric_expr(mapping["vehicle_count"])
    pid = ident(mapping["person_id"])
    grp = (f"CASE WHEN {ve} <= 0 THEN '0 Arac' WHEN {ve} = 1 THEN '1 Arac' "
           f"WHEN {ve} = 2 THEN '2 Arac' ELSE '3+ Arac' END")
    if not weight_col:
        df = _fetch(con, f"""
            SELECT {grp} AS Arac_Grubu, COUNT(*) * 1.0 / COUNT(DISTINCT {pid}) AS Hareketlilik_Orani,
                   COUNT(DISTINCT {pid}) AS Kisi_Sayisi
            FROM {table} WHERE {ve} IS NOT NULL GROUP BY 1
        """)
    else:
        # Ayni "hareketlilik orani = yolculuk agirligi / kisi agirligi" mantigi
        # (bkz. mobility_rate) - burada arac sahipligi grubuna gore.
        we = numeric_expr(weight_col)
        df = _fetch(con, f"""
            WITH trip_w AS (
                SELECT {grp} AS Arac_Grubu, SUM({we}) AS trip_w FROM {table}
                WHERE {ve} IS NOT NULL AND {we} IS NOT NULL GROUP BY 1
            ),
            person_w AS (
                SELECT {grp} AS Arac_Grubu, {pid} AS p, ANY_VALUE({we}) AS w FROM {table}
                WHERE {ve} IS NOT NULL AND {we} IS NOT NULL GROUP BY 1, 2
            ),
            person_agg AS (SELECT Arac_Grubu, SUM(w) AS Kisi_Sayisi FROM person_w GROUP BY 1)
            SELECT trip_w.Arac_Grubu, trip_w.trip_w / person_agg.Kisi_Sayisi AS Hareketlilik_Orani,
                   person_agg.Kisi_Sayisi
            FROM trip_w JOIN person_agg ON trip_w.Arac_Grubu = person_agg.Arac_Grubu
        """)
    if df.empty:
        return None
    order = ["0 Arac", "1 Arac", "2 Arac", "3+ Arac"]
    df["_ord"] = df["Arac_Grubu"].apply(lambda x: order.index(x) if x in order else 99)
    return df.sort_values("_ord").drop(columns="_ord")


def transfer_count_distribution(con, table, mapping: dict) -> pd.Series | None:
    """Aktarma sayisina (0, 1, 2, 3+) gore yolculuklarin dagilimi (%)."""
    if not mapping.get("transfer"):
        return None
    weight_col = mapping.get("weight")
    e = numeric_expr(mapping["transfer"])
    n_expr = f"SUM({numeric_expr(weight_col)})" if weight_col else "COUNT(*)"
    w_where = f" AND {numeric_expr(weight_col)} IS NOT NULL" if weight_col else ""
    df = _fetch(con, f"""
        SELECT CASE WHEN {e} <= 0 THEN '0 (Aktarmasiz)' WHEN {e} = 1 THEN '1 Aktarma'
                    WHEN {e} = 2 THEN '2 Aktarma' ELSE '3+ Aktarma' END AS grp, {n_expr} AS n
        FROM {table} WHERE {e} IS NOT NULL{w_where} GROUP BY 1
    """)
    if df.empty:
        return None
    order = ["0 (Aktarmasiz)", "1 Aktarma", "2 Aktarma", "3+ Aktarma"]
    total = df["n"].sum()
    s = pd.Series(df.set_index("grp")["n"] / total * 100)
    return s.reindex([o for o in order if o in s.index])


# ---------------------------------------------------------------- korelasyon (tam veri, SQL CORR)
def correlation_matrix_sql(con, table, numeric_cols: list[str], max_cols: int = 12) -> pd.DataFrame:
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return pd.DataFrame()
    exprs = []
    for i, a in enumerate(cols):
        for b in cols[i:]:
            exprs.append(f"CORR({numeric_expr(a)}, {numeric_expr(b)}) AS c_{i}_{cols.index(b)}")
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {table}").fetchone()
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    k = 0
    for i, a in enumerate(cols):
        for b in cols[i:]:
            j = cols.index(b)
            v = row[k]
            mat.loc[a, b] = v if v is not None else np.nan
            mat.loc[b, a] = v if v is not None else np.nan
            k += 1
    return mat
