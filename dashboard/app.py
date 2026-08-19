# -*- coding: utf-8 -*-
"""
UAP 2040 tarzi Hanehalki / Yolculuk Verisi Analiz Dashboard'u (v3)
================================================================
Cok buyuk / karmasik ham CSV dosyalarini DuckDB ile TAM VERI uzerinde
(pandas'a yuklemeden, cokmeden) isler; numaralandirilmis sutun gruplarini
(orn. aracadet1, aracadet2, ... ya da yas1, cinsiyet1, yas2, cinsiyet2, ...)
otomatik tespit edip birlestirme/uzun-format donusumu sunar; Izmir Ulasim
Ana Plani 2040 raporundaki analiz mantigiyla tablo/grafik uretir; her bolum
birbirinden izole calisir (bir hata tum sayfayi cokertmez); altinda yerel
istatistiksel+ML tabanli bir yapay zeka motoru ve opsiyonel gercek LLM
entegrasyonu calisir.

Calistirma:
    streamlit run dashboard/app.py
"""
from __future__ import annotations
import os
import re
import sys
import time
import traceback

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from utils import analytics as an
from utils import ai_local
from utils import ai_llm
from utils import benchmarks as bm
from utils import column_groups as cg
from utils import data_io
from utils import geo
from utils.mapping import CONCEPTS, auto_guess_mapping, concepts_by_group
from utils import mapping as mapping_mod

st.set_page_config(page_title="UAP 2040 Tarzi Ulasim Veri Dashboard'u",
                    page_icon="🚌", layout="wide")

# NOT: Daha once burada, Streamlit Community Cloud'un ilk baglantida sidebar'i
# kaybetmesini telafi eden bir "watchdog" (otomatik tek seferlik sayfa
# yenileme) denendi. Kaldirildi: her ziyarette script'i sunucu tarafinda IKI
# KEZ calistirdigi icin (ilk yukleme + otomatik yenileme), zaten kisitli olan
# 1GB RAM'i ekstra zorlayip cokme/"connection reset" durumuna katkida
# bulunuyor olabilirdi. Eger sol menu (Veri Kaynagi) ilk acilista gorunmezse,
# sayfayi elle bir kez yenilemek yeterli.

PALETTE = px.colors.qualitative.Set2

# =========================================================== gorsel cila (CSS)
# SADECE kozmetik - hicbir widget mantigini/degerini etkilemez. Renkler
# bilerek DUSUK OPAKLIKLI (rgba, dusuk alpha) secildi ki hem acik hem koyu
# temada (Streamlit'in kendi tema secimine gore) okunakli kalsin - sabit
# koyu/acik metin rengi ATANMAZ, sadece kenarlik/arka plan/gölge eklenir.
st.markdown("""
<style>
/* metrik kartlari: hafif kenarlik + arka plan + yuvarlatilmis kose */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,110,250,0.10), rgba(99,110,250,0.02));
    border: 1px solid rgba(99,110,250,0.22);
    border-radius: 12px;
    padding: 12px 16px 8px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stMetricValue"] { font-weight: 700; }
div[data-testid="stMetricLabel"] { opacity: 0.85; }

/* sekmeler: biraz daha nefes alan, yuvarlatilmis sekme basliklari */
.stTabs [data-baseweb="tab-list"] { gap: 2px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,110,250,0.10);
    font-weight: 600;
}

/* ana baslik: alt cizgi vurgusu */
h1 { padding-bottom: 0.4rem; border-bottom: 3px solid rgba(99,110,250,0.55); }

/* bolum basliklari (### section_title) icin biraz daha nefes payi */
h3 { margin-top: 1.6rem; }

/* expander basliklari: hafif belirginlestir */
div[data-testid="stExpander"] summary {
    font-weight: 500;
}

/* sidebar: hafif ayrac golgesi */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15);
}
</style>
""", unsafe_allow_html=True)
BASE_TABLE = data_io.TABLE
HASH_FUNCS = {duckdb.DuckDBPyConnection: id}


def cq(fn):
    return st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS)(fn)


data_profile = cq(an.data_profile)
household_size_stats = cq(an.household_size_stats)
household_size_by_district = cq(an.household_size_by_district)
gender_distribution = cq(an.gender_distribution)
gender_by_district = cq(an.gender_by_district)
age_pyramid = cq(an.age_pyramid)
avg_age_fn = cq(an.avg_age)
category_distribution = cq(an.category_distribution)
category_by_district = cq(an.category_by_district)
numeric_summary_sql = cq(an.numeric_summary_sql)
income_summary = cq(an.income_summary)
gini_and_lorenz = cq(an.gini_and_lorenz)
income_quintile_distribution = cq(an.income_quintile_distribution)
income_group_purpose = cq(an.income_group_purpose)
income_group_mode = cq(an.income_group_mode)
vehicle_ownership_distribution = cq(an.vehicle_ownership_distribution)
vehicles_per_1000_by_district = cq(an.vehicles_per_1000_by_district)
mobility_rate = cq(an.mobility_rate)
mobility_rate_by_district = cq(an.mobility_rate_by_district)
purpose_distribution = cq(an.purpose_distribution)
mode_distribution = cq(an.mode_distribution)
purpose_mode_crosstab = cq(an.purpose_mode_crosstab)
numeric_by_group_sql_cached = cq(an.numeric_by_group_sql)
duration_stats_by = cq(an.duration_stats_by)
hourly_distribution = cq(an.hourly_distribution)
hourly_distribution_by_purpose = cq(an.hourly_distribution_by_purpose)
zone_internal_external = cq(an.zone_internal_external)
od_matrix = cq(an.od_matrix)
raw_count_sql = cq(an.raw_count_sql)
age_by_district = cq(an.age_by_district)
age_group_distribution = cq(an.age_group_distribution)
income_by_district = cq(an.income_by_district)
gini_by_district = cq(an.gini_by_district)
vehicle_ownership_rate_by_district = cq(an.vehicle_ownership_rate_by_district)
home_work_district_comparison = cq(an.home_work_district_comparison)
mode_flag_split = cq(an.mode_flag_split)
duration_frequency_distribution = cq(an.duration_frequency_distribution)
duration_quality_flags = cq(an.duration_quality_flags)
range_quality_flags = cq(an.range_quality_flags)
iqr_outlier_flags = cq(an.iqr_outlier_flags)
binned_time_distribution = cq(an.binned_time_distribution)
hourly_distribution_by_group = cq(an.hourly_distribution_by_group)
confidence_interval_mean = cq(an.confidence_interval_mean)
confidence_interval_proportion = cq(an.confidence_interval_proportion)
crosstab_sql_cached = cq(an.crosstab_sql)
transfer_stats = cq(an.transfer_stats)
transfer_count_distribution = cq(an.transfer_count_distribution)
mobility_by_vehicle_ownership = cq(an.mobility_by_vehicle_ownership)
correlation_insights = cq(ai_local.correlation_insights)
detect_row_anomalies_sql = cq(ai_local.detect_row_anomalies_sql)
feature_importance_analysis = cq(ai_local.feature_importance_analysis)
# cluster_districts bir DuckDB baglantisi almaz (girdisi zaten hesaplanmis
# kucuk bir pandas DataFrame'dir), bu yuzden HASH_FUNCS'a gerek yok - duz
# st.cache_data yeterli. Onceden hic onbelleklenmiyordu, ilgisiz bir
# sekmede yapilan HER etkilesimde KMeans+silhouette skoru bastan
# calisiyordu.
cluster_districts = st.cache_data(show_spinner=False)(ai_local.cluster_districts)
repeated_value_consistency = cq(an.repeated_value_consistency)


# =========================================================================
# Asagidaki kucuk sorgular eskiden sekme govdelerinde DOGRUDAN con.execute(...)
# olarak (onbelleksiz) calisiyordu. st.tabs() TUM sekme govdelerini HER
# widget etkilesiminde (baska bir sekmede yapilsa bile) yeniden calistirdigi
# icin bu sorgular gereksiz yere tekrar tekrar calisiyordu. Her biri
# ONCEKI SATIR SATIR AYNI SQL METNini uretir - sadece cq() ile ayni girdiler
# icin sonuc onbellekten donuyor; hesaplanan degerler DEGISMEZ.
@cq
def q_sample_size_by_district(con, table, district_col, id_col):
    de = an.cat_expr(district_col)
    cnt_expr = f"COUNT(DISTINCT {an.ident(id_col)})" if id_col else "COUNT(*)"
    return con.execute(f"""
        SELECT {de} AS Ilce, {cnt_expr} AS Orneklem_Buyuklugu
        FROM {table} WHERE {de} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC
    """).fetchdf()


@cq
def q_zone_survey_counts(con, table, zone_col):
    ze = an.cat_expr(zone_col)
    return con.execute(f"""
        SELECT {ze} AS Mahalle, COUNT(*) AS Anket_Sayisi FROM {table}
        WHERE {ze} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 40
    """).fetchdf()


@cq
def q_student_count_by_district(con, src, district_col, student_col):
    de5, se5 = an.cat_expr(district_col), an.cat_expr(student_col)
    return con.execute(f"""
        SELECT {de5} AS Ilce, {se5} AS Durum, COUNT(*) AS Kisi_Sayisi
        FROM {src} WHERE {de5} IS NOT NULL AND {se5} IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """).fetchdf()


@cq
def q_numeric_values(con, table, numeric_col, positive_only=False):
    e = an.numeric_expr(numeric_col)
    extra = f" AND {e} > 0" if positive_only else ""
    return con.execute(f"SELECT {e} AS val FROM {table} WHERE {e} IS NOT NULL{extra}").fetchdf()


@cq
def q_district_sample_sizes(con, table, district_col, dedup_key_col):
    de = an.cat_expr(district_col)
    n_expr = f"COUNT(DISTINCT {an.ident(dedup_key_col)})" if dedup_key_col else "COUNT(*)"
    return con.execute(f"""
        SELECT {de} AS ilce, {n_expr} AS n FROM {an.ident(table)}
        WHERE {de} IS NOT NULL GROUP BY 1
    """).fetchdf()


@cq
def q_hourly_filtered(con, table, start_time_col, mode_col, values: tuple, mode_siblings: tuple = ()):
    """mode_siblings: DOGRULUK ICIN KRITIK - bkz. mode_flag_split'teki ayni
    konudaki commit notu. 'mode_col' coklu-bacakli (transfer) bir grubun
    tek bir uyesiyse (orn. "yol_arac1"), diger bacaklar (yol_arac2, 3...)
    burada da havuzlanir - baslangic saati o bacaklarin TUMUNDE ayni
    kaldigindan (tek yolculuk), tekrarlanarak dogru sekilde eslenir."""
    he = an.hour_expr(start_time_col)
    vals_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    if mode_siblings and len(mode_siblings) > 1:
        union_sql = " UNION ALL ".join(
            f"SELECT {an.ident(start_time_col)}, {an.cat_expr(sib)} AS {an.ident(mode_col)} FROM {table}"
            for sib in mode_siblings
        )
        table = f"({union_sql}) AS _pooled"
    me = an.cat_expr(mode_col)
    return con.execute(f"""
        SELECT {he} AS h, COUNT(*) AS n FROM {table}
        WHERE {he} IS NOT NULL AND {he} BETWEEN 0 AND 23 AND {me} IN ({vals_sql})
        GROUP BY 1
    """).fetchdf()


@cq
def q_transfer_duration_comparison(con, table, transfer_col, duration_col):
    e_t = an.numeric_expr(transfer_col)
    e_d = an.numeric_expr(duration_col)
    return con.execute(f"""
        SELECT CASE WHEN {e_t} > 0 THEN 'Aktarmali' ELSE 'Aktarmasiz' END AS Grup,
               AVG({e_d}) AS "Ortalama Sure (dk)"
        FROM {table} WHERE {e_t} IS NOT NULL AND {e_d} IS NOT NULL
        GROUP BY 1
    """).fetchdf()


@cq
def q_avg_numeric_by_district(con, table, district_col, numeric_col):
    return con.execute(f"""
        SELECT {an.ident(district_col)} AS Ilce, AVG({an.numeric_expr(numeric_col)}) AS geliri
        FROM {table} WHERE {an.ident(district_col)} IS NOT NULL GROUP BY 1
    """).fetchdf()


@st.cache_data(show_spinner=False)
def compute_rankings(cache_key: str, _columns: list[str]) -> dict[str, list[str]]:
    """Her kavram icin en alakali (en fazla 40) aday sutunu hesaplar ve
    onbellekler. Binlerce sutunlu dosyalarda TUM sutunlari her secim
    kutusuna basmak (5000+ ogeli dropdown) arayuzu ciddi sekilde
    yavaslattigi icin bu on-filtreleme sarttir."""
    return {c: mapping_mod.ranked_candidates(_columns, c, top_k=40) for c in CONCEPTS}


@st.cache_resource(show_spinner=False)
def get_work_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    return data_io.open_work_connection(db_path)


@st.cache_resource(show_spinner=False)
def ingest(path: str, force: bool) -> data_io.DataSource:
    return data_io.get_or_build(path, force_rebuild=force)


def table_columns(con, table: str) -> list[str]:
    try:
        return con.execute(f"DESCRIBE {an.ident(table)}").fetchdf()["column_name"].tolist()
    except Exception:  # noqa: BLE001
        return []


def table_dtypes(con, table: str) -> dict[str, str]:
    try:
        d = con.execute(f"DESCRIBE {an.ident(table)}").fetchdf()
        return dict(zip(d["column_name"], d["column_type"]))
    except Exception:  # noqa: BLE001
        return {}


# =========================================================== hata izolasyonu
ERROR_LOG: list[tuple[str, str]] = []


def safe(label: str, fn, *args, **kwargs):
    """Bir analiz/gorsellestirme adimini calistirir; hata olursa TUM SAYFAYI
    COKERTMEDEN kucuk bir uyari gosterir ve None doner - boylece bir tablo/
    grafik bozuk olsa bile geri kalan her sey calismaya devam eder."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        ERROR_LOG.append((label, f"{e}\n{traceback.format_exc(limit=2)}"))
        st.warning(f"⚠️ **{label}** hesaplanamadi/gosterilemedi: {e}")
        return None


def section_title(txt: str, icon: str = ""):
    st.markdown(f"### {icon} {txt}")


def sample_size_caption(con, table: str, district_col: str | None, dedup_key_col: str | None = None,
                          threshold: int = 30):
    """Ilce bazli bir kirilimin GUVENILIRLIGI hakkinda tutarli bir uyari
    gosterir (NHTS/MPO pratiginde standart: kucuk orneklemli kirilimlar
    'gurultulu'/guvenilmez olur - bkz. gini_by_district'teki ayni fikir,
    burada TUM ilce-bazli grafiklere TUTARLI sekilde uygulanir). Hicbir
    veriyi FILTRELEMEZ/gizlemez - SADECE hangi ilcelerin dusuk orneklemli
    oldugunu bilgilendirici bir notla belirtir, karar kullaniciya kalir."""
    if not district_col:
        return
    df = safe("Orneklem buyuklugu kontrolu", q_district_sample_sizes, con, table, district_col, dedup_key_col)
    if df is None:
        return
    low = df[df["n"] < threshold]
    if not low.empty:
        st.caption(
            f"⚠️ {len(low)} ilcede gozlem sayisi {threshold}'un altinda "
            f"({', '.join(low.sort_values('n')['ilce'].astype(str).head(6))}"
            f"{'...' if len(low) > 6 else ''}) - bu ilcelerin sonuclari istatistiksel "
            "olarak daha az guvenilir olabilir, yorumlarken dikkatli olun."
        )


def render_district_map(df: pd.DataFrame | None, district_col: str, value_col: str,
                          title: str, color_scale: str = "Blues", key: str = ""):
    """Ilce bazli bir DataFrame'i (orn. income_by_district'in ciktisi)
    GERCEK bir Turkiye ilce sinirlari haritasinda (choropleth) gosterir
    (bkz. utils/geo.py). Mevcut cubuk/bar grafigin YANINA EK bir gorunum
    olarak eklenir - hicbir mevcut grafigi degistirmez/kaldirmaz.

    DURUSTLUK ICIN KRITIK: bir ilce adi harita sinirlariyla eslesmiyorsa
    (yanlis yazim) ya da BIRDEN FAZLA ilde ayni adi tasiyorsa (orn.
    'Merkez'), o ilce haritada GOSTERILMEZ - tahmini/yanlis bir konuma
    ASLA yerlestirilmez; bunun yerine ayri bir notla acikca belirtilir."""
    if df is None or df.empty or district_col not in df.columns or value_col not in df.columns:
        return
    names = df[district_col].astype(str).tolist()
    geojson, unmatched, ambiguous = geo.match_districts(names)
    if not geojson["features"]:
        return
    matched_names = {f["properties"]["name"] for f in geojson["features"]}
    plot_df = df[df[district_col].astype(str).isin(matched_names)]
    if plot_df.empty:
        return
    with st.expander(f"🗺️ {title} — harita gorunumu", expanded=False):
        fig = px.choropleth(
            plot_df, geojson=geojson, locations=district_col, featureidkey="properties.name",
            color=value_col, color_continuous_scale=color_scale, hover_name=district_col,
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
        st.plotly_chart(fig, width="stretch", key=f"map_{key}")
        notes = []
        if unmatched:
            notes.append(f"{len(unmatched)} ilce harita sinirlarinda bulunamadi: " + ", ".join(unmatched[:8]))
        if ambiguous:
            notes.append(f"{len(ambiguous)} ilce adi birden fazla ilde bulundugundan (hangisi oldugu "
                         "belli olmadigindan) haritaya eklenmedi: " + ", ".join(ambiguous[:8]))
        if notes:
            st.caption("ℹ️ " + " | ".join(notes))


def render_range_quality(con, table, col, low, high, label, key):
    """duration_quality_flags'teki 'Veri Kalitesi Kontrolu' ile AYNI desenin
    genel surumu (bkz. utils/analytics.py::range_quality_flags docstring'i -
    CMAP My Daily Travel'daki gecerli-aralik-disi kayit raporlamasi mantigi).
    SADECE BILGILENDIRME - hicbir satiri silmez/mevcut hesaplari degistirmez."""
    if not col:
        return
    r = safe(f"Veri Kalitesi ({label})", range_quality_flags, con, table, col, low, high, label)
    if not r:
        return
    with st.expander(f"🔍 Veri Kalitesi Kontrolu ({label})"):
        st.caption(
            f"{low}-{high} araligi disindaki degerler ({label} icin makul/gecerli sinir) ayrica "
            "raporlanir - hicbir satir silinmez, diger hesaplarinizi DEGISTIRMEZ."
        )
        c1, c2 = st.columns(2)
        c1.metric("Toplam Gecerli Kayit", f"{r['toplam']:,}")
        c2.metric(f"{low}-{high} Disinda", f"{r['aralik_disi']:,}", f"%{r['aralik_disi_yuzde']}")
        if r["aralik_disi"]:
            st.caption(f"⚠️ {r['aralik_disi']:,} kayitta {label.lower()} {low}-{high} araliginin disinda - "
                       "veri girisi hatasi olabilir, kontrol etmenizi oneririz.")
        else:
            st.success(f"✅ {label} alaninda {low}-{high} araliginin disinda kayit tespit edilmedi.")


def render_iqr_quality(con, table, col, label, key):
    """Gelir gibi 'normal' araligi sabit bir esikle tanimlanamayacak alanlar
    icin IQR-tabanli istatistiksel aykiri deger raporu (bkz.
    utils/analytics.py::iqr_outlier_flags). SADECE BILGILENDIRME."""
    if not col:
        return
    r = safe(f"IQR Veri Kalitesi ({label})", iqr_outlier_flags, con, table, col, label)
    if not r:
        return
    with st.expander(f"🔍 Veri Kalitesi Kontrolu ({label}, istatistiksel)"):
        st.caption(
            "IQR (ceyrekler-arasi aralik) yontemiyle istatistiksel aykiri deger tespiti - "
            "sabit bir esik yerine verinin KENDI dagilimina gore hesaplanir. SADECE "
            "BILGILENDIRME - hicbir satir silinmez, diger hesaplarinizi DEGISTIRMEZ."
        )
        c1, c2 = st.columns(2)
        c1.metric("Toplam Gecerli Kayit", f"{r['toplam']:,}")
        c2.metric("Istatistiksel Aykiri Deger", f"{r['aykiri']:,}", f"%{r['aykiri_yuzde']}")
        st.caption(f"Normal kabul edilen aralik: {r['alt_esik']:,} — {r['ust_esik']:,} "
                   f"(Q1={r['q1']:,}, Q3={r['q3']:,}, IQR={r['iqr']:,}).")
        if not r["aykiri"]:
            st.success(f"✅ {label} alaninda belirgin istatistiksel aykiri deger tespit edilmedi.")


def build_excel_report_bytes(meta: dict, sheets: list[tuple[str, "pd.DataFrame | pd.Series | None"]]) -> bytes:
    """Sekmelerde ZATEN hesaplanmis (degismeyen) verilerden bir .xlsx (Excel)
    dosyasi uretir - kullanicilarin (orn. belediye yetkilileri) HTML/markdown
    yerine dogrudan Excel'de acabilecegi bir cikti istegi uzerine eklendi.
    HER sayfa, cagiran kodun ELINDEKI ayni Series/DataFrame'den TAZE yazilir -
    yani dosyadaki sayilar ekrandakilerle HER ZAMAN birebir aynidir."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        meta_df = pd.DataFrame(list(meta.items()), columns=["Alan", "Deger"])
        meta_df.to_excel(writer, sheet_name="Ozet", index=False)
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#0e6f6f", "font_color": "white"})
        ws0 = writer.sheets["Ozet"]
        for col_idx, col in enumerate(meta_df.columns):
            ws0.write(0, col_idx, col, header_fmt)
            ws0.set_column(col_idx, col_idx, 40)
        used_names = set()
        for name, obj in sheets:
            if obj is None:
                continue
            df = obj.to_frame() if isinstance(obj, pd.Series) else obj
            if df is None or df.empty:
                continue
            # Excel sayfa adlari en fazla 31 karakter + tekrarsiz olmali
            safe_name = re.sub(r"[\[\]\:\*\?/\\]", " ", name)[:31].strip() or "Sayfa"
            base_name, i = safe_name, 1
            while safe_name in used_names:
                i += 1
                suffix = f" {i}"
                safe_name = base_name[: 31 - len(suffix)] + suffix
            used_names.add(safe_name)
            out_df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
            out_df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            for col_idx, col in enumerate(out_df.columns):
                ws.write(0, col_idx, str(col), header_fmt)
                ws.set_column(col_idx, col_idx, 22)
    return buf.getvalue()


def build_visual_report_html(title: str, meta_lines: list[str], sections: list[tuple[str, "go.Figure | None"]]) -> str:
    """Sekmelerde ZATEN hesaplanmis (degismeyen) verilerden, TAMAMEN cevrimdisi
    acilabilen (plotly.js gomulu, internet gerektirmeyen) tek bir HTML rapor
    dosyasi uretir. 'sections' listesindeki HER figur, cagiran kodun ELINDEKI
    (bu sayfada zaten gosterilmis) ayni Series/DataFrame'lerden TAZE olarak
    cizilir - yani rapordaki sayilar, ekrandaki sekmelerdeki sayilarla
    her zaman BIREBIR aynidir (yeniden hesaplama/farkli bir yol YOKTUR).
    Kullanici, indirdigi HTML dosyasini tarayicida acip 'Yazdir > PDF olarak
    kaydet' ile PDF'e de cevirebilir."""
    import html as _html
    parts = []
    first = True
    for sec_title, fig in sections:
        if fig is None:
            continue
        chart_html = fig.to_html(full_html=False, include_plotlyjs=(True if first else False),
                                  config={"displaylogo": False})
        first = False
        parts.append(f'<section><h2>{_html.escape(sec_title)}</h2>{chart_html}</section>')
    meta_html = "".join(f"<li>{_html.escape(m)}</li>" for m in meta_lines)
    body = "\n".join(parts) if parts else "<p>Gosterilecek grafik bulunamadi (yeterli sutun eslesmemis olabilir).</p>"
    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; max-width: 980px;
         margin: 0 auto; padding: 2rem 1.5rem 4rem; color: #1a2226; background: #fff; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
  .meta {{ color: #566; font-size: 0.9rem; margin-bottom: 2rem; }}
  .meta ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
  section {{ margin-bottom: 2.6rem; page-break-inside: avoid; }}
  section h2 {{ font-size: 1.1rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }}
  @media print {{ section {{ page-break-inside: avoid; }} }}
</style></head>
<body>
  <h1>{_html.escape(title)}</h1>
  <div class="meta"><ul>{meta_html}</ul></div>
  {"".join(parts) if parts else body}
</body></html>"""


def report_item(label: str, title: str):
    """UAP 2040 raporundaki bir Tablo/Grafik basligini AYNI numarayla ve
    AYNI sirayla gosterir (orn. 'Grafik 7. Ilcelere Gore Ortalama Yas').
    Butun rapor-esli sekmelerde tutarli bir gorunum saglar."""
    st.markdown(f"#### {label}. {title}")


def report_skip_note(reason: str = "harita/CBS (cografi) verisi"):
    """Raporda yer alan ama JENERIK bir CSV'den (harita/CBS katmani
    olmadan) otomatik uretilemeyecek gorseller (Sekil'ler - TAZ haritalari,
    saha fotograflari vb.) icin durustce belirtilen bir not gosterir.
    Sahte/yanlis bir gorsel uretmek yerine gercek sinirin acikca
    soylenmesi tercih edilir."""
    st.caption(f"ℹ️ Bu ogenin orijinali {reason} gerektirir; yuklenen tablo verisinden otomatik uretilemez.")


def demographic_scope_warning(active_table: str):
    """DOGRULUK ICIN KRITIK: 'Yolculuk Tablosu' (kisi+yolculuk JOIN'i), o gun
    HIC yolculuk yapmamis kisileri (kucuk cocuklar, ev disina cikmayanlar
    vb.) DOGASI GEREGI icermez - INNER/LEFT JOIN sonucu sadece en az bir
    yolculuk kaydi olan kisiler satir olarak var olur. Bu tablo aktifken
    Cinsiyet/Yas/Egitim/Istihdam gibi SAF DEMOGRAFIK dagilimlar sessizce
    YANLIS (sadece yolculuk yapanlara gore carpitilmis) sonuc verir - gercek
    veride bu, yas ortalamasini ~1 yil, cinsiyet dagilimini ~6 puan
    kaydırdığı olculup dogrulandi. Bu yuzden boyle bir tablo aktifken bu
    sekmelerde acik bir uyari gosterilir; TUM hanehalki icin 'Kisi Tablosu'
    secilmelidir."""
    if "yolculuk" in active_table.lower():
        st.warning(
            "⚠️ Aktif tablo bir **Yolculuk Tablosu** - bu, o gun HIC yolculuk "
            "yapmamis kisileri (once yolculuk kaydi olmadigi icin) icermez. "
            "Bu sekmedeki dagilimlar TUM hanehalki icin degil, sadece "
            "yolculuk yapanlar icin dogru olur. TUM bireyler icin sol "
            "menudeki 'Hangi tabloda analiz yapilsin?' alanindan "
            "**'Kisi Tablosu'** secin."
        )


def missing_note(*concepts):
    """DOGRULUK/NET MESAJ ICIN KRITIK: eskiden GECIRILEN TUM kavramlar
    (bu bolumun ihtiyac duydugu her sey) koşulsuz listeleniyordu - bir
    grafik 2 kavram gerektirip SADECE biri eksikse bile, kullaniciya
    "ikisi de eksik" gibi YANLIS/KAFA KARISTIRICI bir mesaj gosteriliyordu
    (gercek kullanici geri bildirimiyle yakalandi: "Ilce/Bolge (Baslangic)"
    aslinda eslenmisken bu mesajda "eksik" olarak goruntuleniyordu). Artik
    SADECE gercekten haritalanmamis (mapping'de degeri olmayan) kavramlar
    listelenir."""
    global mapping
    truly_missing = [c for c in concepts if not mapping.get(c)]
    if not truly_missing:
        # concepts'in TUMU aslinda eslenmis - o zaman fonksiyonun None/bos
        # donmesinin baska bir nedeni var (orn. secilen filtrelerle eslesen
        # veri yok) - yaniltici "sutun eksik" mesaji YERINE genel bir not.
        st.info("Bu bolum icin gerekli sutunlarin hepsi eslestirilmis, ama "
                "secili veriyle bir sonuc uretilemedi (orn. filtreyle "
                "eslesen kayit yok ya da tum degerler bos).")
        return
    labels = ", ".join(CONCEPTS[c][0] for c in truly_missing)
    st.info(f"Bu bolum icin sutun eslestirmesi eksik: **{labels}**. "
            f"Yukaridaki 'Sutun Eslestirme' bolumunden ilgili sutunlari secin.")


_MAPPING_RENDERED_THIS_RUN: set[str] = set()


def render_mapping_grid(mapping: dict, concept_list: list[str], active_columns: list[str],
                          rankings: dict, key_ns: str, cols_per_row: int = 4):
    """Verilen kavram listesi icin KOMPAKT (satir/sutun izgarali), o anki
    sekmeyle sinirli bir eslestirme paneli cizer. `mapping` sozlugu YERINDE
    (in-place) guncellenir - boylece ayni script calismasi icinde daha sonra
    gelen kod (ayni sekme ya da sonraki sekmeler) guncel degeri hemen gorur.

    KRITIK DUZELTME (Streamlit AppTest ile dogrulandi - bkz. commit notu): ayni
    kavram (ör. "district"/ilce, "mode"/ulasim turu, "income"/gelir) BIRDEN
    FAZLA sekmenin kendi render_mapping_grid cagrisinda tekrar gecebiliyor
    (bu kasitli - "bir kez eslestir, her yerde kullanilsin" seklinde
    tasarlandi, bkz. "kimlik" sekmesindeki aciklama). AMA her cagri kendi
    BAGIMSIZ (farkli key_ns'li) st.selectbox'ini olusturuyordu - Streamlit
    tabs'lerin TUMU HER rerun'da (gorunmez olsa bile) calistigi icin, ayni
    kavram icin sekme sayisi kadar BAGIMSIZ widget ortaya cikiyor, HER BIRI
    KENDI eski/degismemis degerini ayni paylasilan `mapping` sozlugune GERI
    YAZIYORDU - script sirasinda EN SON calisan kopya kazaniyordu. Sonuc:
    kullanici bir sekmede eslestirmeyi degistirse bile, o kavramin GECTIGI
    sonraki bir sekme scripti calisirken kendi eski degerini sessizce GERI
    YUKLUYOR ve kullanicinin degisikligini SILIYORDU (gercek AppTest
    kosumuyla dogrulandi: spatial sekmesinde "district"i degistirmek, script
    sirasinda sonra gelen "duration" sekmesi tarafindan aninda geri
    alindi). Duzeltme: her kavram bir rerun'da SADECE ILK karsilasildigi
    yerde gercek/duzenlenebilir widget olarak cizilir; ayni kavram BASKA bir
    sekmede tekrar istenirse, orada SADECE salt-okunur bir ozet gosterilir -
    boylece TEK bir dogru kaynak (single source of truth) garanti edilir ve
    hicbir degisiklik sessizce kaybolmaz."""
    if not concept_list:
        return
    # DOGRULUK ICIN KRITIK: anahtar-kelime tabanli otomatik tahmin (ranked_candidates)
    # gercek dosyalarda HER ZAMAN bir eslesme bulamaz (orn. yolculuk amacini
    # tutan sutun "yol_bas" gibi anlamsiz/kisaltilmis bir ad tasiyorsa, hicbir
    # anahtar kelimeyle eslesmez ve aday listesi TAMAMEN BOS kalir). Eskiden
    # arama kutusu sadece >200 sutunlu dosyalarda gosteriliyordu; orta
    # buyuklukteki (orn. 75 sutunlu, uzun formata cevrilmis bir "Yolculuk
    # Tablosu" gibi) dosyalarda kullanici boyle bir durumda o kavrami HICBIR
    # SEKILDE elle de eslestiremiyordu (dropdown bomboş). Bu yuzden arama
    # kutusu artik cok kucuk (<=15 sutunlu) dosyalar disinda HER ZAMAN gosterilir.
    search_q = ""
    if len(active_columns) > 15:
        search_q = st.text_input(
            "🔎 Tum sutunlarda ara (bu bolum icin, aday listesi bos ya da yanlissa kullanin)",
            key=f"search_{key_ns}", placeholder="orn. gelir, yas, ilce, yol_bas...",
        )
    search_results = mapping_mod.search_columns(active_columns, search_q, limit=200) if search_q else None

    cols_ui = st.columns(cols_per_row)
    for i, concept in enumerate(concept_list):
        label, desc, required, ctype, _ = CONCEPTS[concept]
        current = mapping.get(concept)
        with cols_ui[i % cols_per_row]:
            if concept in _MAPPING_RENDERED_THIS_RUN:
                # bkz. fonksiyon docstring'i: bu kavram bu rerun'da BASKA bir
                # sekmede ZATEN duzenlenebilir widget olarak cizildi - burada
                # IKINCI bir bagimsiz selectbox olusturmak, kullanicinin oradaki
                # secimini SESSIZCE SILEBILIRDI. Bunun yerine mevcut degeri
                # salt-okunur gosteriyoruz - tek dogru kaynak boylece korunur.
                #
                # ONEMLI UX DUZELTMESI (gercek kullanici geri bildirimiyle
                # yakalandi): salt-okunur gosterimin TEK BASINA yeterli
                # olmadigi ortaya cikti - kullanici o an baktigi sekmede
                # kavrami YANLIS/eksik gorup duzeltmek istediginde, "baska
                # bir sekmede eslestirilir" notu ONA NEREDE oldugunu
                # SOYLEMIYOR, degistirme imkani da vermiyordu ("baslangic ve
                # varis noktalarini secemiyorum" sikayeti). Asagidaki kucuk
                # "degistir" acilir paneli, AI-onerisi mekanizmasiyla AYNI
                # guvenli yontemi (_map_pending_override) kullanarak, HER
                # SEKMEDEN degisiklik yapilabilmesini saglar - boylece hem
                # "tek dogru kaynak" korunur (cakisma/silinme riski YOK) hem
                # de kullanici sikismiyor.
                st.caption(f"**{label}**")
                st.caption(f"`{current}`" if current else "_(esletirilmedi)_")
                with st.expander("✏️ Burada degistir", expanded=False):
                    alt_candidates = search_results if search_results is not None else rankings.get(concept, [])
                    if current and current not in alt_candidates:
                        alt_candidates = [current] + alt_candidates
                    alt_options = ["(Yok)"] + alt_candidates
                    alt_idx = alt_options.index(current) if current in alt_options else 0
                    alt_key = f"altmap_{key_ns}_{concept}"
                    alt_sel = st.selectbox("Yeni deger", alt_options, index=alt_idx, key=alt_key,
                                            label_visibility="collapsed")
                    alt_new = None if alt_sel == "(Yok)" else alt_sel
                    if alt_new != current:
                        mapping[concept] = alt_new
                        st.session_state[map_key][concept] = alt_new
                        st.session_state.setdefault("_map_pending_override", {})[concept] = alt_new
                        st.caption("✅ Guncellendi - diger sekmelerde de gecerli olacak.")
                continue
            _MAPPING_RENDERED_THIS_RUN.add(concept)
            widget_key = f"map_{key_ns}_{concept}"
            # bkz. AI-onerisi uygulama blogundaki commit notu: bu kavram icin
            # PROGRAMATIK (AI onerisi gibi) bekleyen bir deger varsa, widget
            # olusturulmadan HEMEN ONCE kendi session_state'ine yazilir -
            # boylece widget'in kendi eski degeri (varsa) bu atamayi
            # SESSIZCE gec alamaz. (Streamlit, key'in session_state'i widget
            # olusturulmadan once elle atanirsa index= parametresini guvenle
            # yok sayar - hata firlatmaz, AppTest ile dogrulandi.)
            pending = st.session_state.get("_map_pending_override")
            just_overridden = bool(pending and concept in pending)
            if just_overridden:
                ov = pending.pop(concept)
                st.session_state[widget_key] = "(Yok)" if not ov else ov
                current = ov
            candidates = search_results if search_results is not None else rankings.get(concept, [])
            if current and current not in candidates:
                candidates = [current] + candidates
            options = ["(Yok)"] + candidates
            if just_overridden:
                # session_state az once elle atandi - index= de gecirmek
                # Streamlit'te zararsiz ama gereksiz bir uyari kaydina yol
                # aciyor (deger yine de dogru uygulaniyor); temiz loglar icin
                # bu durumda index atlanir.
                sel = st.selectbox(label, options, key=widget_key, help=desc)
            else:
                idx = options.index(current) if current in options else 0
                sel = st.selectbox(label, options, index=idx, key=widget_key, help=desc)
        mapping[concept] = None if sel == "(Yok)" else sel

    mapped_here = sum(1 for c in concept_list if mapping.get(c))
    st.caption(f"✅ {mapped_here}/{len(concept_list)} sutun eslestirildi.")


# =========================================================== sidebar: veri yukleme
st.sidebar.title("📂 Veri Kaynagi")
mode = st.sidebar.radio(
    "Ham CSV nasil yuklensin?",
    ["Dosya Yolu Gir (onerilen, cok buyuk dosyalar icin)", "Dosya Yukle"],
    help="'Dosya Yolu Gir' secilirse dosya hicbir zaman tarayiciya/belleğe tam "
         "yuklenmez; DuckDB diskten parca parca okuyup TAM VERIYI islemek uzere "
         "kalici bir tabloya donusturur (bir kereye mahsus).",
)

path: str | None = None
if mode == "Dosya Yukle":
    up = st.sidebar.file_uploader("CSV dosyasi", type=["csv", "txt"])
    if up is not None:
        with st.spinner("Dosya diske yaziliyor..."):
            path = data_io.materialize_upload(up)
else:
    path = st.sidebar.text_input("CSV dosya yolu (tam yol)",
                                  placeholder=r"C:\Users\firuz\Downloads\veri.csv")

force_rebuild = st.sidebar.checkbox("Onbellegi yoksay, yeniden isle", value=False,
                                     help="Dosya degismedigi halde sorun yasiyorsaniz isaretleyin.")

if not path:
    st.title("🚌 UAP 2040 Tarzi Ulasim Veri Dashboard'u")
    st.markdown(
        "Bu dashboard, **Izmir Ulasim Ana Plani 2040 (UAP 2040) Hanehalki Ulasim Arastirmasi** "
        "raporunun analiz mantigini sizin kendi ham CSV verinize otomatik uygular — "
        "**herhangi bir sehrin** benzer yapidaki anket dosyasi icin de calisir."
    )
    st.info("👈 **Baslamak icin sol menuden bir CSV yolu girin** (ya da kucuk/orta boy dosyalar icin yukleyin).")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 🧩 Otomatik Tespit")
        st.caption("`aracadet1, aracadet2...` gibi numarali sutunlarinizi otomatik bulur, "
                   "dogru analiz formatina cevirir.")
    with c2:
        st.markdown("#### 🤖 Yapay Zeka Destegi")
        st.caption("Sutun adlariniz farkli/beklenmedikse, yapay zeka eslestirmeyi ve "
                   "kod anlamlarini (1=Erkek gibi) tamamlamaya yardim eder.")
    with c3:
        st.markdown("#### 🗺️ Gercek Ilce Haritasi")
        st.caption("Ilce bazli sonuclariniz, gercek Turkiye ilce sinirlariyla "
                   "renkli bir haritada da gosterilir.")
    with c4:
        st.markdown("#### ✅ Dogrulanmis Hesaplar")
        st.caption("Tum formuller (Gini, agirlikli ortalama vb.) bagimsiz referans "
                   "hesaplarla test edilip dogrulanmistir.")
    st.stop()

# ----------------------------------------------------------------- ingest
t_start = time.time()
try:
    with st.spinner("Veri hazirlaniyor (ilk yuklemede biraz surebilir; sonraki acilislar hizli olacak)..."):
        ds = ingest(path, force_rebuild)
except data_io.IngestError as e:
    st.sidebar.error(f"❌ Veri yuklenemedi: {e}")
    st.error(
        "CSV yuklenirken bir sorunla karsilasildi. Olasi nedenler: dosya yolu "
        "yanlis, dosya baska bir programda acik, ya da dosya cok agir bozuk "
        "bir bicimde. 'Onbellegi yoksay, yeniden isle' kutusunu isaretleyip "
        "tekrar deneyebilir ya da dosyanin ilk birkac satirini kontrol "
        "edebilirsiniz."
    )
    st.stop()
except Exception as e:  # noqa: BLE001
    st.sidebar.error("❌ Beklenmeyen bir hata olustu.")
    st.exception(e)
    st.stop()

if ds.from_cache:
    st.sidebar.success(f"✅ Onbellekten yuklendi: **{ds.total_rows:,}** satir, {len(ds.columns)} sutun.")
else:
    st.sidebar.success(
        f"✅ Islendi: **{ds.total_rows:,}** satir, {len(ds.columns)} sutun "
        f"({ds.ingest_seconds:.1f} sn). Bir sonraki acilista bu adim atlanacak."
    )
for w in ds.warnings:
    st.sidebar.warning(w)

con = get_work_connection(ds.db_path)
# Uzun-format (melt) sorgulari, "raw_data" VIEW katmani yerine DOGRUDAN
# ATTACH edilmis ham tabloya karsi calistirilir. Cok genis (binlerce
# sutunlu) dosyalarda 100+ dallı UNION ALL sorgusu bu view katmanindan
# gectiginde DuckDB'nin sorgu planlamasi 30+ saniyeye kadar cikabiliyordu;
# dogrudan ham tabloya karsi ayni sorgu <1 saniyede tamamlaniyor (olculdu).
# Birlestirme (combine) ile turetilen sutunlar hicbir zaman "_1", "_1_2" gibi
# numarali bir desene uymadigindan, melt bloklarinin bu katmani atlamasi
# guvenlidir (birlestirilen sutunlar zaten melt'e taban olarak girmiyor).
MELT_SOURCE = f"src.{BASE_TABLE}"

# =========================================================== 🧩 sutun gruplari
st.sidebar.markdown("---")
st.sidebar.title("🧩 Sutun Gruplari")

all_groups = safe("Sutun grubu tespiti", cg.detect_all_groups, ds.columns) or {"single": {}, "double": {}, "plain": ds.columns}
single_groups = all_groups.get("single", {})
double_groups = all_groups.get("double", {})

# HANGI HAM SUTUNUN "kardesleri" var (orn. "parkyeri_1".."parkyeri_8" - arac
# basina bir sutun) - bir kavram henuz melt edilmemis boyle bir grubun SADECE
# tek bir indeksine eslendiginde, TUM grubu havuzlayip dogru/tam sonuc
# uretmek icin kullanilir (bkz. category_distribution icindeki commit notu -
# gercek bir dosyada "Arac Park Yeri" icin ~%11 eksik cevap sayimina yol
# actigi olculup dogrulandi). Kullanici zaten melt yaptiysa (mapping tek,
# indekssiz bir sutuna esliyorsa) bu sozlukte karsiligi olmaz, davranis
# ONCEKI ile BIREBIR AYNI kalir.
col_group_siblings: dict[str, list[str]] = {}
for _base, _idxmap in single_groups.items():
    _cols = [c for _, c in sorted(_idxmap.items())]
    if len(_cols) > 1:
        for _c in _cols:
            col_group_siblings[_c] = _cols
# double_groups da ayni sekilde (orn. "amac_1_1".."amac_6_3" - kisi x
# yolculuk bacagi basina bir sutun) - "hangi yolculuga ait" bilgisi 2.
# indekste tasindigi icin, TEK bir (kisi,yolculuk) hucresine degil TUM
# yolculuklara gore dagilim gerektiginde (orn. "Amac"/"Mod" ilce-bazinda
# kirilimi) ayni havuzlama mantigi burada da gecerlidir.
for _base, _idxmap2 in double_groups.items():
    _cols2 = [c for _, c in sorted(_idxmap2.items())]
    if len(_cols2) > 1:
        for _c in _cols2:
            col_group_siblings[_c] = _cols2
single_blocks = safe("Kisi/hane blok kumeleme", cg.cluster_single_blocks, single_groups) or []
double_blocks = safe("Yolculuk blok kumeleme", cg.cluster_double_blocks, double_groups) or []

combine_key = f"combines_{ds.db_path}"
melt_key = f"melts_{ds.db_path}"
smart_key = f"smart_{ds.db_path}"
flagcount_key = f"flagcounts_{ds.db_path}"
st.session_state.setdefault(combine_key, {})
st.session_state.setdefault(melt_key, {})
st.session_state.setdefault(smart_key, {"person_view": None, "trip_view": None})
st.session_state.setdefault(flagcount_key, {})

# kimlik sutunu adaylari: performans + anlam icin sadece "duz" (numarasiz
# tekrar grubuna dahil olmayan) sutunlar arasindan sunulur - binlerce sutunlu
# dosyalarda TUM sutun listesini (orn. 5699 secenek) coklu-secim kutusuna
# basmak ciddi sekilde yavaslatir ve zaten anlamli bir kimlik sutunu olamaz.
plain_cols = all_groups.get("plain", ds.columns) or ds.columns
id_universe = plain_cols[:150] if len(plain_cols) > 150 else plain_cols
_id_kw_candidates = [c for c in id_universe if any(
    k in c.lower() for k in ["hane", "haneno", "id", "kumeno", "anketid"])] or id_universe[:1]

# DOGRULUK ICIN KRITIK: anahtar kelimeyle eslesen ilk sutun (orn. "haneno")
# GERCEKTE tekil olmayabilir (bkz. suggest_unique_key docstring'i - gercek
# bir dosyada bu, JOIN'de 15 bin satirdan 30 milyon satirlik curuk bir
# sonuca yol acmisti). Bu yuzden varsayilan secim, ic gorunustekiyle degil
# GERCEK TEKILLIK OLCUMUYLE belirlenir.
@st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS)
def _cached_suggest_key(_con, table, candidates, total_rows):
    return cg.suggest_unique_key(_con, table, candidates, total_rows)


best_key_cols, best_key_ratio = safe(
    "Kimlik sutunu tekillik kontrolu", _cached_suggest_key,
    con, BASE_TABLE, tuple(_id_kw_candidates), ds.total_rows,
) or (_id_kw_candidates[:1], 0.0)
id_candidates = list(best_key_cols) + [c for c in _id_kw_candidates if c not in best_key_cols]
if best_key_ratio < 0.999:
    st.sidebar.warning(
        f"⚠️ Onerilen kimlik sutunu(lari) `{', '.join(best_key_cols)}` bile tam tekil degil "
        f"(tekillik orani: %{best_key_ratio*100:.1f}). Kisi/Yolculuk tablosu olustururken "
        f"'Hane/anket kimlik sutunu' alanindan farkli bir sutun/kombinasyon secmeyi deneyin - "
        f"aksi halde birlestirme (JOIN) hatali/sisirilmis sonuc verebilir."
    )

# ----------------------------------------------------------- akilli oneri (hane->kisi->yolculuk)
# DOGRULUK ICIN KRITIK: bazi anket dosyalarinda (orn. Konya saha verisi) TEK
# BIR degil, BIRDEN FAZLA kisi-bazli (tek-indeksli) blok bulunabilir - orn.
# biri TUM hanehalkinin demografik bilgilerini (cins, yas, egit... - genis
# index araligi, orn. 1-15) tutarken, digeri SADECE is/okul yolculugu yapan
# kisilerin detaylarini (cikmama, isarac... - dar index araligi, orn. 1-10)
# tutar. Eskiden burada SADECE yolculuk blogunun kisi-index kumesiyle TAM
# ESIT olan blok seciliyordu - bu, index araligi trip blogundan GENIS olan
# (ama yine de TUM trip-yapan kisileri kapsayan) demografik blogu YANLISLIKLA
# disarida birakiyordu. Olculdu: gercek bir Konya dosyasinda bu yuzden
# "Ortalama Yas" alakasiz bir sutuna (hanedeki 5-yas-alti sayisi) yanlislikla
# eslesip 3.9 gibi imkansiz bir deger uretiyordu (dogrusu ~30-40 civari).
# Fix: trip blogunun TUM kisi index'lerini KAPSAYAN (superset/esit) adaylar
# arasindan, bilinen kavramlarla (mapping.py) EN COK eslesen - yani "kisi"
# analizinde fiilen en cok ise yarayacak - blok tercih edilir.
def _pick_smart_person_block(blocks, trip_outer_indices=None):
    candidates = blocks
    if trip_outer_indices is not None:
        need = set(trip_outer_indices)
        covering = [b for b in blocks if need.issubset(set(b["indices"]))]
        if covering:
            candidates = covering
    if not candidates:
        return None

    def _score(b):
        guess = mapping_mod.auto_guess_mapping(b["bases"])
        n_matched = sum(1 for v in guess.values() if v)
        return (n_matched, len(b["bases"]))

    return max(candidates, key=_score)


smart_person_block = None
smart_trip_block = None
if double_blocks:
    smart_trip_block = double_blocks[0]
    smart_person_block = _pick_smart_person_block(single_blocks, smart_trip_block["outer_indices"])
elif single_blocks:
    # cift-indeksli yolculuk yoksa, en cok bilinen kavramla eslesen tek-indeksli
    # blok genelde "kisi" blogudur - yine de oneri olarak sunulur.
    smart_person_block = _pick_smart_person_block(single_blocks)

if smart_person_block or smart_trip_block:
    with st.sidebar.expander("✨ Onerilen Donusum (hane → kisi → yolculuk)", expanded=True):
        st.caption(
            "Dosyanizda hanedeki kisilerin (ve varsa kisi basina yolculuklarin) "
            "ayri sutunlar halinde (kisi1, kisi2, ... / yolculuk1, yolculuk2, ...) "
            "tutuldugu tespit edildi. Asagidaki secimle bunlari tek tikla, dogru "
            "analiz icin gereken 'uzun format' tablolara cevirebilirsiniz."
        )
        smart_id_cols = st.multiselect(
            "Hane/anket kimlik sutunu (satirlari birbirine baglayan sutun)",
            id_universe, default=id_candidates[:1], key=f"smart_id_{ds.db_path}",
        )
        c1, c2 = st.columns(2)
        want_person = False
        want_trip = False
        with c1:
            if smart_person_block:
                want_person = st.checkbox(
                    f"🧑 Kisi tablosu ({len(smart_person_block['bases'])} alan: "
                    f"{', '.join(smart_person_block['bases'][:5])}{'...' if len(smart_person_block['bases']) > 5 else ''})",
                    value=True, key=f"smart_person_{ds.db_path}",
                )
        with c2:
            if smart_trip_block:
                want_trip = st.checkbox(
                    f"🚗 Yolculuk tablosu ({len(smart_trip_block['bases'])} alan: "
                    f"{', '.join(smart_trip_block['bases'][:5])}{'...' if len(smart_trip_block['bases']) > 5 else ''})",
                    value=True, key=f"smart_trip_{ds.db_path}",
                )
        if st.button("🪄 Tabloyu/tablolari olustur", key=f"smart_apply_{ds.db_path}"):
            new_state = {"person_view": None, "trip_view": None}
            with st.spinner("Tablolar olusturuluyor (genis dosyalarda bir dakikaya kadar surebilir)..."):
                if want_person and smart_person_block and smart_id_cols:
                    # NOT (performans): eskiden burada once dar bir ara tablo
                    # olusturuluyordu (build_narrow_staging_sql). Olculdu:
                    # 5000+ sutunlu kaynaklarda DuckDB'nin CREATE TABLE AS
                    # SELECT ile ayri bir ara tabloyu MATERYALIZE etmesi,
                    # dogrudan UNION ALL melt sorgusundan DAHA YAVAS (ayni
                    # veri icin ~3,4 kat) - cunku materyalizasyon DuckDB'nin
                    # her UNION dalinda sadece ihtiyac duyulan sutunlari okuma
                    # (projection pushdown) optimizasyonunu engelliyor. Ara
                    # tablo kaldirilinca sonuc satir satir/sutun sutun
                    # BIREBIR AYNI kaliyor (EXCEPT sorgusuyla dogrulandi),
                    # sadece cok daha hizli. Bu yuzden artik dogrudan
                    # kaynaktan (MELT_SOURCE) melt yapiliyor.
                    # extra_cols=plain_cols: ilce/gelir/hanehalki buyuklugu gibi
                    # "duz" (tekrarlamayan) hane sutunlari da tasinsin diye -
                    # bkz. build_melt_view_sql docstring'indeki "DOGRULUK ICIN
                    # KRITIK" notu (bu olmadan bu sutunlar kisi_uzun'da hic
                    # bulunmuyordu, "Gelir Durumu" gibi bolumler sessizce
                    # "eslestirilemedi" gosteriyordu).
                    sql, _ = cg.build_melt_view_sql(MELT_SOURCE, smart_person_block, smart_id_cols,
                                                     "kisi_uzun", index_col_name="kisi_no",
                                                     extra_cols=plain_cols)
                    if safe("Kisi tablosu olusturma", con.execute, sql) is not None:
                        # DOGRULUK ICIN KRITIK: "kisi_no" HER HANEDE 1'den
                        # baslayarak tekrarlanan YEREL bir sira numarasidir
                        # (1. hanenin 3. kisisi de, 500. hanenin 3. kisisi
                        # de kisi_no=3'tur) - TEK BASINA KESINLIKLE tekil
                        # bir kisi kimligi DEGILDIR. Cinsiyet/yas/egitim
                        # gibi kisi-bazli dagilimlar "person_id" ile
                        # tekrarsizlastirildiginda (bkz. analytics.py
                        # _dedup_source) kisi_no yanlislikla kimlik olarak
                        # kullanilirsa, TUM haneler sadece 1-11 arasi
                        # birkac gruba collapse olur (gercek bir dosyada
                        # 15.610 kisi yerine sadece ~11 "kisi" gorunur,
                        # dagilim tamamen yanlis cikar - olculup
                        # dogrulandi). Bu yuzden burada hane kimligi +
                        # kisi_no'yu birlestiren GERCEKTEN tekil bir
                        # "_kisi_id" sutunu ekleniyor.
                        kisi_id_expr = " || '_' || ".join(
                            [f"CAST({an.ident(c)} AS VARCHAR)" for c in smart_id_cols]
                            + ["CAST(kisi_no AS VARCHAR)"]
                        )
                        safe("Kisi ID ekleme", con.execute, f"""
                            CREATE OR REPLACE TABLE kisi_uzun AS
                            SELECT *, {kisi_id_expr} AS {an.ident('_kisi_id')} FROM kisi_uzun
                        """)
                        new_state["person_view"] = "kisi_uzun"
                if want_trip and smart_trip_block and smart_id_cols:
                    # Ayni performans notu (yukariya bkz.) - ara tablo yok.
                    sql2, _ = cg.build_double_melt_view_sql(MELT_SOURCE, smart_trip_block, smart_id_cols,
                                                              "yolculuk_uzun", extra_cols=plain_cols)
                    if safe("Yolculuk tablosu olusturma", con.execute, sql2) is not None:
                        # Ayni "_kisi_id" mantigi burada da gerekli: yolculuk_uzun'daki
                        # "kisi_no" da hane-yerel bir sira numarasi - kisi bazli
                        # (cinsiyet/yas/egitim) dagilimlar bu tablo uzerinden
                        # hesaplanacaksa (orn. yolculuk_zengin JOIN'i sonrasi)
                        # ayni tekil kimlik burada da bulunmali.
                        kisi_id_expr2 = " || '_' || ".join(
                            [f"CAST({an.ident(c)} AS VARCHAR)" for c in smart_id_cols]
                            + ["CAST(kisi_no AS VARCHAR)"]
                        )
                        safe("Yolculuk Kisi ID ekleme", con.execute, f"""
                            CREATE OR REPLACE TABLE yolculuk_uzun AS
                            SELECT *, {kisi_id_expr2} AS {an.ident('_kisi_id')} FROM yolculuk_uzun
                        """)
                        new_state["trip_view"] = "yolculuk_uzun"
                        if new_state["person_view"]:
                            # GUVENLIK KONTROLU: secilen kimlik + kisi_no
                            # Kisi tablosunda TEKIL degilse JOIN devasa
                            # bir fan-out'a yol acar (bkz. count_key_duplicates
                            # docstring'i - gercek bir dosyada 30 MILYON
                            # satirlik curuk sonuc uretmisti). Boyle bir
                            # durumda JOIN hic denenmez, kullaniciya
                            # acik bir uyari gosterilir.
                            dup_n = safe("Kimlik tekillik kontrolu", cg.count_key_duplicates,
                                         con, "kisi_uzun", smart_id_cols + ["kisi_no"])
                            if dup_n:
                                st.error(
                                    f"❌ Kişi+Yolculuk birleştirilemedi: seçtiğiniz kimlik sütunu "
                                    f"(`{', '.join(smart_id_cols)}`) Kişi tablosunda tekil değil "
                                    f"({dup_n:,} tekrar var). Bu, birleştirmede devasa/hatalı bir "
                                    f"sonuca yol açar. Lütfen daha yukarıdaki 'Hane/anket kimlik "
                                    f"sütunu' alanından farklı bir sütun ya da birden fazla sütun "
                                    f"kombinasyonu seçin (örn. tek başına 'haneno' yerine "
                                    f"'kumeno'+'haneno' birlikte, ya da tekil bir 'id' sütunu)."
                                )
                            else:
                                # extra_person_exclude: hem trip hem person tarafi
                                # ARTIK AYNI "duz" hane sutunlarini (plain_cols) da
                                # tasidigi icin, JOIN'de bunlarin kisi tarafindaki
                                # kopyasi disarida birakilir (yoksa "duplicate
                                # column name" hatasi verir) - trip tarafindaki
                                # (t.*) zaten ayni degeri tasir.
                                sql3 = cg.build_join_sql("yolculuk_uzun", "kisi_uzun", smart_id_cols,
                                                          "kisi_no", "yolculuk_zengin",
                                                          extra_person_exclude=["_kisi_id"] + plain_cols)
                                if safe("Kisi+Yolculuk birlestirme", con.execute, sql3) is not None:
                                    new_state["trip_view"] = "yolculuk_zengin"
            st.session_state[smart_key] = new_state
            if new_state["person_view"] or new_state["trip_view"]:
                st.success("Olusturuldu. Asagidaki 'Hangi tabloda analiz yapilsin?' menusunden secebilirsiniz.")

# ----------------------------------------------------------- manuel/gelismis
other_single_blocks = [b for b in single_blocks if b is not smart_person_block]

# KULLANICI TALEBI: "sirali degiskenleri toplama sekmesi olsun" - ONCEDEN
# "cins", "yas", "egit" gibi bir taban BIRDEN FAZLA tabanin ayni index
# kumesini paylastigi bir bloga (kisi/yolculuk blogu) dahilse TAMAMEN
# gizleniyordu (orn. "aracadet" da "aracdurum, parkyeri" ile ayni 1-8
# index'ini paylastigi icin burada GORUNMUYORDU - halbuki aracadet
# TOPLANABILIR bir miktar, sadece ayni bloktaki aracdurum gibi kategorik
# kod alanlari TOPLANAMAZ). Artik TUM tespit edilen sirali/numarali
# gruplar burada listelenir, karar KULLANICIYA birakilir - varsayilan
# "Kullanma" oldugu icin secilmedikce hicbir sey degismez. Sadece
# cift-indeksli (kisi x yolculuk) gruplar haric tutulur, cunku onlar
# zaten bu tek-indeksli listede yer almaz.
_multi_block_bases = {b for blk in single_blocks for b in blk["bases"]}
summable_bases_all = {b: idx for b, idx in single_groups.items() if b not in set(double_groups.keys())}

# DOGRULUK ICIN KRITIK (gercek bir dosyada olculup dogrulandi): isim, adres,
# saat gibi ac1kca METIN-BAZLI gruplar da yukaridaki listede goruntuleniyordu -
# byle bir grup secilip "Topla (SUM)" denildiginde, numeric_expr hicbirini
# sayiya ceviremedigi icin SESSIZCE HER SATIRDA '0' uretiyordu (crash yok,
# ama tamamen yanlis/anlamsiz bir sonuc). Simdi SADECE degerlerinin
# COGUNLUGU (>= %50) gercekten sayisal olarak ayristirilabilen gruplar
# "1️⃣ Topla" listesinde gosterilir; elenenler ayri/acik bir notla belirtilir
# (yine de "3️⃣ Kac kisi bu cevabi verdi" bolumunden KULLANILABILIRLER,
# cunku o islem metin degerler icin de anlamlidir).
_numeric_like = safe("Sayisal-gibi taban tespiti", cg.numeric_like_bases,
                     con, BASE_TABLE, summable_bases_all, an.numeric_expr) or set()
summable_bases = {b: idx for b, idx in summable_bases_all.items() if b in _numeric_like}
summable_bases_excluded = sorted(set(summable_bases_all) - _numeric_like)

# "cins", "ogrenci", "calisma_durumu" gibi kisi/kayit bazli KATEGORIK
# tabanlar (yukaridaki gibi TOPLA/ORTALAMA'dan haric tutulanlar) icin: ham
# degeri toplamak yerine, secilen "pozitif" deger(ler)e sahip KAC KISI/KAYIT
# oldugunu satir (hane/anket) basina SAYAN ayri bir secenek sunulur (bkz.
# build_flag_count_expr). single_groups'ta olup double_groups'a (kisi x
# yolculuk) ait OLMAYAN tabanlarla sinirlanir - cift-indeksli yolculuk
# tabanlari bu satir-bazli sayima uygun degildir.
countable_multi_bases = {b: single_groups[b] for b in _multi_block_bases if b in single_groups}

if summable_bases or other_single_blocks or countable_multi_bases:
    with st.sidebar.expander("🧩 Tekrarlayan sutunlari duzenle (istege bagli)", expanded=False):
        st.caption(
            "Bu bolum SADECE 'aracadet1, aracadet2...' ya da 'kisi1, kisi2...' gibi "
            "YAN YANA TEKRAR EDEN sutunlarinizi ne yapmak istediginizi soruyor. "
            "Boyle sutununuz yoksa ya da emin degilseniz, hicbir sey secmeden "
            "gecebilirsiniz - varsayilan analizler zaten calisir."
        )
        filt = st.text_input("🔎 Bir sutun adi yazip arayin (orn. arac, gelir, kisi...)",
                             key=f"filter_{ds.db_path}")

        # NOT (basitlestirme): SEC KUTUSUNUN GERCEK/SAKLANAN DEGERLERI (ASAGIDAKI
        # options listesi) DEGISTIRILMEDI - sadece format_func ile EKRANDA
        # gosterilen metin sadelestirildi. Boylece kullaniciyi zaten acik olan
        # bir oturumda (session_state'te eski secim saklanmisken) hata
        # olusturma riski YOK, ve asagidaki hesaplama kodu (agg_map.get(choice))
        # da HICBIR SEKILDE degismedi - uretilen sayilar birebir aynidir.
        COMBINE_LABELS = {
            "Ayri birak": "Kullanma (bu grubu analize katma)",
            "Topla (SUM)": "➕ Toplamlarini al  —  orn. toplam arac sayisi",
            "Ortalama": "➗ Ortalamasini al",
            "Maksimum": "🔝 En buyugunu al",
            "Sifir-olmayan sayisi": "🔢 Kac tanesi dolu/bos-degil, onu say",
        }
        st.markdown("**1️⃣ Sirali/numarali sutunlari topla (ya da ortalamasini/en buyugunu al)**")
        if summable_bases:
            _ex_base = sorted(summable_bases)[0]
            _ex_idx = summable_bases[_ex_base]
            _ex_cols = ", ".join(_ex_idx[i] for i in sorted(_ex_idx)[:3])
            st.caption(f"Orn: sizin dosyanizda `{_ex_cols}...` gibi sutunlar bu gruba giriyor.")
        if summable_bases_excluded:
            st.caption(
                f"ℹ️ {len(summable_bases_excluded)} grup ({', '.join(f'`{b}`' for b in summable_bases_excluded[:6])}"
                f"{'...' if len(summable_bases_excluded) > 6 else ''}) burada GORUNMUYOR - degerleri "
                "cogunlukla metin/isim/adres/saat gibi gorundugu icin toplamak/ortalamasini almak "
                "anlamsiz sonuc verir. Bu gruplari asagidaki '3️⃣ Kac kisi bu cevabi verdi' bolumunden "
                "kullanabilirsiniz."
            )
        else:
            st.caption("Dosyanizda sirali/numarali (orn. `alan1, alan2, alan3...`) bir sutun grubu tespit edilmedi.")
        shown = 0
        for base, idxmap in sorted(summable_bases.items()):
            if filt and filt.lower() not in base.lower():
                continue
            shown += 1
            cols_str = ", ".join(idxmap[i] for i in sorted(idxmap)[:5])
            # DOGRULUK/KARARLILIK ICIN: asagidaki uyari captiontu SECIME BAGLI
            # olarak (choice degerine gore) GORUNUP KAYBOLUYOR - bu, dongudeki
            # DIGER selectbox'larla AYNI seviyede, kendi container'i olmadan
            # yapilirsa, React'in DOM agaci karsilastirmasi (reconciliation)
            # sasirip "NotFoundError: removeChild" hatasi verir (bkz. asagidaki
            # 2./3. bolumlerdeki ayni notlar - daha once tam bu sekilde
            # olculup dogrulanmis bir hata). Her grubu kendi container'ina
            # almak bu riski ortadan kaldirir.
            with st.container():
                choice = st.selectbox(
                    f"`{base}` grubu ({len(idxmap)} sutun: {cols_str}{'...' if len(idxmap) > 5 else ''})",
                    list(COMBINE_LABELS.keys()), format_func=lambda k: COMBINE_LABELS.get(k, k),
                    key=f"combine_{ds.db_path}_{base}",
                )
                # KULLANICI UYARISI (sadece bilgi amacli - secimi ENGELLEMEZ):
                # bazi gruplar "yas", "cins", "egit" gibi KOD/KATEGORI degerleri
                # tasir - bunlarin toplami/ortalamasi sayisal olarak hesaplanir
                # ama ANLAMSIZ bir sonuc verir (orn. 11 kisinin cinsiyet kodu
                # toplami). Miktar ifade eden alanlarda (adet, tutar, sure
                # gibi) bu sorun yoktur.
                if base in _multi_block_bases and choice != "Ayri birak":
                    st.caption(
                        "⚠️ Bu grup kod/kategori degeri tasiyorsa (orn. cinsiyet, egitim "
                        "kodu) toplam/ortalama anlamsiz cikar - sadece gercek MIKTAR "
                        "ifade eden alanlarda (adet, tutar gibi) kullanin."
                    )
                agg_map = {"Topla (SUM)": "sum", "Ortalama": "avg", "Maksimum": "max",
                           "Sifir-olmayan sayisi": "count_nonzero"}
                st.session_state[combine_key][base] = agg_map.get(choice)
        if summable_bases and shown == 0:
            st.caption("Eslesen grup yok." if filt else "Bu turde grup bulunamadi.")

        if other_single_blocks:
            st.markdown("**2️⃣ Bu gruplari kisi/yolculuk bazinda tek tek incele**")
            st.caption(
                "Orn: `kisi1, kisi2, kisi3...` gibi sutunlariniz varsa, her kisiyi "
                "kendi satirina ayirir - yas/cinsiyet gibi kisi bazli analizler icin gerekli."
            )
            for i, b in enumerate(other_single_blocks):
                label = cg.humanize_single_label(b)
                if filt and filt.lower() not in label.lower():
                    continue
                # DOGRULUK/KARARLILIK ICIN: checkbox isaretlendiginde ASAGIYA
                # EK widget'lar (multiselect) EKLENIYOR - bu, o widget grubunu
                # KENDI st.container()'ina almadan yapilirsa, React'in DOM
                # agaci karsilastirmasi (reconciliation) genis/karmasik
                # dosyalarda (cok sayida boyle grup ust uste oldugunda)
                # sasirip "NotFoundError: removeChild" hatasi verebiliyor
                # (gercek bir dosyada olculup dogrulandi). Her grubu kendi
                # container'ina almak, React'in degisikligi SADECE o
                # container'a hapsetmesini saglayarak bu hatayi onler.
                with st.container():
                    enabled = st.checkbox(f"`{label}` grubunu kisi bazinda ayir", key=f"melt_on_{ds.db_path}_{i}")
                    if enabled:
                        default_id = id_candidates[:1]
                        with st.expander("🔧 Baglanti sutununu degistir (cogu zaman gerek yok)"):
                            id_cols = st.multiselect(
                                "Hangi sutun ayni haneyi/anketi birbirine bagliyor?", id_universe,
                                default=default_id, key=f"melt_id_{ds.db_path}_{i}",
                            )
                        if not id_cols:
                            id_cols = default_id
                        else:
                            st.caption("Kullanilacak baglanti sutunu: " + ", ".join(f"`{c}`" for c in id_cols))
                        view_name = f"uzun_{'_'.join(b['bases'][:2])}"
                        st.session_state[melt_key][label] = {"block": b, "id_cols": id_cols, "view_name": view_name}
                    elif label in st.session_state[melt_key]:
                        del st.session_state[melt_key][label]

        if countable_multi_bases:
            st.markdown("**3️⃣ Kac kisi belirli bir cevabi verdi, onu say**")
            st.caption(
                "Orn: `cins`, `ogrenci`, `calisma_durumu` gibi her kisi icin ayri ayri "
                "doldurulmus sutunlariniz varsa: 'hanede kac kisi ÖĞRENCİ' gibi TEK bir "
                "sayi sutunu olusturur. Bos/doldurulmamis kisi yuvalari sayilmaz."
            )
            shown2 = 0
            for base in sorted(countable_multi_bases):
                if filt and filt.lower() not in base.lower():
                    continue
                shown2 += 1
                idxmap = countable_multi_bases[base]
                cols_str2 = ", ".join(idxmap[i] for i in sorted(idxmap)[:5])
                # bkz. yukaridaki "Diger tekrarlayan kayit gruplari" ayni notu:
                # her grubu kendi st.container()'ina almak, checkbox
                # isaretlenince ASAGIYA EK widget'lar (onizleme + multiselect)
                # eklenmesinin React'in DOM karsilastirmasini sasirtip
                # "NotFoundError: removeChild" hatasi vermesini onler.
                with st.container():
                    enabled2 = st.checkbox(
                        f"`{base}` grubu icin sayim olustur ({len(idxmap)} sutun: "
                        f"{cols_str2}{'...' if len(idxmap) > 5 else ''})",
                        key=f"flagcount_on_{ds.db_path}_{base}",
                    )
                    if enabled2:
                        cols_for_base = [idxmap[i] for i in sorted(idxmap)]
                        # NOT: bu blok ACTIVE_TABLE tanimlanmadan ONCE calisir (bkz.
                        # asagidaki "aktif tablo secimi"), bu yuzden onizleme
                        # BASE_TABLE'a karsi yapilir - tipki bu ayni bolumdeki kimlik
                        # sutunu tekillik kontrolu (suggest_unique_key) gibi.
                        union_sql = " UNION ALL ".join(
                            f"SELECT {an.cat_expr(c)} AS v FROM {an.ident(BASE_TABLE)}" for c in cols_for_base
                        )
                        preview2 = safe(f"{base} onizleme", con.execute, f"""
                            SELECT v, COUNT(*) AS n FROM ({union_sql}) WHERE v IS NOT NULL
                            GROUP BY 1 ORDER BY n DESC LIMIT 20
                        """)
                        opts2 = preview2.fetchdf()["v"].astype(str).tolist() if preview2 is not None else []
                        if opts2:
                            st.caption("Bu sutunlarda gorulen degerler: " + ", ".join(f"`{v}`" for v in opts2))
                        existing2 = st.session_state[flagcount_key].get(base, [])
                        chosen_vals = st.multiselect(
                            "Hangi cevap(lar) SAYILSIN?", opts2,
                            default=[v for v in existing2 if v in opts2],
                            key=f"flagcount_vals_{ds.db_path}_{base}",
                        )
                        if chosen_vals:
                            st.session_state[flagcount_key][base] = chosen_vals
                        elif base in st.session_state[flagcount_key]:
                            del st.session_state[flagcount_key][base]
                    elif base in st.session_state[flagcount_key]:
                        del st.session_state[flagcount_key][base]
            if shown2 == 0:
                st.caption("Eslesen alan yok." if filt else "Bu turde alan bulunamadi.")

if not single_groups and not double_groups:
    st.sidebar.caption("Numaralandirilmis sutun grubu tespit edilmedi.")

# combine secimlerini TEK bir raw_data goruntusune uygula (var olan sutunlara ek
# olarak). Secim degistikce (ya da tumu "Ayri birak" olduguna) view HER
# RERUN'DA yeniden tanimlanir - boylece eski/artik secimlerden kalma sutunlar
# view'da "takili" kalmaz. SADECE hala gecerli (bu dosyada tespit edilen)
# tabanlar kabul edilir - onceki bir oturumdan/dosyadan "takili" kalmis olsa
# bile burada elenir.
combine_choices = {b: agg for b, agg in st.session_state[combine_key].items()
                   if agg and b in summable_bases}
extra_exprs = []
for base, agg in combine_choices.items():
    idxmap = single_groups.get(base)
    if not idxmap:
        continue
    try:
        expr = cg.build_combine_expr(idxmap, agg, an.numeric_expr)
        new_col = f"{base}_{agg}"
        extra_exprs.append(f"{expr} AS {an.ident(new_col)}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"⚠️ `{base}` birlestirilemedi: {e}")

# kisi/kayit bazli sayim secimleri (bkz. yukaridaki "Kisi/kayit bazli SAYIM"
# paneli) - AYNI extra_exprs/extra_sql akisina eklenir, tek bir view'da
# combine sutunlariyla birlikte olusur. SADECE hala gecerli (coklu-tabanli
# bloktan cikmamis) tabanlar kabul edilir - onceki bir oturumdan "takili"
# kalmis olsa bile burada elenir.
flagcount_choices = {b: vals for b, vals in st.session_state[flagcount_key].items()
                      if vals and b in countable_multi_bases}
for base, positive_vals in flagcount_choices.items():
    idxmap = countable_multi_bases.get(base)
    if not idxmap:
        continue
    try:
        expr = cg.build_flag_count_expr(idxmap, positive_vals)
        new_col = f"{base}_sayi"
        extra_exprs.append(f"{expr} AS {an.ident(new_col)}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"⚠️ `{base}` sayim sutunu olusturulamadi: {e}")

extra_sql = (", " + ", ".join(extra_exprs)) if extra_exprs else ""
safe("Sutun birlestirme", con.execute,
     f"CREATE OR REPLACE VIEW {BASE_TABLE} AS SELECT t.*{extra_sql} FROM src.{BASE_TABLE} AS t")

# manuel melt goruntulerini olustur (dogrudan kaynaktan - bkz. akilli
# donusum butonundaki ayni performans notu: ara tablo kaldirildi, ~3,4 kat
# daha hizli ve sonuc birebir ayni).
# PERFORMANS: bu dongu eskiden HER rerun'da (sayfadaki HERHANGI bir widget'a
# tiklandiginda, ilgisiz bir sekmede olsa bile) CREATE TABLE'i YENIDEN
# calistiriyordu - oysa `con` bir @st.cache_resource baglantisi oldugu icin
# tablo zaten oradan bir onceki calistirmadan KALICI olarak duruyor. Ayni SQL
# metni (= ayni kaynak + ayni ayarlar) tekrar uretilirse, tabloyu yeniden
# olusturmak SONUCU DEGISTIRMEZ, sadece zaman kaybettirir. Bu yuzden her
# view_name icin en son basariyla calistirilan SQL metni session_state'te
# tutulur; SQL metni degismediyse (ayarlar aynıysa) yeniden calistirilmaz.
melt_views: dict[str, list[str]] = {}
_melt_sql_key = f"_melt_last_sql_{ds.db_path}"
_last_melt_sql = st.session_state.setdefault(_melt_sql_key, {})
for label, cfg in st.session_state[melt_key].items():
    sql, new_cols = cg.build_melt_view_sql(MELT_SOURCE, cfg["block"], cfg["id_cols"], cfg["view_name"],
                                            extra_cols=plain_cols)
    view_name = cfg["view_name"]
    if _last_melt_sql.get(view_name) == sql:
        melt_views[view_name] = new_cols
        continue
    if safe(f"Uzun format: {label}", con.execute, sql) is not None:
        _last_melt_sql[view_name] = sql
        melt_views[view_name] = new_cols
    else:
        _last_melt_sql.pop(view_name, None)

# ----------------------------------------------------------------- aktif tablo secimi
table_options = {"📋 Ana Tablo": BASE_TABLE}
smart_state = st.session_state[smart_key]
if smart_state.get("person_view"):
    table_options["🧑 Kisi Tablosu (uzun format)"] = smart_state["person_view"]
if smart_state.get("trip_view"):
    table_options["🚗 Yolculuk Tablosu (uzun format)"] = smart_state["trip_view"]
for view_name in melt_views:
    table_options[f"📐 Uzun Format: {view_name}"] = view_name

if len(table_options) > 1:
    default_idx = len(table_options) - 1  # en zengin/son olusturulan tablo varsayilan
    # ONEMLI DUZELTME (Streamlit'in kendi AppTest cercevesiyle dogrulandi - bkz.
    # commit notu): bu selectbox'in eskiden key= PARAMETRESI YOKTU. Streamlit,
    # key verilmeyen bir widget'in "kimligini" o an gecilen TUM parametrelerden
    # (options listesi dahil) turetir - options listesinin ICERIGI her
    # degistiginde (ör. sidebar'da HERHANGI bir "... kisi bazinda ayir" kutusu
    # isaretlenip yeni bir "Uzun Format" gorunumu eklendiginde, ya da akilli
    # kisi/yolculuk tablosu tespiti degistiginde) Streamlit bunu SIFIRDAN bir
    # widget sayip kullanicinin ONCEDEN sectigi degeri SESSIZCE UNUTUYOR ve
    # index=default_idx'e (= listedeki EN SON tablo) geri donuyordu. Sonuc:
    # kullanici "Kisi Tablosu"nu secmisken sidebar'da ilgisiz bir kutuyu
    # isaretlemesi bile TUM sekmelerdeki (yas, cinsiyet, egitim...) degerlerin
    # BASKA - ve o analiz icin YANLIS - bir tablodan hesaplanmasina yol
    # aciyordu; kullaniciya HICBIR uyari gosterilmeden. Sabit bir key= ile
    # kullanicinin secimi kalici hale getirilir; secim artik gecersizse (o
    # blok/gorunum kaldirildiysa) session_state'teki deger widget
    # olusturulmadan ONCE guvenli varsayilana cekilir (Streamlit, key ile
    # birlikte options'ta olmayan bir deger varsa hata firlatir).
    _table_sel_key = f"active_table_select_{ds.db_path}"
    _table_labels = list(table_options.keys())
    if st.session_state.get(_table_sel_key) not in table_options:
        st.session_state[_table_sel_key] = _table_labels[default_idx]
    active_label = st.sidebar.selectbox(
        "Hangi tabloda analiz yapilsin?", _table_labels, key=_table_sel_key,
        help="Demografik analizler icin 'Kisi Tablosu', yolculuk analizleri icin "
             "'Yolculuk Tablosu' (varsa) ya da 'Ana Tablo' kullanin.",
    )
else:
    active_label = "📋 Ana Tablo"
ACTIVE_TABLE = table_options[active_label]
st.sidebar.caption(f"Aktif tablo: `{ACTIVE_TABLE}`")

active_columns = table_columns(con, ACTIVE_TABLE)
active_dtypes = table_dtypes(con, ACTIVE_TABLE)
active_rows = safe("Satir sayisi", con.execute, f"SELECT COUNT(*) FROM {an.ident(ACTIVE_TABLE)}")
active_rows = active_rows.fetchone()[0] if active_rows is not None else 0

# DOGRULUK ICIN KRITIK (gercek bir dosyada olculup dogrulandi - "Ulasim
# Turu" icin): yukaridaki col_group_siblings SADECE HAM dosyanin
# sutunlarindan (ds.columns) turetiliyordu. Ama bazi "bacak/transfer"
# gruplari (orn. "yol_arac1_1_1".."yol_arac1_4_10" GIBI HAM sutunlarda HER
# BACAK - yol_arac1, yol_arac2, yol_arac3 - AYRI BIR TABAN/GRUP olarak
# gorunur), akilli "Yolculuk Tablosu" melt'i SONRASINDA duz "yol_arac1",
# "yol_arac2", "yol_arac3" gibi YENI, kisi/yolculuk boyutu zaten
# eritilmis SIRALI sutunlar olarak ortaya cikar - bunlar SADECE aktif
# (melt edilmis) tabloda var, ham dosyada YOK, bu yuzden yukaridaki
# ham-dosya taramasi bunlari hic gormuyordu. Sonuc: "Ulasim Turu" kavrami
# byle bir bacak sutununa (orn. sadece "yol_arac1") eslenince, 2. ve 3.
# araci/bacagi (transfer) kullanan yolculuklarin o bacaklari SESSIZCE
# sayilmiyordu - kullanicinin kendi gozlemiyle yakalandi. Simdi AKTIF
# tablonun sutunlari da ayrica taranip ayni kardes-sutun haritasina
# eklenir (ham dosya taramasinda zaten bulunan bir sutunun degeri
# EZILMEZ - sadece eksik olanlar tamamlanir).
#
# PERFORMANS ICIN KRITIK: aktif tablo HALA ham BASE_TABLE ise (henuz melt
# yapilmamis/Ana Tablo secili), active_columns ile ds.columns ZATEN
# BIREBIR AYNIDIR - detect_all_groups'u (binlerce sutunlu dosyalarda
# olculup dogrulandi: tek basina saniyeler surebiliyor) IKINCI KEZ
# calistirmak SONUCU DEGISTIRMEZ, sadece zaman kaybettirir. Bu yuzden
# SADECE aktif tablo ham tablodan FARKLIYSA (bir melt/turetilmis gorunum
# aktifse) bu ek tarama yapilir.
_active_groups = ({"single": {}, "double": {}} if ACTIVE_TABLE == BASE_TABLE else
                   (safe("Aktif tablo sutun grubu tespiti", cg.detect_all_groups, active_columns)
                    or {"single": {}, "double": {}}))
for _base, _idxmap in _active_groups.get("single", {}).items():
    _cols = [c for _, c in sorted(_idxmap.items())]
    if len(_cols) > 1:
        for _c in _cols:
            col_group_siblings.setdefault(_c, _cols)
for _base, _idxmap2 in _active_groups.get("double", {}).items():
    _cols2 = [c for _, c in sorted(_idxmap2.items())]
    if len(_cols2) > 1:
        for _c in _cols2:
            col_group_siblings.setdefault(_c, _cols2)

# Sutun eslestirme artik sidebar'da TEK BUYUK panel olarak degil, HER
# SEKMENIN kendi icinde, o sekmeyle ilgili kavramlarla sinirli, kompakt bir
# panel olarak gosterilir (bkz. render_mapping_grid). Burada sadece ortak
# durum (mapping sozlugu, aday sutun siralamalari) hazirlanir.
map_key = (
    f"mapping_{ds.db_path}_{ACTIVE_TABLE}_{tuple(sorted(combine_choices.items()))}"
    f"_{tuple(sorted((b, tuple(v)) for b, v in flagcount_choices.items()))}"
)
if map_key not in st.session_state:
    guessed = auto_guess_mapping(active_columns)
    # "_kisi_id" (bkz. yukarida "Kisi ID ekleme"), anahtar-kelime skorlamasi
    # ile hicbir zaman guclu bir sekilde on plana cikmaz (kisa/turetilmis bir
    # ad oldugu icin) - ama GERCEKTEN tekil bir kisi kimligi oldugundan HER
    # ZAMAN "kisi_no" gibi hane-yerel/tekrarlayan adaylara TERCIH EDILMELIDIR.
    # Bu yuzden mevcutsa acikca zorlanir; aksi halde cinsiyet/yas/egitim gibi
    # kisi-bazli dagilimlar sessizce yanlis hesaplanir (bkz. commit notu).
    if "_kisi_id" in active_columns:
        guessed["person_id"] = "_kisi_id"
    # DOGRULUK ICIN KRITIK (gercek bir dosyada olculup dogrulandi):
    # "household_id" kavrami SADECE anahtar-kelimeyle (orn. "haneno")
    # tahmin edilir - hicbir TEKILLIK kontrolu yapilmaz. Gercek anket
    # dosyalarinda "haneno" gibi bir sutun HANE-YERELI/tekrarlayan bir
    # sira numarasi olabilir (orn. bu dosyada SADECE 41 tekil deger,
    # 15.610 satirda) - "kisi_no"nun kisiler icin oldugu HATANIN
    # AYNISI, hane duzeyinde. Bu durumda income_summary/gini_and_lorenz
    # gibi fonksiyonlar bu sutunla "tekilllestirme" yaparken aslinda
    # 15.610 haneyi yanlislikla 41 gruba coker, TAMAMEN YANLIS bir
    # ortalama/Gini uretir (bu dosyada olculdu: 39.750 TL yanlis vs
    # 45.383,86 TL dogru - "id" ile tekillik oranindaki fark net).
    # Yukarida ZATEN gercek tekillik olculerek dogrulanmis "best_key_cols"
    # (tek sutunlu ve %99,9+ tekil ise) varsa, o tercih edilir.
    if (len(best_key_cols) == 1 and best_key_ratio >= 0.999 and best_key_cols[0] in active_columns):
        guessed["household_id"] = best_key_cols[0]
    st.session_state[map_key] = guessed
mapping = dict(st.session_state[map_key])  # bu script calismasi boyunca YERINDE guncellenecek

# =========================================================== 🤖 AI destekli eslestirme
# BASKA BIR SEHRIN, FARKLI isimlendirilmis bir CSV'sinde anahtar-kelime
# tabanli otomatik tahmin (auto_guess_mapping) bos kalabilir (orn. sutun adi
# "income_lvl" gibi beklenmedik bir kisaltmaysa). Bu, SADECE kullanici
# butona basinca calisan, opsiyonel bir LLM yardimcisidir. Ham veri
# SATIRLARI DEGIL, sadece sutun adlari + birkac KUCUK ornek deger gonderilir
# (bkz. ai_llm.suggest_column_mapping docstring'i). Var olan/kullanicinin
# zaten gordugu eslestirmeleri ASLA sessizce DEGISTIRMEZ - SADECE hala BOS
# olan kavramlari doldurur; sonuc, asagidaki sekmelerdeki NORMAL dropdown'da
# gorunur ve istendigi gibi degistirilebilir - mimari HICBIR SEKILDE
# degismez, bu SADECE mevcut auto_guess_mapping'e ek bir oneri kaynagidir.
with st.sidebar.expander("🤖 Yapay zeka ile eslestirmeyi tamamla", expanded=False):
    st.caption(
        "Sutun adlarınız çok farklı/beklenmedik bir isimlendirme kullanıyorsa "
        "(örn. başka bir şehrin dosyası), hâlâ boş kalan kavramlar için yapay "
        "zekadan öneri isteyebilirsiniz. SADECE sütun adları ve birkaç örnek "
        "değer gönderilir - ham verideki hiçbir satır dışarı çıkmaz. Var olan "
        "eşleştirmeleriniz DEĞİŞTİRİLMEZ, sadece boş olanlar doldurulur."
    )
    ai_provider = ai_llm.available_provider()
    if not ai_provider:
        st.info(
            "Bu özellik için ANTHROPIC_API_KEY ya da OPENAI_API_KEY ortam "
            "değişkeni tanımlı olmalı (bkz. 'Yapay Zeka İçgörüleri' sekmesi)."
        )
    else:
        eksik_kavramlar = [c for c in CONCEPTS if not mapping.get(c)]
        if not eksik_kavramlar:
            st.caption("✅ Tüm kavramlar zaten eşleşmiş - AI önerisine gerek yok.")
        else:
            st.caption(f"Şu an {len(eksik_kavramlar)} kavram boş: "
                      + ", ".join(CONCEPTS[c][0] for c in eksik_kavramlar))
            if st.button("🤖 Boş kalan kavramlar için AI önerisi al",
                         key=f"ai_map_btn_{ds.db_path}_{ACTIVE_TABLE}"):
                with st.spinner("Yapay zeka sütunları inceliyor..."):
                    # Aday havuzu: HER tekrarlayan grup icin bir kez (orn.
                    # "cins_1".."cins_11" yerine sadece "cins"), artı tum
                    # "duz" sutunlar - boylece binlerce numarali sutun
                    # yerine makul boyutta (~100-150), ANLAMLI bir liste
                    # LLM'e gonderilir. pool[gorunur_ad] = ornekleme icin
                    # kullanilacak GERCEK/somut sutun adi. ZATEN BASKA BIR
                    # kavrama atanmis sutunlar havuzdan CIKARILIR - aksi
                    # halde LLM ayni sutunu iki farkli (yanlis) kavrama
                    # birden onerebilir.
                    _already_used = set(mapping.values())
                    ai_pool: dict[str, str] = {}
                    for c in plain_cols:
                        if c not in _already_used:
                            ai_pool[c] = c
                    for base, idxmap in single_groups.items():
                        if idxmap and base not in ai_pool and base not in _already_used:
                            ai_pool[base] = idxmap[min(idxmap)]
                    for base, idxmap in double_groups.items():
                        if idxmap and base not in ai_pool and base not in _already_used:
                            ai_pool[base] = idxmap[min(idxmap, key=lambda t: (t[0], t[1]))]
                    # Aday basina birkac ornek deger - TEK sorguda (UNION ALL),
                    # 100+ ayri sorgu yerine.
                    sample_parts = [
                        f"SELECT '{disp.replace(chr(39), chr(39)*2)}' AS col, "
                        f"CAST(v AS VARCHAR) AS val FROM (SELECT DISTINCT "
                        f"{an.cat_expr(raw_col)} AS v FROM {an.ident(BASE_TABLE)} "
                        f"WHERE {an.cat_expr(raw_col)} IS NOT NULL LIMIT 4)"
                        for disp, raw_col in ai_pool.items()
                    ]
                    samples_res = safe("AI icin ornek veri", con.execute, " UNION ALL ".join(sample_parts))
                    ai_candidates: dict[str, list[str]] = {c: [] for c in ai_pool}
                    if samples_res is not None:
                        for _, row in samples_res.fetchdf().iterrows():
                            if row["col"] in ai_candidates:
                                ai_candidates[row["col"]].append(row["val"])
                    # SADECE bos kavramlar gonderilir - hem prompt kucuk kalir
                    # hem de LLM zaten dolu kavramlarla ugrasmaz.
                    eksik_concepts = {c: CONCEPTS[c] for c in eksik_kavramlar}
                    try:
                        suggestion = ai_llm.suggest_column_mapping(ai_candidates, eksik_concepts)
                    except Exception as e:  # noqa: BLE001
                        suggestion = None
                        st.error(f"AI önerisi alınamadı: {e}")
                    if suggestion:
                        applied = {}
                        for concept, disp_col in suggestion.items():
                            if concept not in eksik_kavramlar or not disp_col or disp_col not in ai_pool:
                                continue
                            # LLM, aday havuzunun GORUNUR (bazda tekillestirilmis)
                            # adini dondurur (orn. "cins"); bu, aktif tabloda
                            # (melt sonrasi) DOGRUDAN gecerli bir sutunsa oldugu
                            # gibi kullanilir, degilse (henuz melt yapilmamis
                            # HAM tabloda) somut/numarali karsiligina (orn.
                            # "cins_1") duser - aynen auto_guess_mapping'in
                            # ham tabloda yaptigi gibi.
                            resolved = disp_col if disp_col in active_columns else ai_pool[disp_col]
                            if resolved in active_columns:
                                st.session_state[map_key][concept] = resolved
                                # KRITIK DUZELTME (Streamlit AppTest ile dogrulandi - bkz.
                                # commit notu): SADECE paylasilan `mapping` sozlugune
                                # yazmak YETERLI DEGIL - bu kavramin eslestirme
                                # widget'i (render_mapping_grid icinde) daha ONCE en
                                # az bir kez cizildiyse (neredeyse HER ZAMAN dogru,
                                # cunku widget'lar bos olsa bile "(Yok)" secili
                                # olarak cizilir), o widget'in KENDI eski/degismemis
                                # session_state degeri ("(Yok)"), AYNI script
                                # calismasi icinde birazdan tekrar cizildiginde bu
                                # atamayi SESSIZCE GERI ALIYORDU - "AI onerileri
                                # uygulandi" mesaji gorunse bile kavram GERCEKTE
                                # esletirilmemis kaliyordu. Widget'in kendi
                                # session_state degerini de ACIKCA guncelleyerek
                                # (asagida render_mapping_grid'de tuketilir) bu
                                # onlenir.
                                st.session_state.setdefault("_map_pending_override", {})[concept] = resolved
                                applied[concept] = resolved
                        if applied:
                            st.success("AI önerileri uygulandı: " + ", ".join(
                                f"{CONCEPTS[c][0]} → `{v}`" for c, v in applied.items()))
                            mapping = dict(st.session_state[map_key])
                        else:
                            st.info("Yapay zeka, boş kalan kavramlar için uygun bir sütun bulamadı.")

# =========================================================== 🏷️ kod cozme
# Gercek anket verilerinde kategorik alanlar cogunlukla sayisal KOD olarak
# gelir (orn. cinsiyet: 1/2). Grafiklerde "1", "2" gibi anlamsiz kodlar
# yerine gercek etiketleri (Erkek/Kadin) gormek icin kullanici burada
# kod->etiket eslestirmesi tanimlayabilir; sonuc yeni bir "..._etiket"
# sutunu olarak eslestirme ekranlarinda secilebilir hale gelir.
decode_key = f"decodes_{ds.db_path}_{ACTIVE_TABLE}"
st.session_state.setdefault(decode_key, {})
with st.sidebar.expander("🏷️ Kod Cozme (sayisal kodu etikete cevir)", expanded=False):
    st.caption(
        "Sutununuz 1/2/3 gibi sayisal kodlar iceriyorsa (orn. cinsiyet, egitim "
        "duzeyi), gercek karsiliklarini burada tanimlayin - yeni bir 'etiketli' "
        "sutun olusur ve asagidaki sekmelerdeki eslestirmede secilebilir."
    )
    decode_col = st.selectbox("Sutun secin", ["(Sec)"] + active_columns,
                               key=f"decode_pick_{ds.db_path}_{ACTIVE_TABLE}")
    if decode_col != "(Sec)":
        preview = safe("Kod onizleme", con.execute, f"""
            SELECT {an.cat_expr(decode_col)} AS v, COUNT(*) AS n FROM {an.ident(ACTIVE_TABLE)}
            WHERE {an.cat_expr(decode_col)} IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 15
        """)
        if preview is not None:
            pdf = preview.fetchdf()
            st.caption("Gorulen degerler: " + ", ".join(f"`{v}`" for v in pdf["v"].astype(str)))
        existing = st.session_state[decode_key].get(decode_col, {})
        default_txt = "\n".join(f"{k}={v}" for k, v in existing.items())
        txt = st.text_area("Kod=Etiket (her satira bir tane)", value=default_txt,
                            key=f"decode_txt_{ds.db_path}_{ACTIVE_TABLE}_{decode_col}",
                            placeholder="1=Erkek\n2=Kadin", height=100)
        parsed = {}
        for line in (txt or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip():
                    parsed[k.strip()] = v.strip()
        if parsed:
            st.session_state[decode_key][decode_col] = parsed
        elif decode_col in st.session_state[decode_key]:
            del st.session_state[decode_key][decode_col]
    if st.session_state[decode_key]:
        st.caption("✅ Tanimli kod cozmeler: " + ", ".join(st.session_state[decode_key].keys()))

    # --- AI destekli kod->etiket onerisi ---------------------------------
    # Kategorik olarak eslenmis ama HENUZ kod cozmesi tanimlanmamis
    # kavramlar icin (orn. cinsiyet, surucu/yolcu durumu, egitim duzeyi),
    # yapay zekadan bu tur anketlerdeki YERLESIK kaliplara (1=Erkek,
    # 2=Kadin gibi) dayanan bir etiket tahmini istenir. SADECE butona
    # basinca calisir; sonuc AYNI decode_key yapisina yazilir - yukaridaki
    # 'Sutun secin' ile ACILAN AYNI metin kutusunda gorunur ve normal
    # sekilde degistirilebilir. Hicbir sey gizlice/geri donusumsuz
    # uygulanmaz - istemezseniz kutuyu bosaltip silebilirsiniz.
    if ai_llm.available_provider():
        kategorik_bekleyen = {
            c: mapping[c] for c in CONCEPTS
            if mapping.get(c) and CONCEPTS[c][3] == "kategorik"
            and mapping[c] in active_columns
            and mapping[c] not in st.session_state[decode_key]
        }
        if kategorik_bekleyen:
            st.caption(f"🤖 {len(kategorik_bekleyen)} kategorik sutun icin henuz kod cozmesi tanimlanmadi.")
            if st.button("🤖 Yapay zeka ile kod anlamlarini oner",
                         key=f"ai_decode_btn_{ds.db_path}_{ACTIVE_TABLE}"):
                with st.spinner("Yapay zeka kodlari yorumluyor..."):
                    ai_decode_fields: dict[str, dict] = {}
                    for concept, col in kategorik_bekleyen.items():
                        codes_res = safe(f"{col} kodlari", con.execute, f"""
                            SELECT DISTINCT {an.cat_expr(col)} AS v FROM {an.ident(ACTIVE_TABLE)}
                            WHERE {an.cat_expr(col)} IS NOT NULL LIMIT 15
                        """)
                        if codes_res is not None:
                            codes = codes_res.fetchdf()["v"].astype(str).tolist()
                            if codes:
                                ai_decode_fields[col] = {"concept_label": CONCEPTS[concept][0], "codes": codes}
                    try:
                        label_suggestions = ai_llm.suggest_code_labels(ai_decode_fields)
                    except Exception as e:  # noqa: BLE001
                        label_suggestions = {}
                        st.error(f"AI kod önerisi alınamadı: {e}")
                    if label_suggestions:
                        for col, labels in label_suggestions.items():
                            st.session_state[decode_key][col] = labels
                        st.success(
                            "Kod anlamları önerildi: " + ", ".join(label_suggestions.keys())
                            + " — yukarıdaki 'Sütun seçin' ile gözden geçirip değiştirebilirsiniz."
                        )
                    else:
                        st.info("Yapay zeka, uygun bir kod anlamı önerisi bulamadı.")

    # --- Kod sozlugu (codebook) dosyasindan TOPLU ice aktarma -------------
    # NEDEN: gercek kurumlarin (orn. IBB, TUIK) elinde genelde ZATEN resmi
    # bir kod->etiket dokumani (metadata) bulunur - kullanicinin bunu tek
    # tek elle girmesi yerine, DOGRUDAN bir CSV olarak yukleyip TOPLU
    # uygulayabilmesi gerekir (ActivityViz/PSRC gibi projelerdeki "kendi
    # codebook'unu yukle" ozelliginden esinlenilmistir). SADECE mevcut
    # sutunlarda ve BASKA hicbir hesaplamayi ETKILEMEDEN, yukaridaki AYNI
    # decode_key yapisina yazar - manuel/AI onerisiyle AYNI sekilde
    # calisir, istemezseniz "Sutun secin" ile acip duzenleyebilir/
    # silebilirsiniz.
    st.divider()
    st.caption(
        "📖 Ya da resmi bir kod sozlugunuz (codebook) varsa, TOPLU olarak yukleyin. "
        "CSV/Excel dosyasinda 3 sutun olmali: **sutun adi, kod, etiket** "
        "(basliklar tam bu olmak zorunda degil - 'sutun/column', 'kod/code/value', "
        "'etiket/label/aciklama' gibi benzer adlar da taninir)."
    )
    codebook_file = st.file_uploader("Kod sozlugu dosyasi (.csv, .xlsx)", type=["csv", "xlsx", "xls"],
                                      key=f"codebook_upload_{ds.db_path}_{ACTIVE_TABLE}")
    if codebook_file is not None:
        try:
            if codebook_file.name.lower().endswith(".csv"):
                cb_df = pd.read_csv(codebook_file, sep=None, engine="python", dtype=str)
            else:
                cb_df = pd.read_excel(codebook_file, dtype=str)
            # NOT: Turkce ozel karakterler (g,s,i,o,u,c) once ASCII'ye
            # katlanmali (bkz. utils/geo.py::normalize_tr - aynen o
            # fonksiyonun yaptigi gibi) - aksi halde "Degisken Adi" (ASCII
            # anahtar kelime) ile "Değişken Adı" (ğ/ş/ı iceren gercek
            # basliktan normallesen hali) HICBIR ZAMAN eslesmez (biri ASCII
            # 'g', digeri Turkce 'g' harfi) - bir test dosyasinda olculup
            # dogrulandi.
            cb_cols_norm = {c: geo.normalize_tr(c) for c in cb_df.columns}

            def _find_col(*synonyms):
                for orig, norm in cb_cols_norm.items():
                    if any(s in norm for s in synonyms):
                        return orig
                return None

            col_sutun = _find_col("sutun", "column", "field", "degisken", "variable")
            col_kod = _find_col("kod", "code", "value", "deger")
            col_etiket = _find_col("etiket", "label", "aciklama", "anlam", "description")
            if not (col_sutun and col_kod and col_etiket):
                st.error(
                    f"Dosyadaki sutunlar taninamadi (bulunanlar: {list(cb_df.columns)}). "
                    "Sutun adi/kod/etiket icerecek sekilde 3 sutun oldugundan emin olun."
                )
            else:
                cb_df = cb_df[[col_sutun, col_kod, col_etiket]].dropna(subset=[col_sutun, col_kod])
                imported: dict[str, dict[str, str]] = {}
                skipped_cols = set()
                for sutun, grp in cb_df.groupby(col_sutun):
                    sutun = str(sutun).strip()
                    if sutun not in active_columns:
                        skipped_cols.add(sutun)
                        continue
                    imported[sutun] = {str(k).strip(): str(v).strip() if pd.notna(v) else ""
                                        for k, v in zip(grp[col_kod], grp[col_etiket])}
                if imported:
                    for col, labels in imported.items():
                        st.session_state[decode_key][col] = labels
                    st.success(f"✅ {len(imported)} sutun icin kod sozlugu yuklendi: "
                               + ", ".join(imported.keys()))
                if skipped_cols:
                    st.caption(
                        f"ℹ️ {len(skipped_cols)} sutun dosyada mevcut degil, atlandi (muhtemelen bu "
                        f"anket dosyasinda bu sutunlar yok): {', '.join(sorted(skipped_cols)[:10])}"
                        + ("..." if len(skipped_cols) > 10 else "")
                    )
        except Exception as e:  # noqa: BLE001
            st.error(f"Kod sozlugu dosyasi okunamadi: {e}")

decode_exprs = []
for col, label_map in st.session_state[decode_key].items():
    if not label_map or col not in active_columns:
        continue
    when_parts = []
    for code, label in label_map.items():
        code_esc, label_esc = code.replace("'", "''"), label.replace("'", "''")
        when_parts.append(f"WHEN {an.cat_expr(col)} = '{code_esc}' THEN '{label_esc}'")
    new_col = f"{col}__etiket"
    decode_exprs.append(f"(CASE {' '.join(when_parts)} ELSE {an.cat_expr(col)} END) AS {an.ident(new_col)}")

# =========================================================== turetilmis sure
# Yolculuk suresi dogrudan bir sutun olarak yoksa (bircok gercek anket
# dosyasinda yok), ama baslangic/bitis saati eslenmisse, sure otomatik
# olarak (bitis - baslangic, gece yarisini gecen yolculuklar icin duzeltilmis)
# turetilir - boylece "Yolculuk Sureleri" bolumu bos kalmaz.
duration_expr_sql = None
# mapping["duration"] daha once BU fonksiyonun kendisi tarafindan otomatik
# atanmis olabilir ("_yolculuk_suresi_dk"). Eski kontrol sadece "hic sure
# eslenmemisse" turetiyordu; ama bir onceki calistirmada turetilip mapping'e
# yazildiktan sonraki her rerun'da mapping.get("duration") artik DOLU
# gorunuyor, guard atlaniyor, VIEW'a sutun eklenmiyor ama mapping hala o
# sutunu gosteriyordu -> "column not found" hatasi. Kendi urettigimiz
# sentinel deger icin de yeniden turetmeye izin veriyoruz ki her rerun'da
# VIEW ile mapping birbiriyle tutarli kalsin.
duration_is_auto = mapping.get("duration") in (None, "", "_yolculuk_suresi_dk")
if duration_is_auto and an.has(mapping, "start_time", "end_time") and \
        mapping["start_time"] in active_columns and mapping["end_time"] in active_columns:
    s_e = an.minute_of_day_expr(mapping["start_time"])
    e_e = an.minute_of_day_expr(mapping["end_time"])
    duration_expr_sql = f"((({e_e}) - ({s_e}) + 1440) % 1440)"
elif mapping.get("duration") == "_yolculuk_suresi_dk":
    # Once turetilmisti ama artik baslangic/bitis saati eslemesi gecerli
    # degil (kullanici kaldirdi/degistirdi ya da aktif tablo degisti) -
    # stale referansi temizle, yoksa asagidaki analizler var olmayan bir
    # sutuna bakip cokerdi.
    mapping["duration"] = None

# ===================================================== turetilmis aktarma sayisi
# "Aktarma sayisi" dogrudan bir sutun olarak NEREDEYSE HICBIR gercek dosyada
# bulunmaz - ama "Ulasim Turu" (mode) cogu UAP2040-tarzi ankette
# "yol_arac1, yol_arac2, yol_arac3" gibi HER YOLCULUK icin en fazla N ayri
# "bacak" (leg) turunu tutan TEKRARLAYAN bir grubun PARCASIDIR (bkz.
# column_groups double-melt). Boyle bir durumda, doldurulmus (bos olmayan)
# bacak sayisi - 1 = aktarma sayisidir (1 tur kullanildi = 0 aktarma, 2 tur
# = 1 aktarma, vb.) - bu, "duration" icin baslangic/bitis saatinden sure
# turetmekle AYNI mantik: dogrudan sutun yoksa ama turetilebilecek ham
# veri varsa, "Aktarmali Yolculuklar" bolumu bos kalmasin.
transfer_expr_sql = None
transfer_is_auto = mapping.get("transfer") in (None, "", "_aktarma_sayisi")
if transfer_is_auto and mapping.get("mode"):
    _m = re.match(r"^(.*?)(\d+)$", mapping["mode"])
    if _m:
        _prefix = _m.group(1)
        _leg_cols = sorted(
            (c for c in active_columns if re.match(rf"^{re.escape(_prefix)}\d+$", c)),
            key=lambda c: int(re.match(rf"^{re.escape(_prefix)}(\d+)$", c).group(1)),
        )
        if len(_leg_cols) >= 2:
            _filled_count = " + ".join(
                f"CASE WHEN {an.cat_expr(c)} IS NOT NULL THEN 1 ELSE 0 END" for c in _leg_cols
            )
            transfer_expr_sql = f"GREATEST(({_filled_count}) - 1, 0)"
elif mapping.get("transfer") == "_aktarma_sayisi":
    # Once turetilmisti ama artik "mode" eslemesi (ya da onun bacak
    # kardesleri) gecerli degil - stale referansi temizle.
    mapping["transfer"] = None

extra_derived = list(decode_exprs)
if duration_expr_sql:
    extra_derived.append(f"{duration_expr_sql} AS {an.ident('_yolculuk_suresi_dk')}")
if transfer_expr_sql:
    extra_derived.append(f"{transfer_expr_sql} AS {an.ident('_aktarma_sayisi')}")

if extra_derived:
    DERIVED_TABLE = f"{ACTIVE_TABLE}__derived"
    ok = safe("Turetilmis sutunlar", con.execute,
              f"CREATE OR REPLACE VIEW {an.ident(DERIVED_TABLE)} AS "
              f"SELECT tt.*, {', '.join(extra_derived)} FROM {an.ident(ACTIVE_TABLE)} AS tt")
    if ok is not None:
        TABLE = DERIVED_TABLE
        active_columns = table_columns(con, TABLE)
        active_dtypes = table_dtypes(con, TABLE)
        if duration_expr_sql:
            mapping["duration"] = "_yolculuk_suresi_dk"
            st.sidebar.caption("⏱️ Yolculuk suresi, baslangic/bitis saatinden otomatik turetildi.")
        if transfer_expr_sql:
            mapping["transfer"] = "_aktarma_sayisi"
            st.sidebar.caption("🔀 Aktarma sayisi, ulasim turu bacaklarindan (yol_arac1/2/3...) otomatik turetildi.")
    else:
        TABLE = ACTIVE_TABLE
else:
    TABLE = ACTIVE_TABLE  # asagidaki tum analiz kodu aktif tablo uzerinde calisir

grouped_concepts = concepts_by_group()
# DERIVED (kod cozme / sure) sutunlar sonradan eklendigi icin, siralama
# onbellegi de bu degisikligi yakalayacak sekilde map_key'e ek bilgi katilir
# (aksi halde yeni bir kod cozme eklendiginde eslestirme kutulari eski/
# eksik aday listesiyle takili kalirdi).
rankings_cache_key = (f"{map_key}|{sorted(str(x) for x in st.session_state[decode_key].items())}"
                      f"|{bool(duration_expr_sql)}|{bool(transfer_expr_sql)}")
rankings = safe("Aday sutun siralama", compute_rankings, rankings_cache_key, active_columns) or {}

if st.sidebar.button("🗑️ Tum onbellegi temizle"):
    data_io.clear_cache()
    st.sidebar.success("Onbellek temizlendi. Sayfayi yenileyin.")

# =========================================================== TABS
# Sekmeler, UAP 2040 raporunun 1.5. "Arastirma Sonuclari" bolumundeki
# numarali basliklarla (1.5.1.1 ... 1.5.2.9) AYNI SIRAYLA ve AYNI konu
# kirilimiyla duzenlenmistir - boylece disaridan bakan biri raporla
# dashboard arasinda dogrudan eslesme kurabilir. Her sekme SADECE kendi
# konusuyla ilgili Grafik/Tablo'lari icerir (rapor numaralariyla etiketli).
(tab_overview, tab_genel, tab_nufus, tab_cinsyas, tab_egitim, tab_istihdam,
 tab_gelir, tab_arac, tab_yolyol, tab_amacmod, tab_mekan, tab_sure, tab_saat,
 tab_gelyol, tab_aktarma, tab_maliyet, tab_ozelarac, tab_ileri, tab_compare,
 tab_ai, tab_ref) = st.tabs([
    "🆔 Kimlik & Veri Profili",
    "📋 Orneklem / Genel Bilgiler",
    "👨‍👩‍👧 Nufus & Hanehalki Buyuklugu",
    "🧑‍🤝‍🧑 Cinsiyet ve Yas",
    "🎓 Ogrenci & Egitim",
    "💼 Calisan Nufus & Istihdam",
    "💰 Gelir Durumu",
    "🚘 Arac Sahipligi",
    "🚶 Yolculuk Oranlari",
    "🎯 Amac & Ulasim Turu",
    "🗺️ Mekansal Dagilim",
    "⏱️ Yolculuk Sureleri",
    "🕒 Saatlik Dagilim",
    "📈 Gelir-Yolculuk Iliskisi",
    "🔀 Aktarmali Yolculuklar",
    "💸 Yolculuk Maliyetleri",
    "🅿️ Ozel Arac Yolculuklari",
    "🧭 Ileri Analizler",
    "🔄 Karsilastirma",
    "🤖 Yapay Zeka Icgorulari",
    "📖 UAP 2040 Referans",
])

# ----------------------------------------------------------------- GENEL BAKIS
@st.fragment
def _frag_tab_overview():
    with tab_overview:
        section_title("Kimlik / Bolge Eslestirmesi", "🆔")
        st.caption(
            "Bu kavramlar hem Demografik hem Yolculuk analizlerinde ortak kullanilir "
            "(ilce, hane/kisi kimligi). Bir kez burada eslestirin, diger sekmelerde "
            "otomatik kullanilir."
        )
        render_mapping_grid(mapping, grouped_concepts.get("kimlik", []), active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_kimlik")

        with st.expander("📋 Ham veri onizleme (ilk 20 satir)"):
            preview = safe("Onizleme", con.execute, f"SELECT * FROM {an.ident(ACTIVE_TABLE)} LIMIT 20")
            if preview is not None:
                st.dataframe(preview.fetchdf(), width="stretch")
            st.caption(f"Aktif tablo (`{ACTIVE_TABLE}`) toplam {active_rows:,} satir x "
                       f"{len(active_columns)} sutun (TAMAMI analiz edilir).")

        st.divider()
        section_title("Veri Profili (TAM VERI uzerinde)", "📊")
        prof = safe("Veri profili", data_profile, con, TABLE, active_rows, active_columns, active_dtypes)
        if prof is not None:
            mapped_count = sum(1 for v in mapping.values() if v)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Satir", f"{active_rows:,}")
            c2.metric("Sutun Sayisi", prof["n_cols"])
            c3.metric("Mukerrer Satir", f"{prof['dup_rows']:,}" if prof["dup_rows"] is not None else "atlandi (cok buyuk)")
            c4.metric("Eslestirilen Kavram", f"{mapped_count}/{len(CONCEPTS)}")

            if prof.get("profiled_cols_truncated"):
                st.caption(
                    f"⚡ Sutun sayisi cok fazla ({prof['n_cols']:,}) - hiz icin eksik-veri "
                    f"profillemesi ilk {prof['profiled_col_count']:,} sutunla sinirlandi."
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Eksik Veri Orani (%) — ilk 15 sutun**")
                miss = prof["missing_pct"].head(15)
                if miss.sum() > 0:
                    fig = px.bar(miss[miss > 0], orientation="h",
                                 labels={"value": "Eksik %", "index": "Sutun"},
                                 color_discrete_sequence=PALETTE)
                    fig.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.success("Eksik veri tespit edilmedi.")
            with col_b:
                st.markdown("**Sayisal Sutunlar**")
                if prof["numeric_cols"]:
                    st.dataframe(pd.DataFrame({"Sutun": prof["numeric_cols"]}), width="stretch")
                else:
                    st.write("Otomatik tespit edilen sayisal sutun yok (metin olarak okunmus olabilir; "
                             "analizler yine de sayisal donusum uygulayarak calisir).")

            with st.expander("Kategorik sutun listesi"):
                st.write(prof["cat_cols"])

        if ERROR_LOG:
            with st.expander(f"🪲 Bu yuklemede olusan hatalar ({len(ERROR_LOG)})", expanded=False):
                for label, err in ERROR_LOG:
                    st.code(f"[{label}]\n{err}", language="text")

_frag_tab_overview()
# ----------------------------------------------------------------- 1.2-1.3 GENEL BILGILER / ORNEKLEM
@st.fragment
def _frag_tab_genel():
    with tab_genel:
        hh_stats = hh_district = gender_dist = avgage = None

        # KULLANICI GERI BILDIRIMI: bu sekme eskiden EN USTTE, kullanicinin
        # kendi verisinden ONCE, raporun sabit Izmir referans tablosunu (Tablo 1)
        # gosteriyordu - kullanici (haklı olarak) "neden benim verim degil de
        # Izmir'i aciyorsun" diye sasirdi/rahatsiz oldu. Uygulamanin geri kalaninda
        # (orn. Grafik 1-2'deki referans cizgisi/bar) yerlesik dogru desen "once
        # KULLANICININ KENDI verisi, referans sadece EK BAGLAM olarak" seklinde -
        # burasi bu desenden sapiyordu. Fix: sira degistirildi (once SIZIN
        # veriniz), Izmir referans tablosu acikca etiketlenmis, varsayilan KAPALI
        # bir expander'a alindi.
        report_item("Tablo 2", "Ilcelere Gore Orneklem Buyuklugu (Yuklenen Veri)")
        st.caption(
            "Bu tablo SIZIN yukledigin CSV'den hesaplanir: ilce basina kac anket/hane "
            "kaydi oldugunu gosterir. Hane kimligi eslenmisse tekil hane sayisi, "
            "eslenmemisse satir sayisi kullanilir."
        )
        if mapping.get("district"):
            id_col_for_count = mapping.get("household_id") or mapping.get("person_id")
            sdf = safe("Orneklem Buyuklugu", q_sample_size_by_district, con, TABLE, mapping["district"], id_col_for_count)
            if sdf is not None:
                fig = px.bar(sdf, x="Ilce", y="Orneklem_Buyuklugu", color_discrete_sequence=PALETTE,
                             title="Ilcelere Gore Orneklem Buyuklugu")
                st.plotly_chart(fig, width="stretch")
                with st.expander("Tablo olarak gor"):
                    st.dataframe(sdf, width="stretch")
        else:
            missing_note("district")

        st.divider()
        with st.expander("📖 Tablo 1 - UAP 2040 raporunun KENDI Izmir verisi (referans/karsilastirma icin, sizin veriniz DEGIL)"):
            st.caption(
                "Bu tablo, raporun KENDI orijinal Izmir verisidir (sizin yuklediginiz "
                "veriden hesaplanmaz, sizin verinizle dogrudan ilgisi yoktur) - "
                "sadece rapordaki orneklem metodolojisine referans/karsilastirma "
                "olmasi icin burada tutulur."
            )
            tbl1 = pd.DataFrame(bm.UAP2040_REFERENCE["tablo1_ilce_mahalle"]).T
            tbl1.columns = ["Mahalle Sayisi", "Mahalle Toplam Nufusu", "Orneklem Mahalle Sayisi", "Orneklem Nufusu"]
            st.dataframe(tbl1, width="stretch")

        st.divider()
        report_item("Sekil 2", "Mahalle Bazinda Anket Sayisi")
        if mapping.get("zone"):
            zdf = safe("Mahalle Bazinda Anket", q_zone_survey_counts, con, TABLE, mapping["zone"])
            if zdf is not None:
                fig = px.bar(zdf, x="Mahalle", y="Anket_Sayisi", color_discrete_sequence=PALETTE,
                             title="Mahalle/Zon Bazinda Anket Sayisi (ilk 40)")
                st.plotly_chart(fig, width="stretch")
        else:
            missing_note("zone")

        st.divider()
        report_item("Sekil 3-8", "Organizasyon Semasi, Saha Ekibi ve Uygulama Fotograflari")
        report_skip_note("statik rapor fotograflari/organizasyon semalari")

_frag_tab_genel()
# ----------------------------------------------------------------- 1.5.1.1-2 NUFUS & HANEHALKI BUYUKLUGU
@st.fragment
def _frag_tab_nufus():
    with tab_nufus:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Hane, Kimlik)", "🔗")
        nufus_concepts = ["household_id", "person_id", "household_size"]
        render_mapping_grid(mapping, nufus_concepts, active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_nufus")
        st.divider()

        report_item("Sekil 9", "Nufus Buyuklugunun Trafik Analiz Bolgelerine Gore Dagilimi")
        report_skip_note()

        st.divider()
        report_item("Grafik 1", "Ortalama Hanehalki Buyuklugu")
        hh_district = safe("Ilce Bazinda Hanehalki Buyuklugu", household_size_by_district, con, TABLE, mapping)
        if hh_district is not None and not hh_district.empty:
            fig = px.bar(hh_district, x="Ilce", y="Ortalama Hanehalki Buyuklugu",
                         color_discrete_sequence=PALETTE, title="Ilce Bazinda Ortalama Hanehalki Buyuklugu")
            ref_line = bm.UAP2040_REFERENCE["hanehalki_buyuklugu"]["ortalama"]
            fig.add_hline(y=ref_line, line_dash="dash", line_color="crimson",
                          annotation_text=f"UAP2040 referans: {ref_line}")
            st.plotly_chart(fig, width="stretch")
            render_district_map(hh_district, "Ilce", "Ortalama Hanehalki Buyuklugu",
                                "Ilce Bazinda Ortalama Hanehalki Buyuklugu", key="hh_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("household_id"))
        else:
            missing_note("household_size")

        st.divider()
        report_item("Grafik 2", "Hanehalki Buyuklugunun Diger Iller ile Karsilastirilmasi (TUIK)")
        hh_stats = safe("Hanehalki Buyuklugu", household_size_stats, con, TABLE, mapping)
        if mapping.get("household_size") and mapping.get("household_id"):
            consistency = safe("Hanehalki buyuklugu tutarlilik kontrolu", repeated_value_consistency,
                                con, TABLE, mapping["household_id"], mapping["household_size"])
            if consistency is not None and consistency["n_inconsistent"] > 0:
                st.warning(
                    f"⚠️ **Veri kalitesi uyarisi**: `{mapping['household_size']}` sutunu, "
                    f"{consistency['n_inconsistent']}/{consistency['n_keys']} hanede "
                    f"(%{consistency['pct']:.0f}) AYNI hane icindeki farkli satirlarda "
                    "FARKLI degerler aliyor. Bu ya bir veri kalitesi sorunu ya da bu sutunun "
                    "aslinda 'hanehalki buyuklugu' olmadigini (orn. kisinin hane icindeki sira "
                    "numarasi olabilir) gosterebilir - asagidaki ortalama, her hane icin "
                    "bu tutarsiz degerlerden RASTGELE (ama tutarli) SECILMIS birine dayanir. "
                    "Sol menuden dogru sutunu kontrol edin/degistirin."
                )
        if hh_stats is not None:
            il_cmp = dict(bm.UAP2040_REFERENCE["il_karsilastirma_hanehalki_buyuklugu"])
            il_cmp["Yuklenen Veri"] = float(hh_stats["Ortalama"].iloc[0])
            cmp_df = pd.Series(il_cmp).sort_values(ascending=False).rename("Ortalama Hanehalki Buyuklugu").reset_index()
            cmp_df.columns = ["Il", "Ortalama Hanehalki Buyuklugu"]
            colors = ["crimson" if v == "Yuklenen Veri" else PALETTE[0] for v in cmp_df["Il"]]
            fig = px.bar(cmp_df, x="Il", y="Ortalama Hanehalki Buyuklugu",
                         title="Hanehalki Buyuklugu - TUIK Il Karsilastirmasi")
            fig.update_traces(marker_color=colors)
            st.plotly_chart(fig, width="stretch")
            note = safe("Hanehalki Buyuklugu yorumu", ai_local.narrative_household_size, hh_stats["Ortalama"].iloc[0])
            if note:
                st.caption(note)
        else:
            missing_note("household_size")

        st.divider()
        report_item("Tablo 4", "Hanehalki Buyuklugu Istatistikleri")
        if hh_stats is not None:
            st.dataframe(hh_stats.style.format("{:.2f}"), width="stretch")
            hh_ci = safe("Tablo 4 guven araligi", an.confidence_interval_mean, con, TABLE, mapping, "household_size")
            if hh_ci:
                st.caption(f"📏 %95 guven araligi: {hh_ci['ortalama']:.2f} ± {hh_ci['moe']:.2f} "
                           f"[{hh_ci['alt']:.2f} — {hh_ci['ust']:.2f}] (etkin ornek: {hh_ci['n_eff']:,.0f})")
        else:
            missing_note("household_size")

        render_range_quality(con, TABLE, mapping.get("household_size"), 1, 20, "Hanehalki Buyuklugu",
                             key="hhsize_quality")

_frag_tab_nufus()
# ----------------------------------------------------------------- 1.5.1.3 CINSIYET VE YAS
@st.fragment
def _frag_tab_cinsyas():
    with tab_cinsyas:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Cinsiyet, Yas)", "🔗")
        render_mapping_grid(mapping, ["gender", "age"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_cinsyas")
        st.divider()

        report_item("Grafik 3", "Cinsiyet Dagilimi")
        gender_dist = safe("Cinsiyet Dagilimi", gender_distribution, con, TABLE, mapping)
        if gender_dist is not None:
            fig = px.pie(values=gender_dist.values, names=gender_dist.index,
                         color_discrete_sequence=PALETTE, title="Cinsiyet Dagilimi")
            st.plotly_chart(fig, width="stretch")
            note = safe("Cinsiyet yorumu", ai_local.narrative_gender, gender_dist)
            if note:
                st.caption(note)
        else:
            missing_note("gender")

        st.divider()
        report_item("Grafik 4", "TUIK ve UAP 2040'a Gore Cinsiyetin Dagilimi")
        if gender_dist is not None:
            c = bm.UAP2040_REFERENCE["cinsiyet"]
            cmp = pd.DataFrame({
                "Kaynak": ["Yuklenen Veri", "Yuklenen Veri", "UAP 2040 Rapor", "UAP 2040 Rapor"],
                "Cinsiyet": ["Erkek", "Kadin", "Erkek", "Kadin"],
                "Yuzde": [
                    gender_dist.get("Erkek", np.nan), gender_dist.get("Kadin", np.nan),
                    c["uap_erkek_pct"], c["uap_kadin_pct"],
                ],
            }).dropna()
            if not cmp.empty:
                fig = px.bar(cmp, x="Cinsiyet", y="Yuzde", color="Kaynak", barmode="group",
                             color_discrete_sequence=PALETTE, title="Cinsiyet Dagilimi Karsilastirmasi")
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Cinsiyet kategorileri 'Erkek'/'Kadin' olarak eslesmedi - karsilastirma atlandi "
                           "(kod cozme ozelligiyle etiketleyebilirsiniz).")
        else:
            missing_note("gender")

        st.divider()
        report_item("Grafik 5-6", "Ilce Bazli Cinsiyet Karsilastirmasi (Erkek / Kadin)")
        gender_district = safe("Ilce Bazinda Cinsiyet", gender_by_district, con, TABLE, mapping)
        if gender_district is not None and not gender_district.empty:
            gcols = [c for c in gender_district.columns if c != "Ilce"]
            gd1, gd2 = st.columns(2)
            for i, col in enumerate(gcols[:2]):
                with (gd1 if i == 0 else gd2):
                    fig = px.bar(gender_district, x="Ilce", y=col, color_discrete_sequence=[PALETTE[i]],
                                 title=f"Ilce Bazli {col} Orani (%)")
                    st.plotly_chart(fig, width="stretch")
            with st.expander("Tablo olarak gor"):
                fmt = {c: "{:.1f}" for c in gcols}
                st.dataframe(gender_district.style.format(fmt), width="stretch")
        else:
            missing_note("gender", "district")

        st.divider()
        report_item("Grafik 7", "Ilcelere Gore Ortalama Yas")
        age_district = safe("Ilce Bazinda Yas", age_by_district, con, TABLE, mapping)
        avgage = safe("Ortalama Yas", avg_age_fn, con, TABLE, mapping)
        if age_district is not None and not age_district.empty:
            fig = px.bar(age_district, x="Ilce", y="Ortalama Yas", color_discrete_sequence=PALETTE,
                         title="Ilcelere Gore Ortalama Yas")
            ref_line = bm.UAP2040_REFERENCE["yas"]["ortalama_yas"]
            fig.add_hline(y=ref_line, line_dash="dash", line_color="crimson",
                          annotation_text=f"UAP2040 referans: {ref_line}")
            st.plotly_chart(fig, width="stretch")
            render_district_map(age_district, "Ilce", "Ortalama Yas", "Ilcelere Gore Ortalama Yas",
                                color_scale="Teal", key="age_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("person_id"))
        else:
            missing_note("age", "district")
        if avgage is not None:
            st.metric("Genel Ortalama Yas", f"{avgage:.1f}",
                       help=f"UAP 2040 referansi: {bm.UAP2040_REFERENCE['yas']['ortalama_yas']}")
            note = safe("Yas yorumu", ai_local.narrative_avg_age, avgage)
            if note:
                st.caption(note)

        st.divider()
        report_item("Grafik 8", "Nufus Piramidi")
        pyramid = safe("Nufus Piramidi", age_pyramid, con, TABLE, mapping)
        if pyramid is not None:
            gcols = [c for c in pyramid.columns if c != "age_group"]
            if len(gcols) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Bar(y=pyramid["age_group"], x=-pyramid[gcols[0]],
                                      name=str(gcols[0]), orientation="h"))
                fig.add_trace(go.Bar(y=pyramid["age_group"], x=pyramid[gcols[1]],
                                      name=str(gcols[1]), orientation="h"))
                fig.update_layout(barmode="relative", title="Nufus Piramidi", height=500)
                st.plotly_chart(fig, width="stretch")
        else:
            missing_note("age", "gender")

        render_range_quality(con, TABLE, mapping.get("age"), 0, 110, "Yas", key="age_quality")

_frag_tab_cinsyas()
# ----------------------------------------------------------------- 1.5.1.4 OGRENCI & EGITIM
@st.fragment
def _frag_tab_egitim():
    with tab_egitim:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Egitim, Ogrencilik, Okuma-Yazma)", "🔗")
        render_mapping_grid(mapping, ["education", "student", "literacy"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_egitim")
        st.divider()

        report_item("Grafik 9", "Okuma-Yazma Orani")
        lit_dist = safe("Okuma-Yazma", category_distribution, con, TABLE, mapping, "literacy", dedup_by="person_id",
                         group_siblings=col_group_siblings)
        if lit_dist is not None:
            fig = px.pie(values=lit_dist.values, names=lit_dist.index, color_discrete_sequence=PALETTE,
                         title="Okuma-Yazma Orani")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"UAP 2040 referansi: %{bm.UAP2040_REFERENCE['okuryazarlik']['uap_pct']}")
        else:
            missing_note("literacy")

        st.divider()
        report_item("Grafik 10", "Ilcelere Gore Okuryazarlik Orani")
        lit_district = safe("Ilce Bazinda Okuma-Yazma", category_by_district, con, TABLE, mapping, "literacy",
                             dedup_by="person_id", group_siblings=col_group_siblings)
        if lit_district is not None and not lit_district.empty:
            with st.expander("Ilce bazinda okuma-yazma tablosu (%)", expanded=True):
                st.dataframe(lit_district, width="stretch")
        else:
            missing_note("literacy", "district")

        st.divider()
        report_item("Grafik 11", "Egitim Duzeyi (%)")
        edu_dist = safe("Egitim Duzeyi", category_distribution, con, TABLE, mapping, "education",
                         dedup_by="person_id", group_siblings=col_group_siblings)
        if edu_dist is not None:
            fig = px.bar(x=edu_dist.index, y=edu_dist.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Egitim Duzeyi", "y": "%"}, title="Egitim Duzeyi Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("education")

        st.divider()
        report_item("Grafik 12", "Ilcelere Gore Egitim Duzeyi")
        edu_by_district = safe("Ilce Bazinda Egitim", category_by_district, con, TABLE, mapping, "education",
                                dedup_by="person_id", group_siblings=col_group_siblings)
        if edu_by_district is not None:
            with st.expander("Ilce bazinda egitim duzeyi tablosu (%)", expanded=True):
                st.dataframe(edu_by_district, width="stretch")
        else:
            missing_note("education", "district")

        st.divider()
        report_item("Grafik 13", "Izmir'de Ogrenci Nufus Orani (%)")
        student_dist = safe("Ogrencilik Durumu", category_distribution, con, TABLE, mapping, "student",
                             dedup_by="person_id", group_siblings=col_group_siblings)
        if student_dist is not None:
            fig = px.pie(values=student_dist.values, names=student_dist.index,
                         color_discrete_sequence=PALETTE, title="Ogrencilik Durumu Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("student")

        st.divider()
        report_item("Tablo 5", "Ilcelere Gore Ogrenci Nufusu")
        if an.has(mapping, "student", "district"):
            src5 = an._dedup_source(TABLE, mapping.get("person_id"), [mapping["district"], mapping["student"]])
            t5 = safe("Ilce Ogrenci Sayisi", q_student_count_by_district, con, src5, mapping["district"], mapping["student"])
            if t5 is not None:
                st.dataframe(t5, width="stretch")
        else:
            missing_note("student", "district")

        st.divider()
        report_item("Grafik 14", "Ogrencilerin Devam Ettikleri Okullar (Egitim Duzeyi Kirilimi)")
        if an.has(mapping, "student", "education"):
            edu_by_student = safe("Ogrenci x Egitim", an.crosstab_sql, con, TABLE, mapping["student"], mapping["education"])
            if edu_by_student is not None and not edu_by_student.empty:
                fig = px.imshow(edu_by_student, text_auto=".1f", aspect="auto", color_continuous_scale="Purples",
                                title="Ogrencilik Durumu x Egitim Duzeyi (satir %)")
                st.plotly_chart(fig, width="stretch")
        else:
            missing_note("student", "education")

        st.divider()
        report_item("Sekil 10-11", "Ogrenci Nufusun ve Okuduklari Okullarin TAZ'lara Dagilimi")
        report_skip_note()

_frag_tab_egitim()
# ----------------------------------------------------------------- 1.5.1.5 CALISAN NUFUS & ISTIHDAM
@st.fragment
def _frag_tab_istihdam():
    with tab_istihdam:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Istihdam, Bolge)", "🔗")
        render_mapping_grid(mapping, ["employment", "district", "district_dest"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_istihdam")
        st.divider()

        report_item("Grafik 15", "Calisma Durumu")
        emp_dist = safe("Calisma Durumu", category_distribution, con, TABLE, mapping, "employment",
                         dedup_by="person_id", group_siblings=col_group_siblings)
        if emp_dist is not None:
            fig = px.bar(x=emp_dist.index, y=emp_dist.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Durum", "y": "%"}, title="Calisma Durumu Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("employment")

        st.divider()
        report_item("Grafik 16-17", "Aktif Nufus, Isgucu Dagilimlari ve Istatistikleri")
        emp_counts = safe("Calisma Durumu (Sayi)", raw_count_sql, con, TABLE, mapping.get("employment") or "")
        if emp_counts is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Kisi (eslestirilen alan)", f"{int(emp_counts.sum()):,}")
            c2.metric("En Buyuk Kategori", str(emp_counts.idxmax()))
            c3.metric("En Buyuk Kategori Payi", f"%{emp_counts.max() / emp_counts.sum() * 100:.1f}")
            with st.expander("Kategori bazinda sayilar"):
                st.dataframe(emp_counts.rename("Kisi Sayisi").to_frame(), width="stretch")
        else:
            missing_note("employment")

        st.divider()
        report_item("Grafik 18", "Ilcelere Gore Isgucu Dagilimi (%)")
        emp_by_district = safe("Ilce Bazinda Istihdam", category_by_district, con, TABLE, mapping, "employment",
                                dedup_by="person_id", group_siblings=col_group_siblings)
        if emp_by_district is not None:
            with st.expander("Ilce bazinda calisma durumu tablosu (%)", expanded=True):
                st.dataframe(emp_by_district, width="stretch")
        else:
            missing_note("employment", "district")

        st.divider()
        report_item("Grafik 19-20 / Tablo 6", "Calisanlarin Istihdam Edildikleri Ilceler ve Karsilama Orani")
        st.caption(
            "'Ilce / Bolge (Baslangic)' ikamet, 'Ilce / Bolge (Varis)' ise "
            "calisilan/istihdam ilcesi olarak yorumlanir. Isteğe bagli olarak "
            "sadece belirli calisma durumu kategorileri (orn. 'Calisan') "
            "istihdam tarafinda sayilabilir."
        )
        emp_positive19 = []
        if mapping.get("employment"):
            codes19 = safe("Calisma kodlari", raw_count_sql, con, TABLE, mapping["employment"])
            if codes19 is not None:
                emp_positive19 = st.multiselect(
                    "Hangi kategori(ler) 'Istihdam Edilen' sayilsin? (bos = tumu)",
                    list(codes19.index), key="emp_positive19",
                )
        hw = safe("Ikamet-Istihdam Karsilastirma", home_work_district_comparison, con, TABLE, mapping,
                  emp_positive19 or None, mapping.get("employment") if emp_positive19 else None)
        if hw is not None and not hw.empty:
            fig = px.bar(hw, x="Ilce", y=["Ikamet", "Istihdam"], barmode="group", color_discrete_sequence=PALETTE,
                         title="Ikamet vs Istihdam (Kisi/Kayit Sayisi)")
            st.plotly_chart(fig, width="stretch")
            fig2 = px.bar(hw, x="Ilce", y="Karsilama_Orani_%", color_discrete_sequence=[PALETTE[2]],
                          title="Istihdamin Ikameti Karsilama Orani (%)")
            st.plotly_chart(fig2, width="stretch")
            render_district_map(hw, "Ilce", "Istihdam", "Ilcelere Gore Istihdam (Varis) Yogunlugu",
                                color_scale="Magenta", key="hw_district")
            with st.expander("Tablo olarak gor (Tablo 6 - En Yogun Istihdam Bolgeleri)"):
                st.dataframe(hw.sort_values("Istihdam", ascending=False), width="stretch")
        else:
            missing_note("district", "district_dest")

        st.divider()
        report_item("Sekil 12-13", "Calisan Nufusun ve Istihdamin TAZ'lara Dagilimi")
        report_skip_note()

        st.divider()
        report_item("Grafik 21", "Calisan Nufusun Yas Gruplarina Dagilimi (%)")
        emp_positive21 = []
        if mapping.get("employment"):
            codes21 = safe("Calisma kodlari (21)", raw_count_sql, con, TABLE, mapping["employment"])
            if codes21 is not None:
                emp_positive21 = st.multiselect(
                    "Hangi kategori(ler) 'Calisan' sayilsin?", list(codes21.index),
                    default=list(codes21.index)[:1] if len(codes21.index) else [], key="emp_positive21",
                )
        age_grp = safe("Calisan Yas Gruplari", age_group_distribution, con, TABLE, mapping,
                        filter_concept="employment" if emp_positive21 else None,
                        filter_positive_values=emp_positive21 or None)
        if age_grp is not None:
            fig = px.bar(x=age_grp.index, y=age_grp.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Yas Grubu", "y": "%"}, title="Calisan Nufusun Yas Gruplarina Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("age")

        st.divider()
        report_item("Grafik 22", "Calisanlarin Sektorel Dagilimi (%)")
        st.caption("Ayni 'Calisma Durumu / Sektor' sutunu kullanilir - CSV'nizde sektor ayri bir sutunda "
                   "tutuluyorsa, o sutunu yukaridan 'Calisma Durumu / Sektor' olarak eslestirin.")
        if emp_dist is not None:
            fig = px.pie(values=emp_dist.values, names=emp_dist.index, color_discrete_sequence=PALETTE,
                         title="Sektorel/Durumsal Dagilim")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("employment")

        st.divider()
        report_item("Grafik 23", "Calismayanlarin Calismama Sebeplerine Gore Dagilimi (%)")
        st.caption("Ayni 'Calisma Durumu' dagiliminin icinde (orn. 'Ev Hanimi', 'Emekli', 'Issiz' gibi "
                   "kategoriler varsa) zaten gorunur - Grafik 15/22 ile ayni kaynak sutun kullanilir.")

_frag_tab_istihdam()
# ----------------------------------------------------------------- 1.5.1.6 GELIR DURUMU
@st.fragment
def _frag_tab_gelir():
    with tab_gelir:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Gelir)", "🔗")
        render_mapping_grid(mapping, ["income", "household_id", "district"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_gelir")
        st.divider()

        report_item("Grafik 24", "Ilcelere Gore Ortalama Hanehalki Geliri (TL)")
        inc_district = safe("Ilce Bazinda Gelir", income_by_district, con, TABLE, mapping)
        if inc_district is not None and not inc_district.empty:
            fig = px.bar(inc_district, x="Ilce", y="Ortalama Gelir", color_discrete_sequence=PALETTE,
                         title="Ilcelere Gore Ortalama Hanehalki Geliri")
            st.plotly_chart(fig, width="stretch")
            render_district_map(inc_district, "Ilce", "Ortalama Gelir",
                                "Ilcelere Gore Ortalama Hanehalki Geliri", color_scale="Greens", key="inc_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("household_id"))
        else:
            missing_note("income", "district")

        st.divider()
        report_item("Grafik 25", "Gelir Gruplarinin Dagilimi")
        quint = safe("Gelir Ceyrekligi", income_quintile_distribution, con, TABLE, mapping)
        if quint is not None:
            fig = px.bar(x=quint.index, y=quint.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Gelir Grubu", "y": "%"}, title="Gelir Grubu (5'li Dilim) Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("income")

        st.divider()
        report_item("Tablo 7", "Gelir Dagilimi Endeks")
        gl = safe("Gini/Lorenz", gini_and_lorenz, con, TABLE, mapping)
        inc_summary = safe("Gelir Ozeti", income_summary, con, TABLE, mapping)
        if inc_summary is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ortalama Gelir", f"{inc_summary['ortalama']:,.0f}")
            c2.metric("Medyan Gelir", f"{inc_summary['medyan']:,.0f}")
            c3.metric("Std. Sapma", f"{inc_summary['std']:,.0f}")
            if gl is not None:
                c4.metric("Gini Katsayisi", f"{gl[0]:.3f}")
            if quint is not None:
                st.dataframe(quint.rename("Yuzde").to_frame(), width="stretch")
            inc_ci = safe("Tablo 7 guven araligi", an.confidence_interval_mean, con, TABLE, mapping, "income")
            if inc_ci:
                st.caption(f"📏 Ortalama gelir icin %95 guven araligi: {inc_ci['ortalama']:,.0f} ± {inc_ci['moe']:,.0f} "
                           f"[{inc_ci['alt']:,.0f} — {inc_ci['ust']:,.0f}] (etkin ornek: {inc_ci['n_eff']:,.0f})")
        else:
            missing_note("income")

        st.divider()
        report_item("Grafik 26", "Izmir Gelir Dagilimi (Lorenz Egrisi)")
        if gl is not None:
            gini, lorenz = gl
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lorenz["Nufus Payi"], y=lorenz["Gelir Payi"],
                                      mode="lines", name="Lorenz Egrisi", fill="tozeroy"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Esitlik Cizgisi",
                                      line=dict(dash="dash", color="gray")))
            fig.update_layout(title=f"Lorenz Egrisi (Gini={gini:.3f}, tam veri)",
                              xaxis_title="Nufus Payi", yaxis_title="Gelir Payi")
            st.plotly_chart(fig, width="stretch")
            income_hist = safe("Gelir Histogrami", q_numeric_values, con, TABLE, mapping['income'])
            if income_hist is not None:
                fig = px.histogram(income_hist.rename(columns={"val": "gelir"}), x="gelir", nbins=40,
                                   color_discrete_sequence=PALETTE, title="Gelir Dagilimi Histogrami (Tam Veri)")
                st.plotly_chart(fig, width="stretch")
        else:
            missing_note("income")

        st.divider()
        report_item("Grafik 27", "Bolgeler Arasi Gini Katsayisi Karsilastirmasi")
        gini_district = safe("Ilce Bazinda Gini", gini_by_district, con, TABLE, mapping)
        if gini_district is not None and not gini_district.empty:
            fig = px.bar(gini_district, x="Ilce", y="Gini", color_discrete_sequence=PALETTE,
                         title="Ilce Bazinda Gini Katsayisi")
            if gl is not None:
                fig.add_hline(y=gl[0], line_dash="dash", line_color="crimson",
                              annotation_text=f"Genel Gini: {gl[0]:.3f}")
            st.plotly_chart(fig, width="stretch")
            st.caption("Not: guvenilir bir Gini hesabi icin en az 30 hane gozlemi olan ilceler gosterilir.")
            render_district_map(gini_district, "Ilce", "Gini", "Ilce Bazinda Gini Katsayisi",
                                color_scale="Reds", key="gini_district")
        else:
            missing_note("income", "district")

        render_iqr_quality(con, TABLE, mapping.get("income"), "Hanehalki Geliri", key="income_quality")

_frag_tab_gelir()
# ----------------------------------------------------------------- 1.5.1.7 ARAC SAHIPLIGI
@st.fragment
def _frag_tab_arac():
    with tab_arac:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Arac, Park Yeri)", "🔗")
        render_mapping_grid(mapping, ["vehicle_count", "park_location"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_arac")
        st.divider()

        report_item("Grafik 28", "Hanelerdeki Arac ve Otomobil Durumu")
        veh_dist = safe("Arac Sahipligi", vehicle_ownership_distribution, con, TABLE, mapping)
        if veh_dist is not None:
            fig = px.pie(values=veh_dist.values, names=veh_dist.index.astype(str),
                         color_discrete_sequence=PALETTE, title="Hanelerin Arac Sayisina Gore Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("vehicle_count")

        st.divider()
        report_item("Grafik 29", "Ilce Bazinda Arac Sahipligi")
        veh_rate_district = safe("Ilce Bazinda Arac Sahiplik Orani", vehicle_ownership_rate_by_district,
                                  con, TABLE, mapping)
        if veh_rate_district is not None and not veh_rate_district.empty:
            fig = px.bar(veh_rate_district, x="Ilce", y="Arac_Sahip_Yuzde", color_discrete_sequence=PALETTE,
                         title="Ilce Bazinda En Az 1 Aracin Sahip Oldugu Hane Yuzdesi")
            st.plotly_chart(fig, width="stretch")
            render_district_map(veh_rate_district, "Ilce", "Arac_Sahip_Yuzde",
                                "Ilce Bazinda Arac Sahiplik Orani (%)", color_scale="Oranges", key="veh_rate_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("household_id"))
        else:
            missing_note("vehicle_count", "district")

        st.divider()
        report_item("Grafik 30-31", "Arac Sayilarinin Turlere Gore Dagilimi")
        st.caption(
            "Aracinizin turlere gore (otomobil, motosiklet vb.) AYRI sutunlarda "
            "tutuldugu bir dosyanız varsa: sol menudeki '🧩 Sutun Gruplari → "
            "Gelismis → Tek Deger Haline Getirilebilir Gruplar' bolumunden her "
            "turu 'Topla (SUM)' ile birlestirin - olusan sutunlar (orn. "
            "`otomobil_sum`) burada ayrica secilip karsilastirilabilir."
        )

        st.divider()
        report_item("Tablo 8", "TUIK Otomobil Sahipligi (Rapor Referansi)")
        st.caption(f"UAP 2040 raporunda otomobil sahipligi TUIK verileriyle karsilastirilmistir "
                   f"(Izmir referansi icin 'UAP 2040 Referans' sekmesine bakin).")

        st.divider()
        report_item("Grafik 32", "Ilcelere Gore 1000 Kisiye Dusen Otomobil Sayisi")
        veh_district = safe("Ilce Bazinda Arac", vehicles_per_1000_by_district, con, TABLE, mapping)
        if veh_district is not None and not veh_district.empty:
            fig = px.bar(veh_district, x="Ilce", y="1000_Kisiye_Dusen_Arac",
                         color_discrete_sequence=PALETTE, title="Ilcelere Gore 1000 Kisiye Dusen Arac Sayisi")
            st.plotly_chart(fig, width="stretch")
            render_district_map(veh_district, "Ilce", "1000_Kisiye_Dusen_Arac",
                                "Ilcelere Gore 1000 Kisiye Dusen Arac Sayisi", color_scale="Oranges", key="veh_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("household_id"))
        else:
            missing_note("vehicle_count", "district")

        st.divider()
        report_item("Grafik 33", "Araclarin Aksamlari Park Ettigi Yerler")
        park_dist = safe("Park Yeri", category_distribution, con, TABLE, mapping, "park_location",
                          group_siblings=col_group_siblings)
        if park_dist is not None:
            fig = px.pie(values=park_dist.values, names=park_dist.index, color_discrete_sequence=PALETTE,
                         title="Araclarin Aksamlari Park Ettigi Yerler")
            st.plotly_chart(fig, width="stretch")
            ref = bm.UAP2040_REFERENCE["ozel_arac_yolculuk"]
            st.caption(f"UAP 2040 referansi: ucretsiz yol kenari %{ref['otopark_ucretsiz_yol_kenari_pct']}, "
                       f"isyeri/AVM %{ref['otopark_ucretsiz_isyeri_avm_pct']}, ucretli %{ref['otopark_ucretli_pct']}.")
        else:
            missing_note("park_location")

        st.divider()
        report_item("Sekil 14-15", "Otomobil Sayisi ve Sahipliginin TAZ'lara Dagilimi")
        report_skip_note()

_frag_tab_arac()
# ----------------------------------------------------------------- 1.5.2.1 YOLCULUK ORANLARI
@st.fragment
def _frag_tab_yolyol():
    with tab_yolyol:
        rate = rate_district = None
        section_title("Sutun Eslestirme (Kisi, Ilce, Mod)", "🔗")
        render_mapping_grid(mapping, ["person_id", "district", "mode"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_yolyol")
        st.divider()

        report_item("Grafik 34-35", "Brut ve Net Hareketlilik Oranlari (Yolculuk Sayisi/Nufus)")
        st.caption(
            "Rapor 'brut' (toplam nufus basina) ve 'net' (SADECE hareket edenler "
            "basina) oranlari ayirir. Aktif tablonuz sadece yolculuk yapanlari "
            "iceriyorsa (tipik bir 'Yolculuk Tablosu'), asagidaki oran NET "
            "hareketlilik oranina karsilik gelir."
        )
        rate = safe("Hareketlilik Orani", mobility_rate, con, TABLE, mapping)
        if rate is not None:
            st.metric("Hareketlilik Orani (yolculuk/kisi)", f"{rate:.2f}")
        else:
            missing_note("person_id")

        st.divider()
        report_item("Tablo 9", "Ilcelere Gore Yolculuk Sayilari ve Hareketlilik Oranlari")
        rate_district = safe("Ilce Bazinda Hareketlilik", mobility_rate_by_district, con, TABLE, mapping)
        if rate_district is not None and not rate_district.empty:
            fig = px.bar(rate_district, x="Ilce", y="Hareketlilik_Orani", color_discrete_sequence=PALETTE,
                         title="Ilce Bazinda Hareketlilik Orani")
            st.plotly_chart(fig, width="stretch")
            render_district_map(rate_district, "Ilce", "Hareketlilik_Orani",
                                "Ilce Bazinda Hareketlilik Orani", color_scale="Purples", key="rate_district")
            sample_size_caption(con, TABLE, mapping.get("district"), mapping.get("person_id"))
            note = safe("Hareketlilik yorumu", ai_local.narrative_district_extremes,
                         rate_district, "Ilce", "Hareketlilik_Orani", "Hareketlilik orani")
            if note:
                st.caption(note)
            with st.expander("Tablo olarak gor"):
                st.dataframe(rate_district, width="stretch")
        else:
            missing_note("district", "person_id")

        st.divider()
        report_item("Grafik 36 / Tablo 10", "Aracli-Aracsiz Yolculuk Oranlari")
        aracli_vals_yy = []
        if mapping.get("mode"):
            mode_codes_yy = safe("Ulasim turu kodlari", raw_count_sql, con, TABLE, mapping["mode"])
            if mode_codes_yy is not None:
                aracli_vals_yy = st.multiselect(
                    "Hangi ulasim turu deger(ler)i 'ARACLI' sayilsin? (yurume/bisiklet disindaki turleri secin)",
                    list(mode_codes_yy.index), key="aracli_yy",
                )
        split_yy = safe("Aracli/Aracsiz", mode_flag_split, con, TABLE, mapping, aracli_vals_yy,
                         group_siblings=col_group_siblings) if aracli_vals_yy else None
        if split_yy is not None:
            fig = px.pie(values=split_yy.values, names=split_yy.index, color_discrete_sequence=PALETTE,
                         title="Aracli-Aracsiz Yolculuk Oranlari")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Yukaridan en az bir 'Aracli' ulasim turu secin.")

_frag_tab_yolyol()
# ----------------------------------------------------------------- 1.5.2.2-3 AMAC & ULASIM TURU
@st.fragment
def _frag_tab_amacmod():
    with tab_amacmod:
        purpose_dist = mode_dist = None
        section_title("Sutun Eslestirme (Amac, Mod)", "🔗")
        render_mapping_grid(mapping, ["purpose", "mode", "district"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_amacmod")
        st.divider()

        report_item("Grafik 37", "Yolculuk Amaclari")
        purpose_dist = safe("Yolculuk Amaci", purpose_distribution, con, TABLE, mapping,
                             group_siblings=col_group_siblings)
        if purpose_dist is not None:
            fig = px.pie(values=purpose_dist.values, names=purpose_dist.index,
                         color_discrete_sequence=PALETTE, title="Yolculuk Amaci Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("purpose")

        st.divider()
        report_item("Tablo 11", "Amaclarina Gore Yolculuklarin Ilcelere Gore Dagilimi")
        purpose_by_district = safe("Ilce Bazinda Amac", category_by_district, con, TABLE, mapping, "purpose",
                                    group_siblings=col_group_siblings)
        if purpose_by_district is not None:
            st.dataframe(purpose_by_district, width="stretch")
        else:
            missing_note("purpose", "district")

        st.divider()
        report_item("Grafik 38-39", "Aracli ve Aracsiz Yolculuklarin Dagilimi (Genel ve Ilce Bazinda)")
        aracli_vals_am = []
        if mapping.get("mode"):
            mode_codes_am = safe("Ulasim turu kodlari (amacmod)", raw_count_sql, con, TABLE, mapping["mode"])
            if mode_codes_am is not None:
                aracli_vals_am = st.multiselect(
                    "Hangi ulasim turu deger(ler)i 'ARACLI' sayilsin?",
                    list(mode_codes_am.index), key="aracli_am",
                )
        if aracli_vals_am:
            split_am = safe("Aracli/Aracsiz (genel)", mode_flag_split, con, TABLE, mapping, aracli_vals_am,
                             group_siblings=col_group_siblings)
            split_am_d = safe("Aracli/Aracsiz (ilce)", mode_flag_split, con, TABLE, mapping, aracli_vals_am,
                               by_district=True, group_siblings=col_group_siblings)
            ca1, ca2 = st.columns(2)
            with ca1:
                if split_am is not None:
                    fig = px.pie(values=split_am.values, names=split_am.index, color_discrete_sequence=PALETTE,
                                 title="Aracli-Aracsiz Yolculuk Dagilimi")
                    st.plotly_chart(fig, width="stretch")
            with ca2:
                if split_am_d is not None and not split_am_d.empty:
                    fig = px.bar(split_am_d, x="Ilce", y=[c for c in split_am_d.columns if c != "Ilce"],
                                 barmode="stack", color_discrete_sequence=PALETTE,
                                 title="Ilce Bazinda Aracli-Aracsiz Dagilimi (%)")
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("Yukaridan en az bir 'Aracli' ulasim turu secin.")

        st.divider()
        report_item("Grafik 40-41 / Tablo 12", "Ulasim Turlerine Gore Dagilim (Tum ve Aracli Yolculuklar)")
        mode_dist = safe("Ulasim Turu", mode_distribution, con, TABLE, mapping,
                          group_siblings=col_group_siblings)
        cm1, cm2 = st.columns(2)
        with cm1:
            if mode_dist is not None:
                fig = px.pie(values=mode_dist.values, names=mode_dist.index,
                             color_discrete_sequence=PALETTE, title="Tum Yolculuklarin Ulasim Turlerine Gore Dagilimi")
                st.plotly_chart(fig, width="stretch")
            else:
                missing_note("mode")
        with cm2:
            if aracli_vals_am and mode_dist is not None:
                mode_dist_aracli = mode_dist[mode_dist.index.isin(aracli_vals_am)]
                if not mode_dist_aracli.empty:
                    mode_dist_aracli = mode_dist_aracli / mode_dist_aracli.sum() * 100
                    fig = px.pie(values=mode_dist_aracli.values, names=mode_dist_aracli.index,
                                 color_discrete_sequence=PALETTE, title="Aracli Yolculuklarin Ulasim Turlerine Gore Dagilimi")
                    st.plotly_chart(fig, width="stretch")
        for n in (safe("Amac/Mod yorumu", ai_local.narrative_purpose_mode, purpose_dist, mode_dist) or []):
            st.caption(n)
        pm_cross = safe("Amac x Mod", purpose_mode_crosstab, con, TABLE, mapping,
                         group_siblings=col_group_siblings)
        if pm_cross is not None and not pm_cross.empty:
            st.markdown("**Amac x Ulasim Turu (satir %) Isi Haritasi**")
            fig = px.imshow(pm_cross, text_auto=".1f", aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig, width="stretch")
            if pm_cross.shape[1] >= 30:
                st.caption(
                    "ℹ️ En yaygin ulasim turlerinden en fazla 30 tanesi gosterilir - yuzdeler HER ZAMAN "
                    "TUM ulasim turleri uzerinden dogru hesaplanir (sadece goruntu sinirlidir), bu yuzden "
                    "cok fazla farkli ulasim turu varsa satir toplami %100'den DUSUK gorunebilir."
                )

        st.divider()
        report_item("Grafik 42", "Ilcelere Gore Ulasim Turu Dagilimi")
        mode_by_district = safe("Ilce Bazinda Mod", category_by_district, con, TABLE, mapping, "mode",
                                 group_siblings=col_group_siblings)
        if mode_by_district is not None:
            st.dataframe(mode_by_district, width="stretch")
        else:
            missing_note("mode", "district")

        st.divider()
        report_item("Grafik 43-45", "Yolcu Hareketlerinin Arac Turlerine Gore Dagilimi")
        st.caption("Ayni 'Ulasim Turu' dagilimi kullanilir (Grafik 40-41 ile ayni kaynak) - "
                   "yolcu/surucu ayrimi icin ayri bir sutun eslenmemisse rapor'daki tam ayrim uretilemez.")

_frag_tab_amacmod()
# ----------------------------------------------------------------- 1.5.2.4 MEKANSAL DAGILIM
@st.fragment
def _frag_tab_mekan():
    with tab_mekan:
        section_title("Sutun Eslestirme (Baslangic/Varis Ilce)", "🔗")
        render_mapping_grid(mapping, ["district", "district_dest"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_mekan")
        st.divider()

        report_item("Tablo 14-15", "Ilcelere Gore Yolculuk Baslangic-Bitis Degerleri ve Oranlari")
        odm = safe("O-D Matrisi", od_matrix, con, TABLE, mapping)
        if odm is not None and not odm.empty:
            st.markdown("**Baslangic x Varis Ilce Matrisi (Yolculuk Sayisi)**")
            fig = px.imshow(odm, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                            labels={"x": "Varis Ilcesi", "y": "Baslangic Ilcesi"})
            st.plotly_chart(fig, width="stretch")
            with st.expander("Tablo olarak gor"):
                st.dataframe(odm, width="stretch")

            # ---- Sankey akis diyagrami: bu tur O-D (baslangic-varis) verisi icin
            # ulasim planlamasinda standart bir gorsellestirmedir (matris/isi
            # haritasindan farkli olarak, en YOGUN akislari GORSEL AGIRLIKLA
            # tek bakista gosterir). Karmasik gorunmesin diye SADECE en yogun
            # 30 akis gosterilir (odm zaten en fazla top_n=25 ilce iceriyor,
            # burada AYRICA hacme gore siralanir).
            st.markdown("**Baslangic → Varis Yolculuk Akislari (Sankey)**")
            odm_long = odm.reset_index().rename(columns={odm.reset_index().columns[0]: "Baslangic"})
            odm_long = odm_long.melt(id_vars="Baslangic", var_name="Varis", value_name="n")
            odm_long = odm_long[odm_long["n"] > 0].sort_values("n", ascending=False).head(30)
            if not odm_long.empty:
                origins = sorted(odm_long["Baslangic"].unique())
                dests = sorted(odm_long["Varis"].unique())
                origin_idx = {o: i for i, o in enumerate(origins)}
                dest_idx = {d: i + len(origins) for i, d in enumerate(dests)}
                sankey_labels = [f"{o} (B)" for o in origins] + [f"{d} (V)" for d in dests]
                sankey_fig = go.Figure(go.Sankey(
                    node=dict(label=sankey_labels, pad=14, thickness=14, color=PALETTE[0]),
                    link=dict(
                        source=[origin_idx[o] for o in odm_long["Baslangic"]],
                        target=[dest_idx[d] for d in odm_long["Varis"]],
                        value=odm_long["n"].tolist(),
                        color="rgba(150,150,150,0.35)",
                    ),
                ))
                sankey_fig.update_layout(title="En Yogun 30 Baslangic → Varis Akisi (B: Baslangic, V: Varis)",
                                         font_size=11, height=500)
                st.plotly_chart(sankey_fig, width="stretch")
        else:
            missing_note("district", "district_dest")

        st.divider()
        report_item("Grafik 46-47", "Zon Ici ve Zon Disi Yolculuklar")
        zone_io = safe("Zon Ici/Disi", zone_internal_external, con, TABLE, mapping)
        if zone_io is not None and not zone_io.empty:
            fig = px.bar(zone_io, x="Ilce", y=["Zon Ici Yuzde", "Zon Disi Yuzde"], barmode="stack",
                         color_discrete_sequence=PALETTE, title="Ilcelere Gore Zon Ici / Zon Disi Yolculuklar")
            st.plotly_chart(fig, width="stretch")
            render_district_map(zone_io, "Ilce", "Zon Ici Yuzde", "Ilcelere Gore Zon Ici Yolculuk Orani (%)",
                                color_scale="Blues", key="zone_io_district")
        else:
            missing_note("district", "district_dest")

_frag_tab_mekan()
# ----------------------------------------------------------------- 1.5.2.5 YOLCULUK SURELERI
@st.fragment
def _frag_tab_sure():
    with tab_sure:
        section_title("Sutun Eslestirme (Sure)", "🔗")
        render_mapping_grid(mapping, ["duration", "mode", "purpose", "district"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_sure")
        st.divider()

        report_item("Tablo 16 / Grafik 48-49", "Amaclarina ve Ulasim Turlerine Gore Yolculuk Sureleri")
        d1, d2 = st.columns(2)
        with d1:
            dur_mode = safe("Sure x Mod", duration_stats_by, con, TABLE, mapping, "mode")
            if dur_mode is not None:
                st.markdown("**Ulasim Turune Gore Ortalama Sure**")
                fig = px.bar(dur_mode, x="Grup", y="Ortalama Sure (dk)", color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, width="stretch")
            else:
                missing_note("duration", "mode")
        with d2:
            dur_purpose = safe("Sure x Amac", duration_stats_by, con, TABLE, mapping, "purpose")
            if dur_purpose is not None:
                st.markdown("**Amaca Gore Ortalama Sure**")
                fig = px.bar(dur_purpose, x="Grup", y="Ortalama Sure (dk)", color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, width="stretch")
            else:
                missing_note("duration", "purpose")

        st.divider()
        report_item("Grafik 50", "Yolculuklarin Basladiklari Ilcelere Gore Ortalama Seyahat Sureleri")
        dur_district = safe("Sure x Ilce", numeric_by_group_sql_cached, con, TABLE, mapping.get("district"),
                             mapping.get("duration"), "Ilce")
        if dur_district is not None:
            fig = px.bar(dur_district, x="Ilce", y="Ortalama", color_discrete_sequence=PALETTE,
                         title="Baslangic Ilcesine Gore Ortalama Yolculuk Suresi (dk)")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("duration", "district")

        st.divider()
        report_item("Tablo 17", "Kent Ici Toplu Tasima Araclarinda Ortalama Yolculuk Sureleri")
        st.caption("Grafik 48'deki 'Ulasim Turune Gore Ortalama Sure' tablosuyla ayni kaynak - "
                   "toplu tasima turlerini (otobus, metro vb.) o tablodan okuyabilirsiniz.")

        st.divider()
        report_item("Grafik 51 / Tablo 18 / Grafik 53", "Yolculuk Surelerinin Araliklara Gore Frekans Dagilimi (%)")
        dur_freq = safe("Sure Frekans Dagilimi", duration_frequency_distribution, con, TABLE, mapping)
        if dur_freq is not None:
            fig = px.bar(x=dur_freq.index, y=dur_freq.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Sure Araligi (dk)", "y": "%"}, title="Yolculuk Surelerinin Frekans Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("duration")

        st.divider()
        report_item("Grafik 52 / Tablo 19-20", "Ulasim Turlerine Gore Ortalama Yolculuk Surelerinin Dagilimi")
        st.caption("Grafik 48'deki 'Ulasim Turune Gore Ortalama Sure' grafigiyle ayni bilgiyi icerir.")

        st.divider()
        section_title("Veri Kalitesi Kontrolu (Sure)", "🔍")
        st.caption(
            "Benzer acik kaynak anket projelerinde (orn. CMAP 'My Daily Travel') "
            "yolculuklar analize sokulmadan once supheli/olasi hatali kayitlar "
            "ayrica raporlanir - boylece ortalama/medyan gibi ozetlerin birkac "
            "asiri deger yuzunden yaniltici cikip cikmadigi anlasilir. Asagidaki "
            "sayilar SADECE BILGILENDIRME icindir - hicbir satir otomatik "
            "silinmez, diger hesaplariniz DEGISMEZ."
        )
        dq = safe("Sure Veri Kalitesi", duration_quality_flags, con, TABLE, mapping)
        if dq is not None:
            q1, q2, q3 = st.columns(3)
            q1.metric("Gecerli Sure Kaydi", f"{dq['toplam']:,}")
            q2.metric("Sifir/Negatif Sure", f"{dq['gecersiz_sure']:,}", f"%{dq['gecersiz_yuzde']}")
            q3.metric(f"{dq['long_trip_minutes']} dk Uzeri (Olagandisi Uzun)",
                      f"{dq['olagandisi_uzun']:,}", f"%{dq['olagandisi_uzun_yuzde']}")
            notes = []
            if dq["gecersiz_sure"]:
                notes.append(
                    f"⚠️ {dq['gecersiz_sure']:,} kayitta sure sifir ya da negatif (bitis saati baslangictan "
                    f"once/ayni gorunuyor) - bu, saat girisinde bir hata olabilir, kontrol etmenizi oneririz."
                )
            if dq["olagandisi_uzun"]:
                p99_txt = f" (P99: {dq['p99']} dk)" if dq["p99"] is not None else ""
                notes.append(
                    f"ℹ️ {dq['olagandisi_uzun']:,} kayitta sure {dq['long_trip_minutes']} dakikadan uzun{p99_txt} - "
                    f"bunlarin bir kismi gercek (orn. sehirlerarasi) olabilir, bir kismi veri girisi hatasi "
                    f"olabilir; kesin degilse elle kontrol etmek en dogrusudur."
                )
            if not notes:
                st.success("✅ Sure alaninda sifir/negatif ya da asiri uzun kayit tespit edilmedi.")
            for n in notes:
                st.caption(n)
        else:
            missing_note("duration")

_frag_tab_sure()
# ----------------------------------------------------------------- 1.5.2.6 SAATLIK DAGILIM
@st.fragment
def _frag_tab_saat():
    with tab_saat:
        hourly_start = None
        section_title("Sutun Eslestirme (Baslangic/Bitis Saati)", "🔗")
        render_mapping_grid(mapping, ["start_time", "end_time", "purpose"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_saat")
        st.divider()

        report_item("Grafik 54", "Tum Yolculuklarin Baslangic ve Bitis Saatlerinin Dagilimi (%)")
        hourly_start = safe("Saatlik Dagilim (baslangic)", hourly_distribution, con, TABLE, mapping, "start_time")
        if hourly_start is not None:
            hourly_end = safe("Saatlik Dagilim (bitis)", hourly_distribution, con, TABLE, mapping, "end_time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(hourly_start.index), y=hourly_start.values,
                                      mode="lines+markers", name="Baslangic Saati"))
            if hourly_end is not None:
                fig.add_trace(go.Scatter(x=list(hourly_end.index), y=hourly_end.values,
                                          mode="lines+markers", name="Bitis Saati"))
            fig.update_layout(title="Yolculuklarin Saatlik Dagilimi (%, tam veri)", xaxis_title="Saat", yaxis_title="%")
            st.plotly_chart(fig, width="stretch")
            note = safe("Zirve saat yorumu", ai_local.narrative_peak_hour, hourly_start)
            if note:
                st.caption(note)
            z = bm.UAP2040_REFERENCE["zirve_saatler"]
            st.caption(f"UAP 2040 referansi: sabah zirvesi {z['sabah_zirve']}, aksam zirvesi {z['aksam_zirve']}.")
        else:
            missing_note("start_time")

        st.divider()
        section_title("Farkli Zaman Dilimlerinde Saatlik Dagilim (istege bagli)", "⏱️")
        st.caption(
            "Grafik 54, raporla birebir esli sabit 1 saatlik dilimlerle hesaplanir ve DEGISMEZ. "
            "Burada (benzer projelerdeki 'zaman agregasyon secici' ozelliginden esinlenerek) "
            "istege bagli olarak DAHA INCE ya da DAHA KABA dilimlerde de bakabilirsiniz - orn. "
            "zirve saatin tam olarak hangi 30 dakikada oldugunu gormek icin."
        )
        bin_options = {"15 dakika": 15, "30 dakika": 30, "60 dakika (Grafik 54 ile ayni)": 60, "120 dakika": 120}
        bin_label = st.selectbox("Zaman dilimi genisligi", list(bin_options.keys()), index=2,
                                  key=f"{ds.db_path}_{ACTIVE_TABLE}_saat_bin")
        bin_minutes = bin_options[bin_label]
        binned = safe("Zaman Dilimi Dagilimi", binned_time_distribution, con, TABLE, mapping, "start_time", bin_minutes)
        if binned is not None:
            fig = px.bar(x=binned.index, y=binned.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Zaman Dilimi", "y": "%"},
                         title=f"Yolculuk Baslangiclarinin Dagilimi ({bin_label})")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("start_time")

        st.divider()
        report_item("Tablo 21-22 / Grafik 55-59", "Baslangic ve Bitis Saatlerinin Amaclarina Gore Dagilimi")
        hourly_purpose = safe("Saatlik x Amac", hourly_distribution_by_purpose, con, TABLE, mapping)
        if hourly_purpose is not None:
            st.markdown("**Amaca Gore Saatlik Baslangic Dagilimi (%) - Tum 'Ev Uclu' Amaclar Tek Grafikte**")
            fig = px.line(hourly_purpose, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("start_time", "purpose")

        st.divider()
        report_item("Grafik 60-61 / Tablo 23-24 / Grafik 62-66", "Tum Yolculuklar ile Aracli Yolculuklarin Karsilastirilmasi")
        aracli_vals_saat = []
        if mapping.get("mode"):
            mode_codes_saat = safe("Ulasim turu kodlari (saat)", raw_count_sql, con, TABLE, mapping["mode"])
            if mode_codes_saat is not None:
                aracli_vals_saat = st.multiselect(
                    "Hangi ulasim turu deger(ler)i 'ARACLI' sayilsin?",
                    list(mode_codes_saat.index), key="aracli_saat",
                )
        if aracli_vals_saat and mapping.get("start_time") and mapping.get("mode"):
            hourly_aracli = safe("Saatlik (aracli)", q_hourly_filtered, con, TABLE, mapping["start_time"],
                                  mapping["mode"], tuple(aracli_vals_saat),
                                  tuple(col_group_siblings.get(mapping["mode"], [])))
            if hourly_aracli is not None and hourly_start is not None:
                hdf = hourly_aracli
                total_a = hdf["n"].sum()
                if total_a:
                    s_aracli = hdf.set_index("h")["n"].reindex(range(24), fill_value=0) / total_a * 100
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(hourly_start.index), y=hourly_start.values,
                                              mode="lines+markers", name="Tum Yolculuklar"))
                    fig.add_trace(go.Scatter(x=[f"{h:02d}:00" for h in s_aracli.index], y=s_aracli.values,
                                              mode="lines+markers", name="Aracli Yolculuklar"))
                    fig.update_layout(title="Tum Yolculuklar vs Aracli Yolculuklarin Baslangic Saatleri (%)",
                                      xaxis_title="Saat", yaxis_title="%")
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("Yukaridan en az bir 'Aracli' ulasim turu secin.")

_frag_tab_saat()
# ----------------------------------------------------------------- 1.5.2.7 GELIR-YOLCULUK ILISKISI
@st.fragment
def _frag_tab_gelyol():
    with tab_gelyol:
        section_title("Sutun Eslestirme (Gelir, Arac)", "🔗")
        render_mapping_grid(mapping, ["income", "purpose", "mode", "vehicle_count"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_gelyol")
        st.divider()

        report_item("Tablo 25 / Grafik 67", "Gelir Grubuna Gore Ortalama Yolculuk Degerleri ve Amaclarina Gore Iliski")
        ig_purpose = safe("Gelir x Amac", income_group_purpose, con, TABLE, mapping)
        if ig_purpose is not None:
            fig = px.bar(ig_purpose, barmode="group", color_discrete_sequence=PALETTE,
                         title="Gelir Grubuna Gore Yolculuk Amaci Dagilimi (%)")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Rapor referans tablosu (Tablo 25 - UAP 2040)"):
                st.dataframe(pd.DataFrame(bm.UAP2040_REFERENCE["gelir_grubu_ortalama_yolculuk"]).T, width="stretch")
        else:
            missing_note("income", "purpose")

        st.divider()
        report_item("Tablo 26 / Grafik 68", "Arac Turlerine Gore Yolculuklarin Gelir Gruplarina Dagilimi (%)")
        ig_mode = safe("Gelir x Mod", income_group_mode, con, TABLE, mapping)
        if ig_mode is not None:
            fig = px.imshow(ig_mode, text_auto=".1f", aspect="auto", color_continuous_scale="Greens",
                            title="Gelir Grubuna Gore Ulasim Turu Dagilimi (%)")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Rapor notu: {bm.UAP2040_REFERENCE['mod_gelir_iliskisi_notu']}")
        else:
            missing_note("income", "mode")

        st.divider()
        report_item("Tablo 27 / Grafik 69", "Otomobil Sahipligine Gore Hareketlilik Orani")
        mob_veh = safe("Arac Sahipligine Gore Hareketlilik", mobility_by_vehicle_ownership, con, TABLE, mapping)
        if mob_veh is not None and not mob_veh.empty:
            fig = px.bar(mob_veh, x="Arac_Grubu", y="Hareketlilik_Orani", color_discrete_sequence=PALETTE,
                         title="Arac Sahipligine Gore Kisi Bazli Yolculuk Oranlari")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("vehicle_count", "person_id")

_frag_tab_gelyol()
# ----------------------------------------------------------------- 1.5.2.8 AKTARMALI YOLCULUKLAR
@st.fragment
def _frag_tab_aktarma():
    with tab_aktarma:
        section_title("Sutun Eslestirme (Aktarma)", "🔗")
        render_mapping_grid(mapping, ["transfer"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_aktarma")
        st.divider()

        report_item("Grafik 70", "Aktarmali Yolculuk Oranlari")
        tstats = safe("Aktarma Istatistikleri", transfer_stats, con, TABLE, mapping)
        if tstats is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Aktarmali Yolculuk Orani", f"%{tstats['aktarmali_oran_pct']:.1f}")
            c2.metric("Ortalama Aktarma (tum yolculuklar)", f"{tstats['ortalama_aktarma_tum']:.2f}")
            c3.metric("Ortalama Aktarma (aktarmalilarda)", f"{tstats['ortalama_aktarma_aktarmalilarda']:.2f}")
        else:
            missing_note("transfer")

        st.divider()
        report_item("Tablo 28-29", "Turler Arasi Aktarma Sayilari ve Oranlari")
        tdist = safe("Aktarma Sayisi Dagilimi", transfer_count_distribution, con, TABLE, mapping)
        if tdist is not None:
            fig = px.bar(x=tdist.index, y=tdist.values, color_discrete_sequence=PALETTE,
                         labels={"x": "Aktarma Sayisi", "y": "%"}, title="Aktarma Sayisina Gore Yolculuk Dagilimi")
            st.plotly_chart(fig, width="stretch")
        else:
            missing_note("transfer")
        st.caption("Not: turler ARASI (orn. otobus->metro) aktarma cift-cift kirilimi icin ayri bir "
                   "'aktarma turu' sutunu gerekir - CSV'nizde varsa 'Sutun Gruplari' ile eslestirip "
                   "Sekme 1 (Serbest Pivot benzeri) analizlerde inceleyebilirsiniz.")

        st.divider()
        report_item("Tablo 30", "Aktarmali Yolculuklarda Sureler (Dk.)")
        if mapping.get("duration") and mapping.get("transfer"):
            transfer_dur = safe("Aktarmali Sure Karsilastirma", q_transfer_duration_comparison, con, TABLE,
                                 mapping["transfer"], mapping["duration"])
            if transfer_dur is not None:
                fig = px.bar(transfer_dur, x="Grup", y="Ortalama Sure (dk)",
                             color_discrete_sequence=PALETTE, title="Aktarmali vs Aktarmasiz Yolculuklarda Ortalama Sure")
                st.plotly_chart(fig, width="stretch")
        else:
            missing_note("duration", "transfer")

_frag_tab_aktarma()
# ----------------------------------------------------------------- 1.5.2.9 YOLCULUK MALIYETLERI
@st.fragment
def _frag_tab_maliyet():
    with tab_maliyet:
        section_title("Sutun Eslestirme (Maliyet)", "🔗")
        render_mapping_grid(mapping, ["cost", "mode"], active_columns, rankings,
                            key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_maliyet")
        st.divider()

        report_item("Grafik 71", "Toplu Tasima Maliyetleri (TL)")
        cost_col = mapping.get("cost")
        if cost_col:
            cost_summary = safe("Maliyet Ozeti", numeric_summary_sql, con, TABLE, cost_col)
            if cost_summary is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Ortalama Maliyet", f"{cost_summary['ortalama']:,.1f}")
                c2.metric("Medyan Maliyet", f"{cost_summary['medyan']:,.1f}")
                c3.metric("Std. Sapma", f"{cost_summary['std']:,.1f}")
                cost_hist = safe("Maliyet Histogrami", q_numeric_values, con, TABLE, cost_col, True)
                if cost_hist is not None:
                    fig = px.histogram(cost_hist.rename(columns={"val": "maliyet"}), x="maliyet", nbins=40,
                                       color_discrete_sequence=PALETTE, title="Yolculuk Maliyeti Dagilimi (Tam Veri)")
                    st.plotly_chart(fig, width="stretch")
                cost_by_mode = safe("Maliyet x Mod", numeric_by_group_sql_cached, con, TABLE,
                                     mapping.get("mode"), cost_col, "Ulasim Turu")
                if cost_by_mode is not None:
                    st.markdown("**Ulasim Turune Gore Ortalama Maliyet**")
                    fig = px.bar(cost_by_mode, x="Ulasim Turu", y="Ortalama", color_discrete_sequence=PALETTE)
                    st.plotly_chart(fig, width="stretch")
            else:
                missing_note("cost")
        else:
            missing_note("cost")

_frag_tab_maliyet()
# ----------------------------------------------------------------- 1.5.2.10 OZEL ARAC YOLCULUK ISTATISTIKLERI
@st.fragment
def _frag_tab_ozelarac():
    with tab_ozelarac:
        demographic_scope_warning(ACTIVE_TABLE)
        section_title("Sutun Eslestirme (Ozel Arac)", "🔗")
        render_mapping_grid(mapping, ["vehicle_occupancy", "driver_passenger", "trip_park_type"],
                            active_columns, rankings, key_ns=f"{ds.db_path}_{ACTIVE_TABLE}_ozelarac")
        st.divider()

        ozel_ref = bm.UAP2040_REFERENCE.get("ozel_arac_yolculuk", {})

        report_item("Tablo 31", "Ozel Arac Yolculuk Istatistikleri - Aractaki Kisi Sayisi")
        occ_col = mapping.get("vehicle_occupancy")
        if occ_col:
            occ_summary = safe("Arac Icindeki Kisi Sayisi", numeric_summary_sql, con, TABLE, occ_col)
            if occ_summary is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Ortalama Kisi Sayisi (sofor dahil)", f"{occ_summary['ortalama']:.2f}",
                          help=f"Rapor referansi (Izmir): {ozel_ref.get('ortalama_yolcu_sayisi_soforl_dahil', '-')}")
                c2.metric("Medyan", f"{occ_summary['medyan']:.1f}")
                c3.metric("Std. Sapma", f"{occ_summary['std']:.2f}",
                          help=f"Rapor referansi (Izmir): {ozel_ref.get('std', '-')}")
                occ_hist = safe("Kisi Sayisi Histogrami", q_numeric_values, con, TABLE, occ_col)
                if occ_hist is not None:
                    fig = px.histogram(occ_hist.rename(columns={"val": "kisi_sayisi"}), x="kisi_sayisi",
                                       nbins=15, color_discrete_sequence=PALETTE,
                                       title="Aractaki Kisi Sayisinin Dagilimi (Tam Veri)")
                    st.plotly_chart(fig, width="stretch")
            else:
                missing_note("vehicle_occupancy")
        else:
            missing_note("vehicle_occupancy")

        st.divider()
        report_item("Tablo 31", "Surucu / Yolcu Durumu")
        dp_dist = safe("Surucu/Yolcu Dagilimi", category_distribution, con, TABLE, mapping, "driver_passenger",
                       group_siblings=col_group_siblings)
        if dp_dist is not None:
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = px.pie(values=dp_dist.values, names=dp_dist.index, color_discrete_sequence=PALETTE,
                             title="Surucu / Yolcu Dagilimi")
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.caption(
                    f"Rapor referansi (Izmir): Sofor %{ozel_ref.get('sofor_pct', '-')}, "
                    f"Yolcu %{ozel_ref.get('yolcu_pct', '-')}. Sizin verinizdeki kod degerlerinin "
                    "hangisinin 'sofor', hangisinin 'yolcu' oldugunu bilmiyoruz - kesin etiket icin "
                    "yukaridaki 'Kod Cozme' panelinden bu sutuna etiket tanimlayabilirsiniz."
                )
                st.dataframe(dp_dist.rename("Yuzde (%)").to_frame(), width="stretch")
        else:
            missing_note("driver_passenger")

        st.divider()
        report_item("Tablo 31", "Yolculuk Sirasinda Otopark Turu")
        park_dist = safe("Otopark Turu Dagilimi", category_distribution, con, TABLE, mapping, "trip_park_type",
                         group_siblings=col_group_siblings)
        if park_dist is not None:
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = px.bar(x=park_dist.index.astype(str), y=park_dist.values, color_discrete_sequence=PALETTE,
                             labels={"x": "Otopark Turu (kod)", "y": "%"}, title="Yolculuk Sirasinda Otopark Turu")
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.caption(
                    f"Rapor referansi (Izmir): Ucretli %{ozel_ref.get('otopark_ucretli_pct', '-')}, "
                    f"Ucretsiz yol kenari %{ozel_ref.get('otopark_ucretsiz_yol_kenari_pct', '-')}, "
                    f"Ucretsiz isyeri/AVM %{ozel_ref.get('otopark_ucretsiz_isyeri_avm_pct', '-')}. "
                    "Kod->etiket eslemesi icin 'Kod Cozme' panelini kullanabilirsiniz."
                )
                st.dataframe(park_dist.rename("Yuzde (%)").to_frame(), width="stretch")
        else:
            missing_note("trip_park_type")

_frag_tab_ozelarac()
# ----------------------------------------------------------------- ILERI ANALIZLER
# UAP 2040 raporunun numarali Tablo/Grafik yapisinin DISINDA - benzer acik
# kaynak proje/toolkit'lerden (PSRC Household Travel Survey Data Explorer,
# ActivityViz, flowmap.blue, ABD NHTS Data Explorer) esinlenen EK analizler.
# Ana rapor-esli sekmeleri DEGISTIRMEZ, SADECE ek gorunumler ekler.
@st.fragment
def _frag_tab_ileri():
    with tab_ileri:
        section_title("Ileri Analizler", "🧭")
        st.caption(
            "Bu sekme, benzer acik kaynak anket/ulasim analiz projelerinden "
            "esinlenen EK gorunumleri toplar - UAP 2040 raporunun numarali "
            "Tablo/Grafik yapisinin disindadir, diger sekmelerdeki ana "
            "analizlerinizi degistirmez."
        )

        # ---------------------------------------------------- Ek 1: Guven araligi
        st.divider()
        report_item("Ek 1", "Guven Araligi / Orneklem Hatasi (Margin of Error)")
        st.caption(
            "Resmi anket toolkit'lerinin (orn. ABD Ulusal Hanehalki Ulasim "
            "Anketi - NHTS - Veri Kesif Araci, PSRC Household Travel Survey "
            "Data Explorer) standart pratigi: tek bir 'ortalama X' ya da "
            "'yuzde Y' sayisi tek basina yaniltici olabilir - kucuk orneklemli "
            "kirilimlarda gercek deger daha genis bir aralikta olabilir. "
            "Asagida secilen alan icin %95 guven araligi (yaklasik hata payi) "
            "hesaplanir. Formul: buyuk-ornek normal yaklasimi (agirlik "
            "eslenmisse Kish etkin ornek buyuklugu kullanilir)."
        )
        ci_numeric_options = {"Yas": "age", "Hanehalki Geliri": "income", "Yolculuk Suresi (dk)": "duration"}
        ci_cat_options = {"Cinsiyet": "gender", "Ulasim Turu": "mode", "Yolculuk Amaci": "purpose",
                           "Egitim Durumu": "education", "Calisma Durumu": "employment"}
        ci_numeric_avail = {k: v for k, v in ci_numeric_options.items() if mapping.get(v)}
        ci_cat_avail = {k: v for k, v in ci_cat_options.items() if mapping.get(v)}
        if not ci_numeric_avail and not ci_cat_avail:
            missing_note("age", "income", "duration", "gender", "mode", "purpose")
        else:
            ci_all_options = {f"{k} (ortalama)": ("mean", v) for k, v in ci_numeric_avail.items()}
            ci_all_options.update({f"{k} (oran)": ("prop", v) for k, v in ci_cat_avail.items()})
            ci_label = st.selectbox("Hangi alan icin guven araligi hesaplansin?", list(ci_all_options.keys()),
                                     key=f"{ds.db_path}_{ACTIVE_TABLE}_ci_pick")
            ci_kind, ci_concept = ci_all_options[ci_label]
            if ci_kind == "mean":
                ci_res = safe("Guven Araligi (ortalama)", confidence_interval_mean, con, TABLE, mapping, ci_concept)
                if ci_res:
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Ortalama", f"{ci_res['ortalama']:.2f}")
                    cc2.metric("%95 Guven Araligi", f"± {ci_res['moe']:.2f}")
                    cc3.metric("Etkin Ornek Buyuklugu", f"{ci_res['n_eff']:,.0f}")
                    st.caption(f"Gercek deger, %95 olasilikla [{ci_res['alt']:.2f}, {ci_res['ust']:.2f}] "
                               f"araliginda (ham gozlem sayisi: {ci_res['n']:,}).")
                    if ci_res["n_eff"] < 30:
                        st.warning("⚠️ Etkin ornek buyuklugu 30'un altinda - bu araligi TEMKINLI yorumlayin.")
                else:
                    missing_note(ci_concept)
            else:
                ci_res = safe("Guven Araligi (oran)", confidence_interval_proportion, con, TABLE, mapping, ci_concept)
                if ci_res is not None and not ci_res.empty:
                    fig = go.Figure(go.Bar(
                        x=ci_res["Kategori"].astype(str), y=ci_res["Oran (%)"],
                        error_y=dict(type="data", array=ci_res["Guven Araligi (+/- puan)"]),
                        marker_color=PALETTE[0],
                    ))
                    fig.update_layout(title=f"{ci_label} — %95 Guven Araligiyla", yaxis_title="%")
                    st.plotly_chart(fig, width="stretch")
                    n_eff_val = ci_res["Etkin Ornek (n)"].iloc[0]
                    st.caption(f"Etkin ornek buyuklugu: {n_eff_val:,.0f}" +
                               ("  ⚠️ 30'un altinda, temkinli yorumlayin." if n_eff_val < 30 else ""))
                    with st.expander("Tablo olarak gor"):
                        st.dataframe(ci_res, width="stretch")
                else:
                    missing_note(ci_concept)

        # ---------------------------------------------------- Ek 2: Cografi akis haritasi
        st.divider()
        report_item("Ek 2", "Cografi Akis Haritasi (Ilceler Arasi Yolculuk Yogunlugu)")
        st.caption(
            "flowmap.blue gibi acik kaynak akis haritasi araclarindan "
            "esinlenilmistir - ilceler arasindaki yolculuk hacmini, ilce "
            "merkezlerini birlestiren cizgilerle (kalinlik = hacim) gosterir. "
            "Cizgiler YAKLASIK ilce merkezleri arasindadir, gercek yol "
            "guzergahini TEMSIL ETMEZ."
        )
        odm = safe("OD Matrisi (akis haritasi)", od_matrix, con, TABLE, mapping, 20)
        if odm is not None and not odm.empty:
            odm2 = odm.reset_index()
            odm2 = odm2.rename(columns={odm2.columns[0]: "Baslangic"})
            long_od = odm2.melt(id_vars="Baslangic", var_name="Varis", value_name="n")
            long_od = long_od[(long_od["n"] > 0) & (long_od["Baslangic"] != long_od["Varis"])]
            long_od = long_od.sort_values("n", ascending=False).head(40)
            if long_od.empty:
                st.info("Ilceler arasi (kendi disinda) yolculuk kaydi bulunamadi.")
            else:
                cents = geo.district_centroids()
                long_od = long_od.copy()
                long_od["c1"] = long_od["Baslangic"].astype(str).apply(lambda n: cents.get(geo.normalize_tr(n)))
                long_od["c2"] = long_od["Varis"].astype(str).apply(lambda n: cents.get(geo.normalize_tr(n)))
                plot_od = long_od.dropna(subset=["c1", "c2"])
                if not plot_od.empty:
                    max_n = plot_od["n"].max()
                    fig = go.Figure()
                    for _, r in plot_od.iterrows():
                        w = 1 + 7 * (r["n"] / max_n)
                        fig.add_trace(go.Scattergeo(
                            lat=[r["c1"][0], r["c2"][0]], lon=[r["c1"][1], r["c2"][1]],
                            mode="lines", line=dict(width=w, color="rgba(200,30,30,0.45)"),
                            hoverinfo="text", text=f"{r['Baslangic']} → {r['Varis']}: {r['n']:,.0f}",
                            showlegend=False,
                        ))
                    pts = pd.concat([
                        plot_od[["Baslangic", "c1"]].rename(columns={"Baslangic": "ad", "c1": "c"}),
                        plot_od[["Varis", "c2"]].rename(columns={"Varis": "ad", "c2": "c"}),
                    ]).drop_duplicates("ad")
                    fig.add_trace(go.Scattergeo(
                        lat=[c[0] for c in pts["c"]], lon=[c[1] for c in pts["c"]],
                        mode="markers+text", text=pts["ad"], textposition="top center",
                        marker=dict(size=7, color="#1f4e8c"), showlegend=False,
                    ))
                    fig.update_geos(fitbounds="locations", visible=True, resolution=50,
                                     showcountries=True, countrycolor="lightgray")
                    fig.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=0),
                                       title="En Yogun 40 Ilceler Arasi Yolculuk Akisi")
                    st.plotly_chart(fig, width="stretch")
                    skipped = len(long_od) - len(plot_od)
                    if skipped:
                        st.caption(f"ℹ️ {skipped} ilce cifti, ilce merkezi harita verisinde bulunamadigi "
                                   "icin gosterilemedi.")
                else:
                    st.info("Ilce merkezleri harita verisinde eslenemedigi icin akis haritasi cizilemedi.")
        else:
            missing_note("district", "district_dest")

        # ---------------------------------------------------- Ek 3: Amac -> Mod sunburst
        st.divider()
        report_item("Ek 3", "Amactan Ulasim Turune Akis Diyagrami (Sunburst)")
        st.caption(
            "ActivityViz gibi projelerdeki hiyerarsik mod-payi gorsellestirmesinden "
            "esinlenilmistir - ic halka yolculuk AMACINI, dis halka o amacla "
            "yapilan yolculuklarin ULASIM TURUNU gosterir; dilim buyuklugu "
            "yolculuk sayisiyla orantilidir. Amac x Mod Isi Haritasi (bkz. "
            "'Amac & Ulasim Turu' sekmesi) ile AYNI veriye dayanir, farkli bir "
            "gorsel sunar."
        )
        if an.has(mapping, "purpose", "mode"):
            raw_ct = safe("Amac x Mod (ham sayilar)", crosstab_sql_cached, con, TABLE,
                           mapping["purpose"], mapping["mode"], None, 12, 10, mapping.get("weight"))
            if raw_ct is not None and not raw_ct.empty:
                long_ct = raw_ct.reset_index()
                long_ct = long_ct.rename(columns={long_ct.columns[0]: "Amac"})
                long_ct = long_ct.melt(id_vars="Amac", var_name="Mod", value_name="n")
                long_ct = long_ct[long_ct["n"] > 0]
                if long_ct.empty:
                    missing_note("purpose", "mode")
                else:
                    fig = px.sunburst(long_ct, path=["Amac", "Mod"], values="n",
                                       color_discrete_sequence=PALETTE,
                                       title="Yolculuk Amacindan Ulasim Turune Akis")
                    st.plotly_chart(fig, width="stretch")
            else:
                missing_note("purpose", "mode")
        else:
            missing_note("purpose", "mode")

        # ---------------------------------------------------- Ek 4: Saat x demografi
        st.divider()
        report_item("Ek 4", "Saatlik Dagilim x Demografi (Kimler Ne Zaman Yolculuk Yapiyor?)")
        st.caption(
            "ActivityViz'deki 'zaman kullanimi' analizinden esinlenilmistir - "
            "secilen demografik grubun alt kategorilerinin GUNUN HANGI "
            "SAATLERINDE yolculuk yaptigini karsilastirir. Her kategori KENDI "
            "icinde %100'e normalize edilir - yani grubun BUYUKLUGU degil, "
            "gunun hangi saatlerine YAYILDIGI (sekli/deseni) karsilastirilir."
        )
        hg_options = {"Cinsiyet": "gender", "Egitim Durumu": "education",
                      "Calisma Durumu": "employment", "Ogrencilik Durumu": "student"}
        hg_avail = {k: v for k, v in hg_options.items() if mapping.get(v)}
        if hg_avail and mapping.get("start_time"):
            hg_label = st.selectbox("Hangi gruba gore karsilastirilsin?", list(hg_avail.keys()),
                                     key=f"{ds.db_path}_{ACTIVE_TABLE}_hg_pick")
            hmap = safe("Saatlik Dagilim x Grup", hourly_distribution_by_group, con, TABLE, mapping,
                        hg_avail[hg_label], "start_time")
            if hmap is not None and not hmap.empty:
                fig = px.imshow(hmap, aspect="auto", color_continuous_scale="Blues",
                                 labels=dict(x="Saat", y=hg_label, color="%"),
                                 title=f"{hg_label} Grubuna Gore Saatlik Yolculuk Dagilimi (%, kendi icinde)")
                st.plotly_chart(fig, width="stretch")
            else:
                missing_note(hg_avail[hg_label], "start_time")
        else:
            missing_note("start_time", "gender")

        # ---------------------------------------------------- Ek 5: Ilce profili radar
        st.divider()
        report_item("Ek 5", "Ilce Profili Karsilastirmasi (Radar)")
        st.caption(
            "Secilen ilceleri, birden fazla gostergede (hanehalki buyuklugu, "
            "yas, gelir, arac sahipligi, hareketlilik orani) AYNI ANDA "
            "karsilastirir. Gostergeler karsilastirilabilir olsun diye "
            "SECILEN ilceler arasindaki en dusuk-en yuksek araligina gore "
            "0-100 arasina olceklenir - MUTLAK degerler icin tablonun "
            "altindaki 'Ham degerler' bolumune ya da ilgili sekmelere "
            "(Nufus, Gelir vb.) bakin."
        )
        radar_metrics = {}
        if mapping.get("district"):
            rm1 = safe("Radar-Hanehalki", household_size_by_district, con, TABLE, mapping)
            if rm1 is not None and not rm1.empty:
                radar_metrics["Hanehalki Buyuklugu"] = rm1.set_index("Ilce")["Ortalama Hanehalki Buyuklugu"]
            rm2 = safe("Radar-Yas", age_by_district, con, TABLE, mapping)
            if rm2 is not None and not rm2.empty:
                radar_metrics["Ortalama Yas"] = rm2.set_index("Ilce")["Ortalama Yas"]
            rm3 = safe("Radar-Gelir", income_by_district, con, TABLE, mapping)
            if rm3 is not None and not rm3.empty:
                radar_metrics["Ortalama Gelir"] = rm3.set_index("Ilce")["Ortalama Gelir"]
            rm4 = safe("Radar-Arac", vehicle_ownership_rate_by_district, con, TABLE, mapping)
            if rm4 is not None and not rm4.empty:
                radar_metrics["Arac Sahiplik Orani"] = rm4.set_index("Ilce")["Arac_Sahip_Yuzde"]
            rm5 = safe("Radar-Hareketlilik", mobility_rate_by_district, con, TABLE, mapping)
            if rm5 is not None and not rm5.empty:
                radar_metrics["Hareketlilik Orani"] = rm5.set_index("Ilce")["Hareketlilik_Orani"]
        if len(radar_metrics) >= 2:
            profile_df = pd.DataFrame(radar_metrics)
            profile_df.index = profile_df.index.astype(str)
            all_districts = sorted(profile_df.dropna(how="all").index.tolist())
            if len(all_districts) >= 2:
                radar_picked = st.multiselect("Karsilastirilacak ilceler (2-5 onerilir)", all_districts,
                                               default=all_districts[:min(3, len(all_districts))],
                                               key=f"{ds.db_path}_{ACTIVE_TABLE}_radar_pick")
                if len(radar_picked) >= 2:
                    sub = profile_df.reindex(radar_picked)
                    scaled = (sub - sub.min()) / (sub.max() - sub.min()).replace(0, 1) * 100
                    fig = go.Figure()
                    for idx in sub.index:
                        vals = scaled.loc[idx].fillna(0).tolist()
                        fig.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]], theta=list(scaled.columns) + [scaled.columns[0]],
                            fill="toself", name=str(idx),
                        ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                       title="Ilce Profili Karsilastirmasi (0-100 olcekli)")
                    st.plotly_chart(fig, width="stretch")
                    with st.expander("Ham degerler (olceklenmemis)"):
                        st.dataframe(sub, width="stretch")
                else:
                    st.info("Karsilastirmak icin en az 2 ilce secin.")
            else:
                st.info("Radar karsilastirmasi icin yeterli ilce verisi yok.")
        else:
            missing_note("district")

        # ---------------------------------------------------- Ek 6: Mahalle/Zon bazli ozet
        st.divider()
        report_item("Ek 6", "Mahalle / Zon Bazli Ozet Tablosu")
        st.caption(
            "'Zon/Mahalle' kavrami eslenmis olsa da, ilce haritasinin aksine, mahalle/TAZ SINIRLARI icin "
            "guvenilir bir cografi veri kaynagimiz yok (Turkiye capinda ~50 bin mahallenin sinirlarini "
            "iceren, dogrulugundan emin olabilecegimiz ucretsiz bir kaynak bulunmuyor) - bu yuzden GERCEK "
            "OLMAYAN/tahmini bir mahalle haritasi cizmek yerine (ActivityViz/PSRC gibi projelerdeki zone-level "
            "tablo gorunumune benzer), sayisal bir OZET TABLO sunuyoruz."
        )
        if mapping.get("zone"):
            zone_metrics = {}
            zn_count = safe("Zon - Anket Sayisi", raw_count_sql, con, TABLE, mapping["zone"], 500)
            if zn_count is not None and not zn_count.empty:
                zone_metrics["Anket/Kayit Sayisi"] = zn_count
            if mapping.get("age"):
                zn_age = safe("Zon - Yas", numeric_by_group_sql_cached, con, TABLE, mapping["zone"],
                               mapping["age"], "Zon")
                if zn_age is not None and not zn_age.empty:
                    zone_metrics["Ortalama Yas"] = zn_age.set_index("Zon")["Ortalama"]
            if mapping.get("income"):
                zn_inc = safe("Zon - Gelir", numeric_by_group_sql_cached, con, TABLE, mapping["zone"],
                               mapping["income"], "Zon")
                if zn_inc is not None and not zn_inc.empty:
                    zone_metrics["Ortalama Gelir"] = zn_inc.set_index("Zon")["Ortalama"]
            if zone_metrics:
                zone_df = pd.DataFrame(zone_metrics)
                zone_df.index.name = "Mahalle/Zon"
                zone_df = zone_df.sort_values(zone_df.columns[0], ascending=False)
                st.dataframe(zone_df, width="stretch")
                st.caption(f"Toplam {len(zone_df):,} farkli mahalle/zon degeri bulundu. "
                           "Kucuk orneklemli mahalleleri yorumlarken temkinli olun.")
            else:
                missing_note("zone")
        else:
            missing_note("zone")

_frag_tab_ileri()
# ----------------------------------------------------------------- KARSILASTIRMA
# YENI, TAMAMEN AYRI/EK bir sekme: ActivityViz'in coklu-senaryo destegi, MTC
# BATS Dashboard'unun 2019-vs-2023 karsilastirmasi ve summarizeNHTS'in cok-
# yilli rapor araci gibi projelerdeki "iki veri setini yan yana karsilastir"
# ihtiyacindan esinlenilmistir. DOGRULUK/GUVENLIK ICIN KRITIK: bu sekme
# TAMAMEN KENDI baglantisini (con_cmp), kendi tablosunu ve kendi eslestirme
# sozlugunu kullanir - yukaridaki ana analizin `con`, `TABLE`, `mapping`
# degiskenlerine ASLA DOKUNMAZ, bu yuzden mevcut hicbir sekmedeki/grafikteki
# deger bu ekleme yuzunden DEGISEMEZ.
@st.fragment
def _frag_tab_compare():
    with tab_compare:
        section_title("Iki Veri Setini Karsilastir", "🔄")
        st.caption(
            "Su an yuklu olan verinizi (yukaridaki tum sekmelerde kullanilan), BASKA bir CSV dosyasiyla "
            "(orn. gecen yilin anketi, baska bir sehir) yan yana karsilastirir. Bu sekme TAMAMEN AYRI "
            "calisir - yukaridaki hicbir sekmeyi/grafigi/hesabi ETKILEMEZ ya da DEGISTIRMEZ."
        )
        cmp_path = st.text_input(
            "Karsilastirilacak ikinci CSV dosyasinin tam yolu",
            key="cmp_path_input", placeholder=r"C:\Users\...\baska_sehir_veya_yil.csv",
        )
        if not cmp_path:
            st.info("Karsilastirmaya baslamak icin yukariya ikinci bir CSV dosya yolu girin.")
        elif not os.path.exists(cmp_path):
            st.error("Bu yolda bir dosya bulunamadi. Yolu kontrol edin.")
        else:
            cmp_result = safe("Karsilastirma verisi yukleme", data_io.get_or_build, cmp_path)
            if cmp_result is None:
                st.error("Ikinci dosya okunamadi.")
            else:
                con_cmp = duckdb.connect()
                con_cmp.execute(f"ATTACH '{cmp_result.db_path}' AS src_cmp (READ_ONLY)")
                groups_cmp = cg.detect_all_groups(cmp_result.columns)
                singles_cmp = cg.cluster_single_blocks(groups_cmp["single"])
                plain_cmp = groups_cmp.get("plain", cmp_result.columns) or cmp_result.columns

                if singles_cmp:
                    # Kisi/hane bazli en alakali bloğu sec (bkz. _pick_smart_person_block -
                    # ayni "en cok bilinen kavramla eslesen blok" mantigi, mevcut ana akista
                    # zaten test edilmis/dogrulanmis).
                    person_block_cmp = _pick_smart_person_block(singles_cmp)
                    id_candidates_cmp = [c for c in plain_cmp if any(
                        k in c.lower() for k in ["hane", "id", "kumeno", "anketid"])] or list(plain_cmp[:1])
                    best_id_cmp, _ = safe("Kimlik tekillik (karsilastirma)", cg.suggest_unique_key,
                                           con_cmp, "src_cmp.raw_data", tuple(id_candidates_cmp[:5]),
                                           cmp_result.total_rows) or (id_candidates_cmp[:1], 0.0)
                    sql_cmp, _ = cg.build_melt_view_sql("src_cmp.raw_data", person_block_cmp,
                                                         list(best_id_cmp), "kisi_cmp",
                                                         index_col_name="kisi_no", extra_cols=plain_cmp)
                    built_cmp = safe("Karsilastirma kisi tablosu", con_cmp.execute, sql_cmp)
                    table_cmp = "kisi_cmp" if built_cmp is not None else None
                    if table_cmp:
                        # DOGRULUK ICIN KRITIK (ana akistaki AYNI uyariya bkz., ~satir 810):
                        # "kisi_no" HER HANEDE 1'den baslayan YEREL bir sira numarasidir,
                        # TEK BASINA kesinlikle tekil bir kisi kimligi DEGILDIR. Bunu
                        # duzeltmezsek, auto_guess_mapping "kisi_no"yu yanlislikla
                        # person_id sanabilir ve dedup/ortalama hesaplari birkac
                        # HANEYE degil birkac YEREL-INDEKS-DEGERINE gore gruplanip
                        # tamamen yanlis (olculup dogrulandi: 15,610 kisi yerine
                        # ~sadece 20 "kisi" gibi davranip ortalama yasi 34'ten 15'e
                        # dusurur) sonuc uretir. Hane kimligi + kisi_no'yu birlestiren
                        # GERCEKTEN tekil bir _kisi_id eklenir (ana akistaki cozumle
                        # BIREBIR ayni mantik).
                        kisi_id_expr_cmp = " || '_' || ".join(
                            [f"CAST({an.ident(c)} AS VARCHAR)" for c in best_id_cmp] + ["CAST(kisi_no AS VARCHAR)"]
                        )
                        safe("Karsilastirma kisi ID ekleme", con_cmp.execute, f"""
                            CREATE OR REPLACE TABLE kisi_cmp AS
                            SELECT *, {kisi_id_expr_cmp} AS {an.ident('_kisi_id')} FROM kisi_cmp
                        """)
                else:
                    # Zaten tekrarlayan grup yok (dosya duz/uzun formatta). NOT: ATTACH
                    # edilmis semaya nitelikli ("src_cmp.raw_data") adla DEGIL, once
                    # niteliksiz bir VIEW'a takma adla erisilir - an.ident() gibi
                    # tirnaklamalar noktali/nitelikli tablo adlarini TEK bir tanimlayici
                    # sanip hataya yol acar (bu, sadece bu yerel karsilastirma
                    # akisinda dogrulanan bir sinirlamadir, ana akisi etkilemez).
                    con_cmp.execute("CREATE OR REPLACE VIEW raw_cmp AS SELECT * FROM src_cmp.raw_data")
                    table_cmp = "raw_cmp"

                if table_cmp is None:
                    st.error("Ikinci dosyada kisi/hane bazli bir tablo olusturulamadi.")
                else:
                    cols_cmp = table_columns(con_cmp, table_cmp)
                    mapping_cmp = mapping_mod.auto_guess_mapping(cols_cmp)
                    if "_kisi_id" in cols_cmp:
                        # bkz. yukaridaki "_kisi_id ekleme" notu - auto_guess_mapping'in
                        # yanlislikla YEREL "kisi_no"yu person_id sanmasini ONLER.
                        mapping_cmp["person_id"] = "_kisi_id"
                    n_mapped_cmp = sum(1 for v in mapping_cmp.values() if v)
                    st.caption(f"Ikinci dosyada otomatik eslesen kavram: {n_mapped_cmp}/{len(mapping_cmp)} "
                               f"— {cmp_result.total_rows:,} satir.")

                    def _cmp_metric(label, fn, *args):
                        v1 = safe(f"Karsilastirma (A): {label}", fn, con, TABLE, mapping, *args)
                        v2 = safe(f"Karsilastirma (B): {label}", fn, con_cmp, table_cmp, mapping_cmp, *args)
                        return v1, v2

                    rows_cmp = []
                    hh1, hh2 = _cmp_metric("Hanehalki Buyuklugu", an.household_size_stats)
                    if hh1 is not None or hh2 is not None:
                        rows_cmp.append(("Ort. Hanehalki Buyuklugu",
                                          float(hh1["Ortalama"].iloc[0]) if hh1 is not None else None,
                                          float(hh2["Ortalama"].iloc[0]) if hh2 is not None else None))
                    age1, age2 = _cmp_metric("Ortalama Yas", an.avg_age)
                    if age1 is not None or age2 is not None:
                        rows_cmp.append(("Ortalama Yas", age1, age2))
                    inc1, inc2 = _cmp_metric("Ortalama Gelir", an.income_summary)
                    if inc1 is not None or inc2 is not None:
                        rows_cmp.append(("Ort. Hanehalki Geliri",
                                          inc1["ortalama"] if inc1 is not None else None,
                                          inc2["ortalama"] if inc2 is not None else None))
                    gini1, gini2 = _cmp_metric("Gini", an.gini_and_lorenz)
                    if gini1 is not None or gini2 is not None:
                        rows_cmp.append(("Gini Katsayisi",
                                          gini1[0] if gini1 is not None else None,
                                          gini2[0] if gini2 is not None else None))

                    veh1 = safe("Karsilastirma (A): Arac", an.vehicle_ownership_distribution, con, TABLE, mapping)
                    veh2 = safe("Karsilastirma (B): Arac", an.vehicle_ownership_distribution, con_cmp, table_cmp, mapping_cmp)
                    if veh1 is not None or veh2 is not None:
                        def _pct_at_least_1(s):
                            if s is None:
                                return None
                            return float(sum(v for k, v in s.items() if str(k) not in ("0", "0.0")))
                        rows_cmp.append(("En Az 1 Arac Sahiplik Orani (%)",
                                          _pct_at_least_1(veh1), _pct_at_least_1(veh2)))

                    if rows_cmp:
                        label_a = os.path.basename(path)
                        label_b = os.path.basename(cmp_path)
                        cmp_df = pd.DataFrame(rows_cmp, columns=["Gosterge", label_a, label_b]).dropna(how="all",
                                                                                                          subset=[label_a, label_b])
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name=label_a, x=cmp_df["Gosterge"], y=cmp_df[label_a]))
                        fig.add_trace(go.Bar(name=label_b, x=cmp_df["Gosterge"], y=cmp_df[label_b]))
                        fig.update_layout(barmode="group", title="Iki Veri Setinin Karsilastirmasi",
                                          height=450)
                        st.plotly_chart(fig, width="stretch")
                        with st.expander("Tablo olarak gor"):
                            st.dataframe(cmp_df, width="stretch")

                        # ---- Istatistiksel anlamlilik testi (NHTS/PSRC tarzi) ----
                        # NOT: yukaridaki cmp_df/grafik BURADAN ETKILENMEZ - bu, SADECE
                        # EK bir yorum katmanidir (hangi farklar orneklem hatasindan
                        # farkli/gercek gorunuyor). Degerlerin kendisi degismez.
                        sig_rows = []
                        for label, concept in [("Ortalama Yas", "age"), ("Ort. Hanehalki Geliri", "income"),
                                                ("Ort. Hanehalki Buyuklugu", "household_size")]:
                            ci1 = safe(f"Anlamlilik (A): {label}", an.confidence_interval_mean, con, TABLE, mapping, concept)
                            ci2 = safe(f"Anlamlilik (B): {label}", an.confidence_interval_mean, con_cmp, table_cmp,
                                       mapping_cmp, concept)
                            if ci1 and ci2:
                                t = an.two_sample_z_test(ci1["ortalama"], ci1["se"], ci2["ortalama"], ci2["se"])
                                sig_rows.append((label, round(ci1["ortalama"], 2), round(ci2["ortalama"], 2),
                                                  round(t["fark"], 2), round(t["p_value"], 4),
                                                  "✅ Anlamli (p<0.05)" if t["anlamli_mi"] else "— Anlamli degil"))
                        if sig_rows:
                            sig_df = pd.DataFrame(sig_rows, columns=["Gosterge", label_a, label_b, "Fark",
                                                                      "p-degeri", "Sonuc"])
                            st.markdown("**Istatistiksel Anlamlilik Testi** (bu fark orneklem hatasindan mi kaynaklaniyor?)")
                            st.caption(
                                "Buyuk-ornek z-testi (NHTS/PSRC gibi resmi anket araclarinin standart yontemi): "
                                "p-degeri 0.05'ten KUCUKSE, iki deger arasindaki fark %95 guvenle GERCEK kabul "
                                "edilir (sadece orneklem rastgeleligi degildir). p-degeri buyukse, gorunen fark "
                                "istatistiksel olarak anlamli DEGILDIR - orneklem buyuklugu yetersiz olabilir."
                            )
                            st.dataframe(sig_df, width="stretch")
                        st.caption(
                            "ℹ️ Her iki taraf da KENDI dosyasindaki otomatik eslesen sutunlardan hesaplanir - "
                            "iki dosyanin sutun adlandirmasi farkli olsa bile (akilli/bulanik sutun tespiti "
                            "sayesinde) karsilastirma yapilabilir. Bir gosterge tek tarafta gorunuyorsa, o "
                            "kavram o dosyada eslesmemis demektir."
                        )
                        st.caption(
                            "⚠️ Farkli yillara ait anketleri karsilastirirken dikkat: bazi resmi anketlerde "
                            "(orn. NSW Household Travel Survey) yontem/kod tanimlari yildan yila DEGISEBILIR "
                            "(orn. pandemi sonrasi amac/mod kategorileri yeniden gruplanmis olabilir) - "
                            "buyuk bir sapma gorurseniz, once iki anketin kod sozlugunu (metadata) "
                            "karsilastirmanizi oneririz."
                        )
                    else:
                        st.warning("Karsilastirilabilir ortak bir gosterge bulunamadi - her iki dosyada da "
                                   "en az yas, gelir, hanehalki buyuklugu ya da arac sayisi gibi bir alan "
                                   "eslesmis olmali.")

_frag_tab_compare()
# ----------------------------------------------------------------- AI ICGORULERI
@st.fragment
def _frag_tab_ai():
    with tab_ai:
        # Bu sekme, Gelir/Cinsiyet-Yas/Arac/Yolculuk Oranlari/Amac-Mod/Saatlik
        # Dagilim sekmelerinde ZATEN hesaplanmis degerleri (gini, mod/amac
        # dagilimi, hane buyuklugu, hareketlilik orani, arac sahipligi, yas,
        # saatlik dagilim) ozetlemek icin kullanir. Fragment izolasyonu
        # sonrasinda diger sekmelerin degiskenlerine guvenilemeyecegi (her
        # sekme kendi kapsaminda calisir) icin BURADA BAGIMSIZ OLARAK yeniden
        # cagriliyor - hepsi zaten cq()/st.cache_data ile onbellekli oldugundan
        # bu, ek bir sorgu maliyeti getirmez (onbellekten doner), sadece bu
        # sekmenin kendi basina, guncel veriyle calismasini garanti eder.
        gl = safe("Gini/Lorenz (AI ozeti)", gini_and_lorenz, con, TABLE, mapping)
        mode_dist = safe("Ulasim Turu (AI ozeti)", mode_distribution, con, TABLE, mapping,
                          group_siblings=col_group_siblings)
        purpose_dist = safe("Yolculuk Amaci (AI ozeti)", purpose_distribution, con, TABLE, mapping,
                             group_siblings=col_group_siblings)
        hh_stats = safe("Hanehalki Buyuklugu (AI ozeti)", household_size_stats, con, TABLE, mapping)
        hh_district = safe("Ilce Bazinda Hanehalki Buyuklugu (AI ozeti)", household_size_by_district,
                            con, TABLE, mapping)
        rate = safe("Hareketlilik Orani (AI ozeti)", mobility_rate, con, TABLE, mapping)
        rate_district = safe("Ilce Bazinda Hareketlilik (AI ozeti)", mobility_rate_by_district,
                              con, TABLE, mapping)
        veh_dist = safe("Arac Sahipligi (AI ozeti)", vehicle_ownership_distribution, con, TABLE, mapping)
        age_district = safe("Ilce Bazinda Yas (AI ozeti)", age_by_district, con, TABLE, mapping)
        avgage = safe("Ortalama Yas (AI ozeti)", avg_age_fn, con, TABLE, mapping)
        gender_dist = safe("Cinsiyet Dagilimi (AI ozeti)", gender_distribution, con, TABLE, mapping)
        hourly_start = safe("Saatlik Dagilim (AI ozeti)", hourly_distribution, con, TABLE, mapping, "start_time")

        section_title("Otomatik Icgoru Ozeti (Yerel AI)", "🧠")
        narratives = []
        for label, fn, args in [
            ("Hane", ai_local.narrative_household_size, [hh_stats["Ortalama"].iloc[0] if hh_stats is not None else None]),
            ("Cinsiyet", ai_local.narrative_gender, [gender_dist]),
            ("Yas", ai_local.narrative_avg_age, [avgage]),
            ("Hareketlilik", ai_local.narrative_district_extremes,
             [rate_district, "Ilce", "Hareketlilik_Orani", "Hareketlilik orani"]),
            ("Zirve saat", ai_local.narrative_peak_hour, [hourly_start]),
        ]:
            n = safe(f"Icgoru: {label}", fn, *args)
            if n:
                narratives.append(n)
        narratives.extend(safe("Amac/Mod icgoru", ai_local.narrative_purpose_mode, purpose_dist, mode_dist) or [])

        if narratives:
            for n in narratives:
                st.markdown(f"- {n}")
        else:
            st.warning("Icgoru uretmek icin en az birkac sutun eslestirilmeli (cinsiyet, yas, "
                       "ilce, amac, mod gibi).")

        st.divider()
        section_title("Bolge / Ilce Profilleme (KMeans Kumeleme, otomatik k secimi)", "🧩")
        if mapping.get("district"):
            feature_candidates = {}
            if rate_district is not None and not rate_district.empty:
                feature_candidates["Hareketlilik_Orani"] = rate_district.set_index("Ilce")["Hareketlilik_Orani"]
            if hh_district is not None and not hh_district.empty:
                feature_candidates["Hane_Buyuklugu"] = hh_district.set_index("Ilce")["Ortalama Hanehalki Buyuklugu"]
            veh_district2 = safe("Arac (kumeleme icin)", vehicles_per_1000_by_district, con, TABLE, mapping)
            if veh_district2 is not None and not veh_district2.empty:
                feature_candidates["Arac_1000Kisi"] = veh_district2.set_index("Ilce")["1000_Kisiye_Dusen_Arac"]
            if mapping.get("income"):
                inc_by_district = safe("Gelir (kumeleme icin)", q_avg_numeric_by_district, con, TABLE,
                                        mapping["district"], mapping["income"])
                if inc_by_district is not None:
                    feature_candidates["Ortalama_Gelir"] = inc_by_district.set_index("Ilce")["geliri"]

            if len(feature_candidates) >= 2:
                wide = pd.DataFrame(feature_candidates).reset_index().rename(columns={"index": "Ilce"})
                feat_cols = [c for c in wide.columns if c != "Ilce"]
                auto = st.checkbox("Kume sayisini otomatik sec (silhouette skoru)", value=True)
                n_clusters = None
                if not auto:
                    n_clusters = st.slider("Kume sayisi", 2, min(8, max(2, len(wide) - 1)), 4)
                result = safe("Kumeleme", cluster_districts, wide, "Ilce", feat_cols, n_clusters=n_clusters)
                if result is not None:
                    clustered_df, profile_text, used_k = result
                    st.caption(f"Kullanilan kume sayisi: {used_k}" + (" (otomatik secildi)" if auto else ""))
                    fig = px.scatter(clustered_df, x=feat_cols[0], y=feat_cols[1] if len(feat_cols) > 1 else feat_cols[0],
                                     color=clustered_df["kume"].astype(str), text="Ilce",
                                     color_discrete_sequence=PALETTE, title="Ilce Kumeleri")
                    fig.update_traces(textposition="top center")
                    st.plotly_chart(fig, width="stretch")
                    st.markdown(profile_text)
                    with st.expander("Kumeleme detay tablosu"):
                        st.dataframe(clustered_df, width="stretch")
                else:
                    st.info("Kumeleme icin yeterli ilce/veri yok.")
            else:
                st.info("Kumeleme icin en az 2 sayisal ilce-bazli metrik gerekiyor "
                        "(hareketlilik, hane buyuklugu, arac sahipligi, gelir gibi).")
        else:
            missing_note("district")

        st.divider()
        section_title("Agir Analizler (isteğe bagli calistirilir)", "🐢")
        st.caption(
            "Anomali taramasi ve ozellik onemi analizi hesaplama yogun oldugu icin "
            "sayfa acilisinda OTOMATIK calismaz - butona basinca calisir. Sonuc bir "
            "kez hesaplandiktan sonra onbelleklenir (tekrar tekrar calismaz)."
        )

        prof2 = safe("Veri profili (AI icin)", data_profile, con, TABLE, active_rows, active_columns, active_dtypes)
        numeric_cols = (prof2["numeric_cols"] if prof2 else []) or [
            c for c in [mapping.get("income"), mapping.get("age"), mapping.get("duration")] if c]

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**🚨 Anomali / Veri Kalitesi Tespiti (IsolationForest)**")
            if len(numeric_cols) >= 2:
                if st.button("Anomali taramasini calistir"):
                    with st.spinner("Anomali taramasi calisiyor..."):
                        result = safe("Anomali tespiti", detect_row_anomalies_sql, con, TABLE, numeric_cols, active_rows)
                    if result is not None:
                        anomalies, used_n = result
                        scope_note = (f"(TAM VERI, {used_n:,} satir)" if used_n >= active_rows
                                      else f"(temsili ornek, {used_n:,}/{active_rows:,} satir)")
                        if not anomalies.empty:
                            st.warning(f"{len(anomalies):,} satirda olasi aykiri deger tespit edildi {scope_note}.")
                            st.dataframe(anomalies.head(50), width="stretch")
                        else:
                            st.success(f"Sayisal alanlarda belirgin aykiri deger tespit edilmedi {scope_note}.")
            else:
                st.info("Anomali tespiti icin en az 2 sayisal sutun gerekiyor.")

        with colB:
            st.markdown("**🌳 Ozellik Onemi (RandomForest) — Neyi Ne Etkiliyor?**")
            target_candidates = [c for c in [mapping.get("duration"), mapping.get("income"),
                                              mapping.get("mode"), mapping.get("purpose")] if c]
            if target_candidates:
                target = st.selectbox("Hedef degisken secin (ne etkileniyor?)", target_candidates,
                                       key=f"{ds.db_path}_{ACTIVE_TABLE}_fi_target")
                if st.button("Ozellik onemini hesapla"):
                    feature_pool = [c for c in active_columns if c != target][:20]
                    with st.spinner("Model egitiliyor..."):
                        fi = safe("Ozellik onemi", feature_importance_analysis, con, TABLE, target,
                                   feature_pool, active_rows)
                    if fi is not None:
                        importance, fi_text = fi
                        st.caption(fi_text)
                        fig = px.bar(importance, x="Onem Skoru", y="Degisken", orientation="h",
                                     color_discrete_sequence=PALETTE, title=f"'{target}' Degiskenini Etkileyen Faktorler")
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, width="stretch")
                    elif fi is None:
                        st.info("Bu hedef icin yeterli/uygun veri bulunamadi.")
            else:
                st.info("Sure, gelir, mod ya da amac gibi bir hedef sutun eslestirilmeli.")

        if mapping.get("district"):
            for metric_name, df_metric, col_name in [
                ("Hareketlilik orani", rate_district, "Hareketlilik_Orani"),
                ("Hanehalki buyuklugu", hh_district, "Ortalama Hanehalki Buyuklugu"),
            ]:
                if df_metric is not None and not df_metric.empty:
                    outliers = safe(f"Aykiri ilce: {metric_name}", ai_local.detect_outlier_districts,
                                     df_metric, col_name, "Ilce")
                    if outliers is not None and not outliers.empty:
                        st.markdown(f"**{metric_name} - istatistiksel aykiri ilceler (|z| >= 2):**")
                        st.dataframe(outliers, width="stretch")

        st.divider()
        section_title("Korelasyon Analizi (tam veri, DuckDB CORR())", "🔗")
        if len(numeric_cols) >= 2:
            corr_result = safe("Korelasyon", correlation_insights, con, TABLE, numeric_cols)
            if corr_result is not None:
                corr_table, corr_matrix = corr_result
                if not corr_table.empty:
                    st.dataframe(corr_table.style.format({"Korelasyon": "{:.3f}"}), width="stretch")
                if not corr_matrix.empty:
                    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                    title="Sayisal Degiskenler Korelasyon Matrisi (tam veri)")
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("Korelasyon analizi icin en az 2 sayisal sutun gerekiyor.")

        st.divider()
        section_title("Gercek LLM ile Derinlemesine Yorum (Opsiyonel)", "🌐")
        provider = ai_llm.available_provider()
        if provider:
            st.success(f"'{provider}' API anahtari bulundu.")
            if st.button("🔎 LLM ile yorumla"):
                summary = {
                    "toplam_satir": active_rows,
                    "hanehalki_buyuklugu_ortalama": float(hh_stats["Ortalama"].iloc[0]) if hh_stats is not None else None,
                    "cinsiyet_dagilimi": gender_dist.to_dict() if gender_dist is not None else None,
                    "ortalama_yas": avgage,
                    "hareketlilik_orani_genel": rate,
                    "yolculuk_amaci_dagilimi": purpose_dist.to_dict() if purpose_dist is not None else None,
                    "ulasim_turu_dagilimi": mode_dist.to_dict() if mode_dist is not None else None,
                    "uap2040_referans": bm.UAP2040_REFERENCE,
                }
                with st.spinner("LLM yaniti bekleniyor..."):
                    try:
                        text = ai_llm.generate_llm_commentary(summary)
                        st.markdown(text)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"LLM cagrisi basarisiz: {e}")
        else:
            st.info(
                "LLM destekli derinlemesine yorum icin ortam degiskeni olarak "
                "**ANTHROPIC_API_KEY** ya da **OPENAI_API_KEY** tanimlayin "
                "(PowerShell: `$env:ANTHROPIC_API_KEY=\"...\"`). Tanimli degilse "
                "dashboard yalnizca yukaridaki yerel AI motoruyla calisir."
            )

        st.divider()
        section_title("Yonetici Ozeti (indirilebilir)", "📝")
        report_lines = [f"# UAP 2040 Tarzi Analiz - Yonetici Ozeti\n",
                         f"Kaynak dosya: `{os.path.basename(path)}` — {active_rows:,} satir "
                         f"(tablo: {TABLE})\n"]
        report_lines += [f"- {n}" for n in narratives]
        report_md = "\n".join(report_lines)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.download_button("⬇️ Ozet raporu indir (Markdown)", data=report_md,
                               file_name="uap2040_analiz_ozeti.md", mime="text/markdown")
        with rc3:
            # DOGRULUK ICIN KRITIK: Excel'deki HER sayfa da (HTML raporundaki gibi)
            # sayfada ZATEN hesaplanmis AYNI degiskenlerden TAZE yazilir.
            excel_sheets: list[tuple[str, "pd.DataFrame | pd.Series | None"]] = [
                ("Ilce - Hane Buyuklugu", hh_district),
                ("Cinsiyet Dagilimi", gender_dist),
                ("Ilce - Ortalama Yas", age_district),
                ("Arac Sahipligi", veh_dist),
                ("Ilce - Hareketlilik", rate_district),
                ("Yolculuk Amaci", purpose_dist),
                ("Ulasim Turu", mode_dist),
                ("Saatlik Dagilim", hourly_start),
            ]
            excel_meta = {
                "Kaynak dosya": os.path.basename(path),
                "Satir sayisi": f"{active_rows:,}",
                "Aktif tablo": TABLE,
                "Olusturulma": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Not": "Her sayfadaki degerler, dashboard'daki ilgili sekmeyle BIREBIR aynidir.",
            }
            has_any_sheet = any(
                (obj is not None) and (not obj.empty if hasattr(obj, "empty") else True)
                for _, obj in excel_sheets
            )
            excel_bytes = safe("Excel raporu", build_excel_report_bytes, excel_meta, excel_sheets) if has_any_sheet else None
            st.download_button("⬇️ Excel raporu indir (.xlsx)", data=excel_bytes or b"",
                               file_name="uap2040_analiz_ozeti.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               disabled=not excel_bytes,
                               help="Her sekme ayri bir Excel sayfasi olarak indirilir.")
        with rc2:
            # DOGRULUK ICIN KRITIK: buradaki HER grafik, sayfada ZATEN gosterilmis
            # AYNI degiskenlerden (hh_district, gender_dist, ... - sekmelerde
            # kullanilanlarla BIREBIR ayni Series/DataFrame) TAZE cizilir - farkli
            # bir hesap yolu YOKTUR, bu yuzden rapordaki sayilar ekrandakilerle
            # HER ZAMAN eslesir.
            report_sections: list[tuple[str, go.Figure | None]] = []
            if hh_district is not None and not hh_district.empty:
                report_sections.append(("Ilce Bazinda Ortalama Hanehalki Buyuklugu",
                                         px.bar(hh_district, x="Ilce", y="Ortalama Hanehalki Buyuklugu",
                                                color_discrete_sequence=PALETTE)))
            if gender_dist is not None and not gender_dist.empty:
                report_sections.append(("Cinsiyet Dagilimi",
                                         px.pie(values=gender_dist.values, names=gender_dist.index.astype(str),
                                                color_discrete_sequence=PALETTE)))
            if age_district is not None and not age_district.empty:
                report_sections.append(("Ilcelere Gore Ortalama Yas",
                                         px.bar(age_district, x="Ilce", y="Ortalama Yas",
                                                color_discrete_sequence=PALETTE)))
            if gl is not None:
                gini_val, lorenz_df = gl
                fig_lorenz = go.Figure()
                fig_lorenz.add_trace(go.Scatter(x=lorenz_df["Nufus Payi"], y=lorenz_df["Gelir Payi"],
                                                mode="lines", name="Lorenz Egrisi", fill="tozeroy"))
                fig_lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Esitlik",
                                                line=dict(dash="dash", color="gray")))
                fig_lorenz.update_layout(title=f"Gelir Dagilimi (Gini={gini_val:.3f})")
                report_sections.append(("Gelir Dagilimi (Lorenz Egrisi)", fig_lorenz))
            if veh_dist is not None and not veh_dist.empty:
                report_sections.append(("Arac Sahipligi Dagilimi",
                                         px.bar(x=veh_dist.index.astype(str), y=veh_dist.values,
                                                labels={"x": "Arac Sayisi", "y": "%"},
                                                color_discrete_sequence=PALETTE)))
            if rate_district is not None and not rate_district.empty:
                report_sections.append(("Ilce Bazinda Hareketlilik Orani",
                                         px.bar(rate_district, x="Ilce", y="Hareketlilik_Orani",
                                                color_discrete_sequence=PALETTE)))
            if purpose_dist is not None and not purpose_dist.empty:
                report_sections.append(("Yolculuk Amaci Dagilimi",
                                         px.bar(x=purpose_dist.index.astype(str), y=purpose_dist.values,
                                                labels={"x": "Amac", "y": "%"}, color_discrete_sequence=PALETTE)))
            if mode_dist is not None and not mode_dist.empty:
                report_sections.append(("Ulasim Turu Dagilimi",
                                         px.bar(x=mode_dist.index.astype(str), y=mode_dist.values,
                                                labels={"x": "Ulasim Turu", "y": "%"}, color_discrete_sequence=PALETTE)))
            if hourly_start is not None and not hourly_start.empty:
                report_sections.append(("Yolculuklarin Saatlik Dagilimi",
                                         px.line(x=list(hourly_start.index), y=hourly_start.values,
                                                labels={"x": "Saat", "y": "%"})))

            report_meta = [
                f"Kaynak dosya: {os.path.basename(path)}",
                f"Satir sayisi: {active_rows:,} (tablo: {TABLE})",
                f"Olusturulma: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                "Bu HTML dosyasi tamamen cevrimdisi acilir (internet gerekmez). "
                "PDF'e cevirmek icin tarayicida ac, Yazdir > 'PDF olarak kaydet' secin.",
            ]
            html_report = build_visual_report_html("UAP 2040 Tarzi Analiz - Gorsel Rapor", report_meta, report_sections)
            st.download_button("⬇️ Gorsel rapor indir (HTML → PDF'e cevrilebilir)", data=html_report,
                               file_name="uap2040_gorsel_rapor.html", mime="text/html",
                               disabled=not report_sections,
                               help="Indirdikten sonra tarayicida acip Ctrl+P ile PDF olarak kaydedebilirsiniz.")

        if ERROR_LOG:
            st.divider()
            with st.expander(f"🪲 Bu sayfada olusan tum hatalar ({len(ERROR_LOG)})"):
                for label, err in ERROR_LOG:
                    st.code(f"[{label}]\n{err}", language="text")

_frag_tab_ai()
# ----------------------------------------------------------------- REFERANS
@st.fragment
def _frag_tab_ref():
    with tab_ref:
        section_title("UAP 2040 Rapor Referans Degerleri", "📖")
        st.info(
            "⚠️ Bu referans degerler **sadece Izmir** icin gecerlidir (Izmir Ulasim "
            "Ana Plani 2040 raporundan alinmistir). Baska bir sehrin verisini "
            "analiz ediyorsaniz (orn. Bursa), diger sekmelerdeki karsilastirma "
            "notlari ('UAP 2040 referansi ile benzer/farkli') sadece KABA bir "
            "olcek fikri vermek icindir - o sehrin kendi resmi degerleriyle "
            "(orn. TUIK) karsilastirma yapmak daha dogru olur."
        )
        st.markdown(bm.get_reference_text_block())
        st.divider()
        st.markdown("**Gelir Grubuna Gore Ortalama Yolculuk Sayilari (Rapor Tablo 25)**")
        ref_income = pd.DataFrame(bm.UAP2040_REFERENCE["gelir_grubu_ortalama_yolculuk"]).T
        st.dataframe(ref_income, width="stretch")
        st.divider()
        st.markdown("**Ozel Arac Yolculuk Istatistikleri (Rapor Tablo 31)**")
        st.json(bm.UAP2040_REFERENCE["ozel_arac_yolculuk"])
        st.caption("Kaynak: KSR-045-UAP2040_YENI_BILGILER_RAPORU_CILT 1_R01-17-104.pdf")

_frag_tab_ref()
# Tum sekmelerde yapilan eslestirme degisiklikleri, tek bir mapping sozlugu
# uzerinde YERINDE biriktirildi; bir sonraki calismada hatirlanmasi icin
# script sonunda TEK SEFERDE onbellege yaziliyor.
st.session_state[map_key] = mapping
mapped_total = sum(1 for v in mapping.values() if v)
st.sidebar.caption(f"🔗 {mapped_total}/{len(CONCEPTS)} kavram eslestirildi · Toplam sure: {time.time() - t_start:.2f} sn")
