# -*- coding: utf-8 -*-
"""
Yerel (offline, API key gerektirmeyen) Yapay Zeka / istatistiksel icgoru motoru - v2.

Kapsam:
  - Otomatik Turkce anlati (narrative) uretimi: UAP 2040 referans degerleriyle
    kiyaslama, en yuksek/en dusuk ilceler, genel egilimler.
  - Anomali / veri kalitesi tespiti (IsolationForest) - buyuk veride SQL
    reservoir-sample ile TAM VERIDEN cekilen temsili bir ornek uzerinde.
  - Ilce/bolge profillemesi icin KMeans kumeleme + otomatik optimum k secimi
    (silhouette skoru) - ilce-duzeyi agregatlar zaten kucuk oldugundan TAM
    VERIYE dayanir, ornekleme yapilmaz.
  - Sayisal degiskenler arasi korelasyon: DuckDB'nin CORR() agregat
    fonksiyonuyla TAM VERI uzerinde (satirlari Python'a hic cekmeden).
  - Ozellik onemi (feature importance): RandomForest ile hangi degiskenlerin
    hedef degiskeni (orn. yolculuk suresi) en cok etkiledigini bulur.
  - Yonetici ozeti: tum uretilen bulgular indirilebilir tek bir Markdown raporu.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from . import benchmarks as bm
from . import analytics as an


# --------------------------------------------------------------- narrative
def narrative_household_size(user_avg: float | None) -> str | None:
    if user_avg is None or pd.isna(user_avg):
        return None
    ref = bm.UAP2040_REFERENCE["hanehalki_buyuklugu"]["ortalama"]
    delta = user_avg - ref
    yon = "yuksek" if delta > 0 else "dusuk"
    return (
        f"Yuklenen veride ortalama hanehalki buyuklugu **{user_avg:.2f}** kisi olarak hesaplandi. "
        f"Bu deger, UAP 2040 raporunun referans degeri olan **{ref:.2f}** kisiye gore "
        f"%{abs(delta) / ref * 100:.1f} daha **{yon}**tir."
    )


def narrative_gender(pct_series: pd.Series | None) -> str | None:
    if pct_series is None or pct_series.empty:
        return None
    top = pct_series.sort_values(ascending=False)
    ref = bm.UAP2040_REFERENCE["cinsiyet"]
    return (
        f"Cinsiyet dagiliminda en yuksek pay **{top.index[0]}** icin %{top.iloc[0]:.1f}. "
        f"UAP 2040 referansi Erkek %{ref['uap_erkek_pct']} / Kadin %{ref['uap_kadin_pct']} seklindeydi; "
        f"yuklenen veri bu dengeyle {'benzer' if abs(top.iloc[0]-50) < 5 else 'farkli'} bir yapida."
    )


def narrative_avg_age(user_avg: float | None) -> str | None:
    if user_avg is None or pd.isna(user_avg):
        return None
    ref = bm.UAP2040_REFERENCE["yas"]["ortalama_yas"]
    return (
        f"Ortalama yas **{user_avg:.1f}** olarak hesaplandi (UAP 2040 referansi: {ref})."
        f" {'Nufus yapisi rapora gore daha genc.' if user_avg < ref else 'Nufus yapisi rapora gore daha yasli.'}"
    )


def narrative_district_extremes(df_metric: pd.DataFrame | None, label_col: str, value_col: str,
                                 metric_name: str, unit: str = "") -> str | None:
    if df_metric is None or df_metric.empty or len(df_metric) < 2:
        return None
    d = df_metric.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    if d.empty:
        return None
    top = d.iloc[0]
    bottom = d.iloc[-1]
    mean_v = d[value_col].mean()
    return (
        f"**{metric_name}** en yuksek **{top[label_col]}** ({top[value_col]:.2f}{unit}), "
        f"en dusuk **{bottom[label_col]}** ({bottom[value_col]:.2f}{unit}) bolgesinde gorulmus; "
        f"genel ortalama {mean_v:.2f}{unit}."
    )


def narrative_purpose_mode(purpose_pct: pd.Series | None, mode_pct: pd.Series | None) -> list[str]:
    out = []
    if purpose_pct is not None and not purpose_pct.empty:
        top = purpose_pct.sort_values(ascending=False)
        out.append(f"Yolculuklarin en buyuk bolumunu **{top.index[0]}** amaci olusturuyor (%{top.iloc[0]:.1f}).")
    if mode_pct is not None and not mode_pct.empty:
        top = mode_pct.sort_values(ascending=False)
        out.append(f"En cok kullanilan ulasim turu **{top.index[0]}** (%{top.iloc[0]:.1f}).")
    return out


def narrative_peak_hour(hourly: pd.Series | None) -> str | None:
    if hourly is None or hourly.empty:
        return None
    peak = hourly.idxmax()
    ref = bm.UAP2040_REFERENCE["zirve_saatler"]["sabah_zirve"]
    return (
        f"Yolculuklarin zirve saati **{peak}** olarak tespit edildi "
        f"(%{hourly.max():.1f} pay ile). UAP 2040 raporunda sabah zirvesi {ref} olarak bulunmustu."
    )


# --------------------------------------------------------------- anomali tespiti
def detect_outlier_districts(df_metric: pd.DataFrame | None, value_col: str, label_col: str,
                              z_thresh: float = 2.0) -> pd.DataFrame:
    if df_metric is None:
        return pd.DataFrame()
    d = df_metric.dropna(subset=[value_col]).copy()
    if d.empty or d[value_col].std(ddof=0) == 0:
        return pd.DataFrame()
    d["z_skoru"] = (d[value_col] - d[value_col].mean()) / d[value_col].std(ddof=0)
    return d[d["z_skoru"].abs() >= z_thresh][[label_col, value_col, "z_skoru"]].sort_values(
        "z_skoru", key=lambda s: s.abs(), ascending=False)


def detect_row_anomalies_sql(con: duckdb.DuckDBPyConnection, table: str, numeric_cols: list[str],
                              total_rows: int, contamination: float = 0.02,
                              sample_n: int = 100_000) -> tuple[pd.DataFrame, int]:
    """IsolationForest satir-bazli anomali taramasi. TAM veriden DuckDB'nin
    reservoir-sample ozelligiyle temsili bir ornek cekilir (buyuk veride
    bellekte ML modeli calistirmanin tek pratik yolu budur); dondurulen
    ikinci deger orneklemenin gercekte kacinci satirdan cekildigidir."""
    use_cols = numeric_cols[:15]
    if len(use_cols) < 2:
        return pd.DataFrame(), 0

    exprs = ", ".join(f"{an.numeric_expr(c)} AS {an.ident('c'+str(i))}" for i, c in enumerate(use_cols))
    if total_rows > sample_n:
        query = f"SELECT {exprs} FROM {table} USING SAMPLE {sample_n} ROWS (reservoir)"
        used_n = sample_n
    else:
        query = f"SELECT {exprs} FROM {table}"
        used_n = total_rows
    work = con.execute(query).fetchdf()
    work.columns = use_cols
    work = work.dropna()
    if len(work) < 50:
        return pd.DataFrame(), used_n

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = model.fit_predict(work)
    scores = model.decision_function(work)
    result = work.copy()
    result["anomali_skoru"] = scores
    result = result[labels == -1].sort_values("anomali_skoru")
    return result, used_n


# --------------------------------------------------------------- kumeleme
def _cluster_profile_text(d: pd.DataFrame, label_col: str, feature_cols: list[str]) -> str:
    overall_mean = d[feature_cols].mean()
    profiles = []
    for c in sorted(d["kume"].unique()):
        sub = d[d["kume"] == c]
        diffs = (sub[feature_cols].mean() - overall_mean) / overall_mean.replace(0, np.nan).abs()
        diffs = diffs.dropna().sort_values(ascending=False)
        if diffs.empty:
            continue
        top_feat = diffs.index[0]
        direction = "yuksek" if diffs.iloc[0] > 0 else "dusuk"
        members = ", ".join(sub[label_col].astype(str).head(6).tolist())
        profiles.append(
            f"**Kume {c}** ({len(sub)} bolge, orn: {members}): ortalamaya gore **{top_feat}** "
            f"degeri belirgin sekilde **{direction}**."
        )
    return "\n\n".join(profiles)


def auto_select_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """Silhouette skoruna gore optimum kume sayisini otomatik secer (basit,
    ama gercek bir model-secim adimi - rastgele bir varsayilan degil)."""
    best_k, best_score = k_min, -1
    k_max = min(k_max, len(X) - 1)
    for k in range(k_min, max(k_min, k_max) + 1):
        if k >= len(X):
            break
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            score = silhouette_score(X, km.labels_)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:  # noqa: BLE001
            continue
    return best_k


def cluster_districts(df_wide: pd.DataFrame, label_col: str, feature_cols: list[str],
                       n_clusters: int | None = None) -> tuple[pd.DataFrame, str, int] | None:
    d = df_wide.dropna(subset=feature_cols).copy()
    if len(d) < 3:
        return None
    X = StandardScaler().fit_transform(d[feature_cols])
    k = n_clusters or auto_select_k(X)
    k = max(2, min(k, len(d) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    d["kume"] = km.fit_predict(X)
    return d[[label_col, "kume"] + feature_cols], _cluster_profile_text(d, label_col, feature_cols), k


# --------------------------------------------------------------- korelasyon (tam veri)
def correlation_insights(con, table, numeric_cols: list[str], top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = an.correlation_matrix_sql(con, table, numeric_cols)
    if corr.empty:
        return pd.DataFrame(), corr
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((cols[i], cols[j], v))
    out = pd.DataFrame(pairs, columns=["Degisken 1", "Degisken 2", "Korelasyon"])
    if out.empty:
        return out, corr
    out = out.reindex(out["Korelasyon"].abs().sort_values(ascending=False).index).head(top_n)
    return out, corr


# --------------------------------------------------------------- ozellik onemi
def _numeric_fraction(con, table, col: str) -> tuple[float, int]:
    """Sutunun ne kadarinin (Turkce virgul/nokta bicimleri dahil) sayisal
    oldugunu TAM VERI uzerinde SQL ile olcer - Python tarafinda yanlis
    pd.to_numeric tahminine (orn. '12,5' -> NaN) dusmemek icin."""
    e = an.numeric_expr(col)
    row = con.execute(
        f"SELECT COUNT({an.ident(col)}), COUNT({e}), COUNT(DISTINCT {e}) FROM {table}"
    ).fetchone()
    total, numok, distinct_num = row
    if not total:
        return 0.0, 0
    return numok / total, (distinct_num or 0)


def feature_importance_analysis(con, table, target_col: str, feature_cols: list[str],
                                 total_rows: int, sample_n: int = 100_000) -> tuple[pd.DataFrame, str] | None:
    """RandomForest ile 'hangi degisken hedefi en cok etkiliyor' analizi.
    Sutunlarin sayisal mi kategorik mi oldugu ONCE SQL ile (Turkce ondalik
    bicimleri de dogru taniyarak) TAM VERIDEN tespit edilir; boylece aslinda
    sayisal olan (orn. '12,5' dk gibi virgullu) bir sutun yanlislikla binlerce
    sinifli bir siniflandirma hedefi olarak islenip modeli yavaslatmaz.
    Egitim, bellekte calisabilmesi icin temsili bir SQL reservoir-sample
    uzerinde yapilir (acikca belirtilir)."""
    cols = [c for c in feature_cols if c != target_col][:12]
    if not cols or not target_col:
        return None

    target_frac, target_distinct = _numeric_fraction(con, table, target_col)
    is_classification = not (target_frac > 0.9 and target_distinct > 12)
    if is_classification:
        # asiri fazla sinifli (orn. bozuk/ID benzeri) hedefleri elemeli -
        # yoksa model gereksiz yavaslar ve anlamsiz sonuc uretir.
        n_classes_probe = con.execute(
            f"SELECT COUNT(DISTINCT {an.ident(target_col)}) FROM {table} WHERE {an.ident(target_col)} IS NOT NULL"
        ).fetchone()[0]
        if n_classes_probe is None or n_classes_probe > 50:
            return None

    col_is_numeric = {}
    for c in cols:
        frac, distinct_n = _numeric_fraction(con, table, c)
        col_is_numeric[c] = frac > 0.85 and distinct_n > 1

    sel_exprs = []
    target_alias = "target"
    sel_exprs.append(f"{an.numeric_expr(target_col) if not is_classification else an.ident(target_col)} AS {target_alias}")
    for i, c in enumerate(cols):
        alias = f"f{i}"
        expr = an.numeric_expr(c) if col_is_numeric[c] else an.ident(c)
        sel_exprs.append(f"{expr} AS {alias}")
    select_sql = ", ".join(sel_exprs)

    if total_rows > sample_n:
        query = f"SELECT {select_sql} FROM {table} USING SAMPLE {sample_n} ROWS (reservoir)"
    else:
        query = f"SELECT {select_sql} FROM {table}"
    work = con.execute(query).fetchdf()
    work.columns = [target_alias] + [f"f{i}" for i in range(len(cols))]
    work = work.dropna(subset=[target_alias])
    if len(work) < 100:
        return None

    X = pd.DataFrame(index=work.index)
    feature_names = {}
    for i, c in enumerate(cols):
        alias = f"f{i}"
        feature_names[alias] = c
        col = work[alias]
        if col_is_numeric[c]:
            med = col.median()
            X[alias] = col.fillna(med if pd.notna(med) else 0)
        else:
            X[alias] = LabelEncoder().fit_transform(col.astype(str).fillna("NA"))

    if is_classification:
        y = LabelEncoder().fit_transform(work[target_alias].astype(str))
        if pd.Series(y).nunique() < 2:
            return None
        model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8, n_jobs=-1)
    else:
        y = work[target_alias]
        model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=8, n_jobs=-1)

    model.fit(X, y)
    importance = pd.DataFrame({
        "Degisken": [feature_names[a] for a in X.columns], "Onem Skoru": model.feature_importances_,
    }).sort_values("Onem Skoru", ascending=False)

    top = importance.iloc[0]
    kind = "siniflandirma (kategori tahmini)" if is_classification else "regresyon (sayisal tahmin)"
    text = (
        f"**{target_col}** degiskenini {kind} modeliyle en cok etkileyen degisken "
        f"**{top['Degisken']}** (onem skoru {top['Onem Skoru']:.3f}). "
        f"Model {len(work):,} satirlik {'ornek' if total_rows > sample_n else 'tam veri'} uzerinde egitildi."
    )
    return importance, text


# --------------------------------------------------------------- yonetici ozeti
def build_full_report(sections: list[str]) -> str:
    return "\n\n".join(s for s in sections if s)
