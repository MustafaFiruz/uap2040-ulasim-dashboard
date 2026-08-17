# -*- coding: utf-8 -*-
"""
Turkiye ilce sinirlarini (GeoJSON) kullanarak GERCEK bir harita (choropleth)
cizmek icin yardimcilar.

Veri kaynagi: "geojsons/turkey-admin-level-6" (uyasarkocal/borders-of-turkey,
CC0 1.0 - kamu malı), sadece ilce ADI + SINIR geometrisi kalacak sekilde
sadelestirilip (gereksiz KML alanlari atilip, koordinat hassasiyeti/nokta
yogunlugu azaltilip) 14,6 MB'tan ~3,4 MB'a indirilmis, projeye statik olarak
gomulmus (utils/data/tr_districts.geojson) - calisma zamaninda INTERNETE
IHTIYAC YOKTUR.

ONEMLI/DURUSTLUK: Turkce sehir/ilce adlari CSV'de ve GeoJSON'da FARKLI
yazilabilir (orn. "Yıldırım" vs "Yildirim", "İnegöl" vs "Inegöl") - bu yuzden
esleme, Turkce karakterleri ASCII'ye indirgeyen bir normalizasyon ile yapilir.
Bazi ilce adlari (orn. "Merkez") BIRDEN FAZLA ilde tekrarlanir - boyle
BELIRSIZ adlar (hangi ile ait oldugu GeoJSON'da tutulmadigi icin) haritaya
YANLIS/tahmini bir sekilde eklenmez, acikca "eslenemedi" listesine duser.
"""
from __future__ import annotations
import json
import os

_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "data", "tr_districts.geojson")

_TR_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def normalize_tr(name: str) -> str:
    """Turkce bir il/ilce adini karsilastirmaya uygun, sade bir anahtara
    indirger: kucuk harfe cevirir, Turkce ozel karakterleri ASCII'ye
    (ı/İ->i, ş->s, ğ->g, ü->u, ö->o, ç->c) katlar, harf/rakam disindaki
    HER SEYI (bosluk, tire vb.) siler. Boylece "Mustafakemalpaşa" ile
    "Mustafa Kemalpasa" gibi FARKLI yazilmis ama AYNI ilceyi kasteden
    adlar dogru sekilde eslesir."""
    if not name:
        return ""
    s = str(name).strip().lower().translate(_TR_FOLD)
    return "".join(ch for ch in s if ch.isalnum())


_geojson_cache: dict | None = None
_name_index_cache: dict[str, list[dict]] | None = None


def _load_geojson() -> dict:
    global _geojson_cache
    if _geojson_cache is None:
        with open(_GEOJSON_PATH, encoding="utf-8") as f:
            _geojson_cache = json.load(f)
    return _geojson_cache


def _name_index() -> dict[str, list[dict]]:
    """normalize_tr(ilce_adi) -> o adla eslesen TUM GeoJSON feature'lari
    (birden fazlaysa BELIRSIZ demektir - bkz. modul docstring'i)."""
    global _name_index_cache
    if _name_index_cache is None:
        idx: dict[str, list[dict]] = {}
        for feat in _load_geojson()["features"]:
            key = normalize_tr(feat["properties"].get("name", ""))
            if key:
                idx.setdefault(key, []).append(feat)
        _name_index_cache = idx
    return _name_index_cache


def _polygon_centroid(coords) -> tuple[float, float] | None:
    """Bir GeoJSON Polygon/MultiPolygon'un YAKLASIK merkezini bulur (dis
    halkanin noktalarinin ORTALAMASI - gercek alan-agirlikli centroid
    DEGIL, ama akis haritasi gibi 'yaklasik konum yeter' kullanimlar icin
    yeterince dogru ve basit/hizli). MultiPolygon'da EN BUYUK (nokta
    sayisi en fazla) parca kullanilir."""
    def _ring_points(poly):
        # Polygon: [ [ring], [hole], ...] -> ilk (dis) halka
        return poly[0] if poly else []

    if not coords:
        return None
    # Polygon: coords = [[ [lon,lat], ... ]]; MultiPolygon: coords = [ [[ [lon,lat],...]], ... ]
    is_multi = isinstance(coords[0][0][0], (list, tuple))
    rings = [_ring_points(p) for p in coords] if is_multi else [_ring_points(coords)]
    rings = [r for r in rings if r]
    if not rings:
        return None
    best = max(rings, key=len)
    lons = [pt[0] for pt in best]
    lats = [pt[1] for pt in best]
    if not lons or not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


_centroid_cache: dict[str, tuple[float, float]] | None = None


def district_centroids() -> dict[str, tuple[float, float]]:
    """GeoJSON'daki HER ilce icin YAKLASIK merkez (lat, lon) sozlugu doner
    (normalize_tr anahtarli). Akis haritasi (ilceler arasi cizgi) gibi
    'kesin sinir degil, yaklasik nokta yeter' kullanimlar icindir."""
    global _centroid_cache
    if _centroid_cache is None:
        idx = _name_index()
        out = {}
        for key, feats in idx.items():
            if len(feats) != 1:
                continue  # belirsiz (birden fazla ilde ayni ad) - atla, match_districts ile tutarli
            geom = feats[0].get("geometry") or {}
            c = _polygon_centroid(geom.get("coordinates"))
            if c:
                out[key] = c
        _centroid_cache = out
    return _centroid_cache


def match_districts(district_names: list[str]) -> tuple[dict, list[str], list[str]]:
    """Verilen ilce adi listesini GeoJSON sinirlarina eslestirir.

    Doner: (matched_geojson, eslenemeyen_adlar, belirsiz_adlar)
      - matched_geojson: SADECE eslesen ilcelerin feature'larini iceren,
        harita cizmeye HAZIR bir GeoJSON FeatureCollection sozlugu. Her
        feature'in "properties.name" alani, KULLANICININ KENDI ilce adiyla
        (GeoJSON'daki degil) degistirilir - boylece plotly choropleth'in
        `locations` degeri kullanicinin veri tablosundaki adla BIREBIR
        eslesir.
      - eslenemeyen_adlar: GeoJSON'da hicbir karsiligi bulunamayan adlar.
      - belirsiz_adlar: GeoJSON'da AYNI adla BIRDEN FAZLA (farkli illerde)
        ilce bulunan, bu yuzden HANGISI oldugu bilinemeyen ve haritaya
        EKLENMEYEN adlar (orn. 'Merkez')."""
    idx = _name_index()
    out_features = []
    unmatched, ambiguous = [], []
    for name in district_names:
        key = normalize_tr(name)
        candidates = idx.get(key, [])
        if not candidates:
            unmatched.append(name)
        elif len(candidates) > 1:
            ambiguous.append(name)
        else:
            feat = json.loads(json.dumps(candidates[0]))  # sig degistirmeden kopyala
            feat["properties"] = {"name": name}
            out_features.append(feat)
    return {"type": "FeatureCollection", "features": out_features}, unmatched, ambiguous
