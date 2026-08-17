# -*- coding: utf-8 -*-
"""
Ham CSV sutunlarini analiz kavramlarina otomatik eslestirme (auto-guess) motoru.
Kullanici CSV'sinin sutun adlari bilinmedigi icin, anahtar kelime tabanli bir
tahmin yapip UI'da kullaniciya onay/duzeltme imkani sunulur.
"""
from __future__ import annotations
import difflib
import re
import unicodedata

# concept -> (etiket, aciklama, zorunlu mu, tip, UI grubu)
# UI grubu: mapping ekranini konularina gore kucuk, sade bolumlere ayirmak icin.
GROUP_LABELS = {
    "kimlik": "🆔 Kimlik / Bolge",
    "kisi": "🧑 Kisi & Demografi",
    "hane": "🏠 Hane, Gelir & Arac",
    "yolculuk": "🚗 Yolculuk",
    "gelismis": "⚙️ Gelismis (opsiyonel)",
}

CONCEPTS = {
    "district": ("Ilce / Bolge (Baslangic)", "Yolcunun/hanenin bagli oldugu ya da yolculugun baslangic ilcesi", False, "kategorik", "kimlik"),
    "zone": ("Zon / TAZ / Mahalle", "Trafik analiz zonu veya mahalle kodu", False, "kategorik", "kimlik"),
    "household_id": ("Hane No", "Haneyi tekil belirleyen kimlik", False, "id", "kimlik"),
    "person_id": ("Kisi No", "Bireyi tekil belirleyen kimlik", False, "id", "kimlik"),

    "gender": ("Cinsiyet", "Kisinin cinsiyeti", False, "kategorik", "kisi"),
    "age": ("Yas", "Kisinin yasi (sayisal)", False, "sayisal", "kisi"),
    "education": ("Egitim Duzeyi", "Bitirilen egitim seviyesi", False, "kategorik", "kisi"),
    "student": ("Ogrencilik Durumu", "Ogrenci olup olmadigi / egitim durumu", False, "kategorik", "kisi"),
    "employment": ("Calisma Durumu / Sektor", "Istihdam durumu, meslek ya da sektor", False, "kategorik", "kisi"),
    "literacy": ("Okuma Yazma Durumu", "Okuryazar olup olmadigi (Rapor Grafik 9-10)", False, "kategorik", "kisi"),

    "income": ("Hanehalki Geliri", "Sayisal gelir ya da gelir grubu", False, "sayisal", "hane"),
    "household_size": ("Hanehalki Buyuklugu", "Hanedeki kisi sayisi", False, "sayisal", "hane"),
    "vehicle_count": ("Arac Sayisi", "Hanedeki otomobil/arac sayisi", False, "sayisal", "hane"),

    "purpose": ("Yolculuk Amaci", "Is, okul, universite, diger vb.", False, "kategorik", "yolculuk"),
    "mode": ("Ulasim Turu", "Otomobil, otobus, metro, yurume vb.", False, "kategorik", "yolculuk"),
    "duration": ("Yolculuk Suresi (dk)", "Yolculugun sayisal suresi", False, "sayisal", "yolculuk"),
    "start_time": ("Baslangic Saati", "Yolculuk baslangic zamani", False, "zaman", "yolculuk"),
    "end_time": ("Bitis Saati", "Yolculuk bitis zamani", False, "zaman", "yolculuk"),
    "district_dest": ("Ilce / Bolge (Varis)", "Yolculugun vardigi ilce (O-D matrisi icin)", False, "kategorik", "yolculuk"),

    "transfer": ("Aktarma Sayisi", "Yolculuktaki aktarma sayisi", False, "sayisal", "gelismis"),
    "cost": ("Yolculuk Maliyeti", "Yolculuga iliskin ucret/maliyet", False, "sayisal", "gelismis"),
    # NOT: grup "kimlik" - bu, ilce/hane kimligi gibi TUM sekmelerde ortak
    # kullanilan, MERKEZI "Kimlik / Bolge Eslestirmesi" panelinde BIR KEZ
    # eslestirilen bir kavram olsun diye bilerek boyle secildi (bkz. app.py
    # tab_overview). Eslenirse TUM istatistikler - ortalama, yuzde, Gini vb. -
    # otomatik olarak AGIRLIKLI hesaplanir (bkz. analytics.py _with_weight);
    # eslenmezse (varsayilan) HICBIR SEY degismez.
    "weight": ("Anket Agirligi (Genisletme Katsayisi)",
               "Varsa: her satirin gercek nufusu temsil etme agirligi. Eslenirse TUM "
               "ortalama/yuzde/Gini hesaplari otomatik AGIRLIKLI yapilir.", False, "sayisal", "kimlik"),
    "park_location": ("Arac Park Yeri", "Aracin aksam park edildigi yer (Rapor Grafik 33)", False, "kategorik", "gelismis"),

    # Rapor 1.5.2.10 "Ozel Arac Yolculuk Istatistikleri" (Tablo 31) icin.
    "vehicle_occupancy": ("Aractaki Kisi Sayisi", "Yolculuk sirasinda aracta soför dahil kac kisi oldugu", False, "sayisal", "gelismis"),
    "driver_passenger": ("Surucu / Yolcu Durumu", "Kisinin bu yolculukta surucu mu yolcu mu oldugu", False, "kategorik", "gelismis"),
    "trip_park_type": ("Yolculuk Sirasinda Otopark Turu", "O yolculukta aracin nereye park edildigi (ucretli/ucretsiz vb.)", False, "kategorik", "gelismis"),
}


def concepts_by_group() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {g: [] for g in GROUP_LABELS}
    for concept, meta in CONCEPTS.items():
        out[meta[4]].append(concept)
    return out

_KEYWORDS = {
    "district": ["ilçe", "ilce", "district", "ilcekodu", "ilceadi", "baslangic ilce", "başlangıç ilçe"],
    "district_dest": ["varis ilce", "varış ilçe", "hedef ilce", "hedef ilçe", "dest district",
                       "bitis ilce", "bitiş ilçe", "yol bit ilce", "bitilce"],
    "zone": ["taz", "zon", "mahalle", "trafik analiz", "mahal", "evzon", "zone code", "zncode"],
    "gender": ["cinsiyet", "sex", "gender", "cnsyt", "cins"],
    "age": ["yaş", "yas", "age", "dogumyili", "doğum yılı"],
    "education": ["eğitim", "egitim", "education", "ogrenim", "öğrenim", "egit", "edu"],
    "student": ["öğrenci", "ogrenci", "student", "öğrencilik", "devegit"],
    "employment": ["çalışma durumu", "calisma durumu", "istihdam", "meslek",
                   "employment", "iş durumu", "is durumu", "sektor", "sektör",
                   "cdurum", "occupation", "occup", "job"],
    "literacy": ["okuryazar", "okur yazar", "okuma yazma", "literacy"],
    # NOT: "gelirgrup" bilerek DAHIL EDILMEDI - bircok gercek dosyada bu sutun
    # "20.001-35.000 TL" gibi bir ARALIK METNI tutar (SAYISAL DEGIL); "income"
    # kavrami SAYISAL bir deger bekledigi icin bu sutun otomatik secilirse
    # TUM gelir analizleri (Gini, gelir dagilimi, ilce/gelir kirilimlari)
    # sessizce %100 bos/NULL cikar - gercek bir dosyada olculup dogrulandi
    # (gelirgrup: 15.610 satirin 0'i sayisala cevrilebildi; gelirtext ayni
    # dosyada 6.683 satirda gecerli sayisal gelir veriyor). "gelirtext" gibi
    # GERCEKTEN sayisal bir sutun varsa o tercih edilsin diye "gelirgrup"
    # anahtar kelime listesinden cikarildi (yine de "gelir" alt-dizesiyle
    # zayif bir aday olarak bulunabilir, arama kutusuyla elle secilebilir).
    "income": ["gelir", "income", "hane geliri", "gelirtext"],
    "household_id": ["hane no", "hane_no", "hane id", "household_id", "haneid", "hanekod",
                      "haneno", "hhid", "hh id"],
    "person_id": ["kişi no", "kisi no", "kisi_no", "person_id", "bireyno", "birey no"],
    "household_size": ["hanehalkı büyüklüğü", "hanehalki buyuklugu", "hane büyüklüğü",
                        "household_size", "hane kisi sayisi", "hanedeki kisi", "kisisay",
                        "hhsize", "hh size"],
    "vehicle_count": ["araç sayısı", "arac sayisi", "oto sayısı", "oto sayisi",
                       "vehicle_count", "vehicle count", "otomobil sayisi",
                       "aracadet", "arac adet", "araç adet", "toplam arac", "toplam araç"],
    # "yol bit" - bircok UAP2040-tarzi anket dosyasinda yolculuk amaci ayri bir
    # "amac" sutunu olarak DEGIL, varis (bitis) noktasinin TURU/KODU olarak
    # tutulur (orn. "yol_bit": 1=eve donus, 3=is, 7=egitim gibi) - bu, "neden
    # oraya gittiniz" sorusunun standart kodlanma bicimidir. Alt-dize eslesme
    # riskini (yol_bit_ilce/saat/adres/mahal gibi kardes sutunlarla
    # karismasini) asagidaki _CONCEPT_EXCLUDE_SUBSTR onler.
    # "bityer" - ayni "yol bit" mantigi (yukaridaki nota bkz.), ama BASKA bir
    # UAP2040-tarzi anket saglayicisinin kullandigi ISIMLENDIRME (gercek bir
    # 2. sehrin sahada verisiyle olculup dogrulandi: "kisi_N_bityer_M" ->
    # donusum sonrasi "bityer_N_M"). "bityer" (yer=place/destination) tek
    # basina kardes sutunlarla (bitilce/bitmahal/bitsaat/bitadres) CAKISMAZ
    # (hicbiri "yer" alt-dizesini icermez) - ayri bir exclude kurali gerekmez.
    "purpose": ["amaç", "amac", "purpose", "yolculuk amacı", "seyahat amacı", "yol bit", "bityer",
                "trip purpose", "purp"],
    # "yol arac" (sondaki bacak/transfer numarasi OLMADAN, orn. "yol_arac1",
    # "yol_arac2", "yol_arac3"...): bazi UAP2040-tarzi dosyalarda bir
    # yolculugun HER aktarma bacagi icin ayri bir "yol_aracN" sutunu olur -
    # sonundaki rakami DA iceren eski bir anahtar kelime ("yol arac1")
    # SADECE ILK bacagi eslestirip diger bacaklari (yol_arac2, yol_arac3...)
    # aday listesinden tamamen dislyordu (gercek bir dosyada kullanicinin
    # kendi gozlemiyle yakalandi - dropdown'da SADECE "yol_arac1" goruluyordu).
    # Rakamsiz "yol arac" alt-dize olarak HEPSIYLE eslesir.
    "mode": ["ulaşım türü", "ulasim turu", "mode", "araç türü", "arac turu", "ulasim_tipi",
             "ulaşım tipi", "yol arac", "ulastur1", "travel mode", "travmode"],
    "duration": ["süre", "sure", "duration", "dakika", "yolculuk süresi"],
    "start_time": ["başlangıç saat", "baslangic saat", "start_time", "start time",
                   "kalkış saati", "kalkis saati", "yol bas saat", "bas saat", "basssat",
                   "starttime", "departure time"],
    "end_time": ["bitiş saat", "bitis saat", "end_time", "end time", "varış saati",
                 "varis saati", "yol bit saat", "bit saat", "bitsaat",
                 "endtime", "arrival time"],
    "transfer": ["aktarma", "transfer"],
    "cost": ["maliyet", "ücret", "ucret", "cost", "fiyat", "bilet", "yol maliyet1", "fare"],
    "weight": ["ağırlık", "agirlik", "weight", "genişletme", "genisletme katsayisi",
               "expansion weight", "exp wt", "expwt"],
    "park_location": ["park yeri", "park yer", "parkyeri", "park location", "otopark"],
    # Rapor 1.5.2.10 (Tablo 31) icin - "yol arackisi1" gibi cift-indeksli
    # (kisi x yolculuk) bacak sutunlari; "yol surucu1"/"yol parktipi1" ayni
    # sekilde. "park_location" (yukaridaki hane-bazli 'aksam nereye park
    # edilir' sorusu) ile KARISMAMASI icin buradaki anahtar kelimeler
    # bilerek "yol " onekiyle SINIRLI tutuldu (parkyeri ile cakismaz).
    "vehicle_occupancy": ["araç içi kişi", "arac ici kisi", "yol arackisi", "vehicle occupancy",
                          "araçtaki kişi", "aractaki kisi", "occupancy", "arackisi1"],
    "driver_passenger": ["sürücü müydü", "surucu muydu", "yol surucu", "sürücü/yolcu",
                         "driver passenger", "sürücü yolcu", "surucu1"],
    "trip_park_type": ["yol parktipi", "park tipi", "park türü", "park turu",
                       "otopark tipi", "otopark türü", "parktipi1"],
}


def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    # snake_case / kebab-case sutun adlarinin da (orn. "arac_sayisi") bosluklu
    # anahtar kelimelerle (orn. "arac sayisi") eslesebilmesi icin ayirac
    # karakterlerini bosluga cevirip fazla boslugu sadelestir.
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NORMED_KEYWORDS: dict[str, list[str]] = {
    concept: [nk for kw in kws if (nk := _norm(kw))] for concept, kws in _KEYWORDS.items()
}

# "Sutun Gruplari" bolumunde bir taban (orn. "cins", "yas", "aracadet")
# TOPLA/ORTALAMA/MAKS ile birlestirilebilir; sonuc sutun adi HER ZAMAN
# "{taban}_{sum|avg|max|count_nonzero}" seklindedir. Bu, hanedeki arac
# SAYISI gibi gercekten toplanabilir seyler icin anlamlidir - ama cinsiyet,
# yas, egitim gibi KISI OZNITELIKLERI icin hicbir zaman anlamli degildir
# (11 kisinin cinsiyet kodunu TOPLAMAK/ORTALAMASINI ALMAK anlamsiz bir
# sayi uretir, orn. yas toplami "116" gibi imkansiz bir "yas" gorunur).
# Bu yuzden bu tur turetilmis sutunlar, asagidaki kavramlar DISINDA hicbir
# yerde otomatik tahmin/oneri olarak GOSTERILMEZ.
_COMBINE_SUFFIXES = ("_sum", "_avg", "_max", "_count_nonzero")
_COMBINE_SAFE_CONCEPTS = {"vehicle_count"}


def _is_combine_derived(col: str) -> bool:
    return col.endswith(_COMBINE_SUFFIXES)


def _filter_unsafe_combines(concept: str, columns: list[str]) -> list[str]:
    if concept in _COMBINE_SAFE_CONCEPTS:
        return columns
    return [c for c in columns if not _is_combine_derived(c)]


# BAZI kavramlar icin, belirli bir alt-dizeyi iceren sutunlar KESINLIKLE
# YANLIS bir eslesme olur - gercek bir dosyada "yol_parksure1" (PARK suresi)
# "duration" (yolculuk suresi) icin "sure" alt-dizesiyle en iyi (ama YANLIS)
# aday olarak seciliyordu, bu da hem yanlis bir sutunun eslenmesine hem de
# 'baslangic/bitis saatinden otomatik sure turetme' ozelliginin hic
# tetiklenmemesine (zaten 'dolu' gorundugu icin) yol aciyordu.
_CONCEPT_EXCLUDE_SUBSTR: dict[str, tuple[str, ...]] = {
    "duration": ("park",),
    # "yol bit" anahtar kelimesi (yukariya bkz.) hem amac kodunu tutan CIPLAK
    # "yol_bit" sutununa HEM DE onun ilce/saat/adres/mahal kardeslerine
    # (orn. "yol_bit_ilce") alt-dize olarak eslesir - bu kardesler burada
    # elenerek eslesme SADECE amac kodunu tutan sutuna daralir.
    "purpose": ("ilce", "saat", "adres", "mahal"),
    # "yol arac" anahtar kelimesi (yukariya bkz. - rakamsiz, tum bacaklarla
    # eslessin diye) "yol_aracicii1/2/3" (aractaki kisi/doluluk) ve
    # "yol_arackisi1/2/3" (arac-kisi eslestirmesi) gibi TAMAMEN FARKLI
    # kardes sutunlarla da alt-dize olarak eslesirdi - bunlar "arac TURU"
    # (mode) degil, ayri kavramlardir; burada elenir.
    "mode": ("icii", "kisi"),
}


def _excluded(concept: str, ncol: str) -> bool:
    return any(bad in ncol for bad in _CONCEPT_EXCLUDE_SUBSTR.get(concept, ()))


# BASKA BIR SEHRIN dosyasinda ayni kavram, bilinen anahtar kelimelerin
# HICBIRIYLE tam/alt-dize eslesmeyen bir KISALTMA/YAZIM ile gelebilir (orn.
# "cinsiyet" yerine "cinsiyt", "gender" yerine "gendr", "yas" yerine "yss").
# Bu durumda, ek bagimlilik gerektirmeyen (Python standart kutuphanesi)
# difflib.SequenceMatcher ile HARF DIZILISI benzerligine bakilarak son bir
# tahmin denemesi yapilir. Esik (0.82) KASTEN YUKSEK tutulur - yanlis bir
# sutunu "belki" diye eslemek, hic eslememekten DAHA KOTUDUR (kullanicinin
# "hesaplanan degerler asla yanlis/uydurma olmayacak" kurali geregi); bu
# yuzden fuzzy skor HER ZAMAN gercek bir alt-dize eslesmesinden DUSUK
# kalacak sekilde olceklenir - sadece BASKA HICBIR ADAY yoksa devreye girer.
_FUZZY_MIN_RATIO = 0.82


def _column_score(ncol: str, normed_keywords: list[str]) -> float:
    """En alakali eslesmenin skorunu dondurur.

    DOGRULUK ICIN ONEMLI: TAM KELIME eslesmesi (orn. 'yas_1' -> token 'yas'),
    bir baska kelimenin ICINE GOMULU alt-dize eslesmesinden (orn.
    'besyassay' icinde 'yas' gecmesi - bu aslinda 'bes yas alti sayisi'
    gibi TAMAMEN FARKLI bir hane-bazli sayim alanidir, kisi yasi degildir)
    KESINLIKLE daha yuksek puan alir. Bu ayrim olmadan kisa anahtar
    kelimeler (orn. 'yas', 'il') yanlis sutunlarla eslesebiliyordu."""
    tokens = ncol.split(" ")
    token_set = set(tokens)
    # "compact" - ncol'un BOSLUKSUZ hali (orn. "zn code" -> "zncode"). Bazi
    # gercek dosyalarda sutun adi birden fazla kelimeye ayrilirken (norm
    # sonrasi), bilinen anahtar kelime TEK (bosluksuz) bir kisaltmadir
    # (orn. "zncode") - sadece tek tek token'larla karsilastirma bu durumu
    # KACIRIR, compact ile de karsilastirma bunu yakalar.
    compact = ncol.replace(" ", "")
    fuzzy_targets = tokens if compact in token_set else tokens + [compact]
    best = 0
    fuzzy_best = 0.0
    for nkw in normed_keywords:
        if not nkw:
            continue
        if nkw in token_set:
            score = len(nkw) * 10  # tam kelime eslesmesi - guclu tercih
        elif nkw in ncol:
            score = len(nkw)  # alt-dize eslesmesi - zayif, yedek tercih
        else:
            score = 0
            if " " not in nkw and len(nkw) >= 3:
                for tok in fuzzy_targets:
                    if len(tok) < 3:
                        continue
                    ratio = difflib.SequenceMatcher(None, nkw, tok).ratio()
                    if ratio >= _FUZZY_MIN_RATIO and ratio > fuzzy_best:
                        fuzzy_best = ratio
        if score > best:
            best = score
    if best > 0:
        return best
    # Gercek (tam/alt-dize) eslesme HICBIR anahtar kelime icin bulunamadi -
    # sadece bu durumda, bulanik eslesme SON CARE olarak kullanilir. Puan
    # kasten dusuk tutulur (en kisa gercek alt-dize eslesmesinin bile
    # ALTINDA kalir) ki gercek bir eslesme HER ZAMAN bulanik olana tercih
    # edilsin.
    return fuzzy_best * 2 if fuzzy_best else 0


def auto_guess_mapping(columns: list[str]) -> dict[str, str | None]:
    """Her kavram icin en olasi CSV sutununu tahmin eder. Bulunamazsa None doner.
    Buyuk/genis (binlerce sutunlu) dosyalarda hizli kalmasi icin anahtar
    kelimeler bir kez normallestirilir (dongu icinde tekrar tekrar degil)."""
    normed = {c: _norm(c) for c in columns}
    used: set[str] = set()
    guess: dict[str, str | None] = {k: None for k in CONCEPTS}

    for concept, normed_keywords in _NORMED_KEYWORDS.items():
        if not normed_keywords:
            continue
        unsafe_ok = concept in _COMBINE_SAFE_CONCEPTS
        best_col = None
        best_score = 0
        for col, ncol in normed.items():
            if col in used:
                continue
            if not unsafe_ok and _is_combine_derived(col):
                continue  # bkz. _filter_unsafe_combines - yanlis/anlamsiz eslesmeyi onler
            if _excluded(concept, ncol):
                continue  # bkz. _CONCEPT_EXCLUDE_SUBSTR - orn. "park suresi" != "yolculuk suresi"
            score = _column_score(ncol, normed_keywords)
            if score > best_score:
                best_score = score
                best_col = col
        if best_col:
            guess[concept] = best_col
            used.add(best_col)
    return guess


def ranked_candidates(columns: list[str], concept: str, top_k: int = 40) -> list[str]:
    """Bir kavram icin en olasi (anahtar kelimeyle eslesen) sutunlari, en
    alakalidan en aza siralanmis sekilde, en fazla top_k adet dondurur.
    Amac: binlerce sutunlu dosyalarda her eslestirme kutusuna TUM sutunlari
    (orn. 5699 adet) degil, kucuk/hizli bir aday listesi vermek."""
    normed_keywords = _NORMED_KEYWORDS.get(concept, [])
    if not normed_keywords:
        return []
    candidates = _filter_unsafe_combines(concept, columns)
    scored = []
    for c in candidates:
        nc = _norm(c)
        if _excluded(concept, nc):
            continue
        score = _column_score(nc, normed_keywords)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in scored[:top_k]]


def search_columns(columns: list[str], query: str, limit: int = 300) -> list[str]:
    """Serbest metin arama: sutun adinda gecen alt-dize eslesmesi (kucuk/
    buyuk harf duyarsiz). Sonuc, ekrani/agi bogmamak icin limit ile sinirlanir."""
    q = _norm(query)
    if not q:
        return []
    return [c for c in columns if q in _norm(c)][:limit]
