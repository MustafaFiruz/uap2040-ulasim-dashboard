# -*- coding: utf-8 -*-
"""
UAP 2040 (Izmir Ulasim Ana Plani 2040) Hanehalki Ulasim Arastirmasi raporundan
(KSR-045-UAP2040_YENI_BILGILER_RAPORU_CILT 1) cikarilan referans degerler.

Bu sabitler, kullanicinin yukledigi ham CSV verisinden hesaplanan sonuclari
karsilastirmali olarak degerlendirmek icin kullanilir (rapordaki TUIK
karsilastirma mantiginin ayni sekilde uygulanmasi).

Kaynak: KSR-045-UAP2040_YENI_BILGILER_RAPORU_CILT 1_R01-17-104.pdf (88 sayfa)
"""

UAP2040_REFERENCE = {
    "meta": {
        "rapor": "Izmir Ulasim Ana Plani 2040 (UAP 2040) - Hanehalki ve Yurt Ulasim Arastirmasi",
        "toplam_anket": 21508,
        "hanehalki_anketi": 20332,
        "yurt_anketi": 1176,
        "ilce_sayisi": 20,
        "mahalle_sayisi": 692,
        "taz_sayisi": 988,
        "orneklem_kitle_nufusu": 3860109,
        "guven_duzeyi_pct": 95,
        "hata_payi_pct": 1.1,
    },
    "hanehalki_buyuklugu": {
        "ortalama": 2.81,
        "tuik_ortalama": 2.77,
        "min": 1,
        "maks": 13,
        "medyan": 3,
        "mode": 2,
        "std": 1.239767,
    },
    "cinsiyet": {
        "uap_erkek_pct": 50.2,
        "uap_kadin_pct": 49.8,
        "tuik_erkek_pct": 50.5,
        "tuik_kadin_pct": 49.5,
    },
    "yas": {
        "ortalama_yas": 38.1,
    },
    "okuryazarlik": {
        "uap_pct": 98.8,
        "tuik_izmir_pct": 98.9,
        "turkiye_ort_pct": 97.8,
    },
    "egitim_duzeyi_pct": {
        # kategori: (UAP2040, TUIK)
        "Okuma Yazma Bilmeyen": (1.2, 1.1),
        "Ilkogretim": (50.6, 48.4),
        "Lise": (27.3, 25.2),
        "Yuksekokul veya Fakulte": (21.6, 20.1),
        "Yuksek Lisans / Doktora": (1.5, 3.0),
    },
    "gelir_grubu_ortalama_yolculuk": {
        # gelir_grubu: {amac: deger}
        "1. %20 (Alt Gelir)": {"is": 0.47, "okul": 0.58, "diger": 0.58, "ev_uclu_olmayan": 0.06, "toplam": 1.73},
        "2. %20": {"is": 0.49, "okul": 0.34, "diger": 0.79, "ev_uclu_olmayan": 0.05, "toplam": 1.73},
        "3. %20": {"is": 0.63, "okul": 0.32, "diger": 0.69, "ev_uclu_olmayan": 0.05, "toplam": 1.75},
        "4. %20": {"is": 0.63, "okul": 0.21, "diger": 0.81, "ev_uclu_olmayan": 0.06, "toplam": 1.77},
        "5. %20 (Ust Gelir)": {"is": 0.89, "okul": 0.18, "diger": 0.65, "ev_uclu_olmayan": 0.10, "toplam": 1.87},
        "Ortalama": {"is": 0.63, "okul": 0.32, "diger": 0.71, "ev_uclu_olmayan": 0.07, "toplam": 1.77},
    },
    "mod_gelir_iliskisi_notu": (
        "Rapora gore gelir arttikca yurume orani %41,5'ten %24,0'e geriliyor, "
        "otomobil kullanimi %12,2'den %38,2'ye (~3 kat) yukseliyor, belediye otobusu "
        "kullanimi %19,1'den %8,2'ye dusuyor; metro/IZBAN gibi rayli sistemler ise "
        "tum gelir gruplarinda %7-9 araliginda istikrarli kaliyor."
    ),
    "zirve_saatler": {
        "sabah_zirve": "08:00-09:00",
        "aksam_zirve": "17:00-19:00",
        "is_baslangic_zirve_pct": 16.7,
        "is_bitis_zirve_pct": 20.6,
        "okul_baslangic_zirve_pct": 26.8,
        "okul_bitis_zirve_pct": 30.2,
        "universite_baslangic_zirve_pct": 17.0,
        "diger_baslangic_zirve_saat": "12:00-13:00",
        "diger_baslangic_zirve_pct": 11.2,
        "ev_uclu_olmayan_baslangic_zirve_saat": "12:00-13:00",
        "ev_uclu_olmayan_baslangic_zirve_pct": 40.6,
    },
    "ozel_arac_yolculuk": {
        "ortalama_yolcu_sayisi_soforl_dahil": 1.57,
        "std": 0.78,
        "sofor_pct": 71.2,
        "yolcu_pct": 28.8,
        "otopark_ucretli_pct": 1.03,
        "otopark_ucretsiz_yol_kenari_pct": 81.90,
        "otopark_ucretsiz_isyeri_avm_pct": 17.07,
    },
    "tablo1_ilce_mahalle": {
        # ilce: (mahalle_sayisi, mahalle_toplam_nufus, orneklem_mahalle_sayisi, orneklem_nufus)
        "Aliaga": (32, 108701, 19, 86891), "Balcova": (8, 76613, 13, 76613),
        "Bayrakli": (24, 299859, 46, 299640), "Bergama": (137, 107346, 22, 84135),
        "Bornova": (45, 448737, 63, 446426), "Buca": (47, 523189, 76, 516294),
        "Cigli": (26, 215685, 33, 215685), "Foca": (16, 36688, 7, 34487),
        "Gaziemir": (16, 136929, 22, 136929), "Guzelbahce": (12, 38500, 8, 37888),
        "Karabaglar": (58, 473058, 76, 472179), "Karsiyaka": (27, 341580, 51, 341247),
        "Kemalpasa": (49, 120332, 15, 103832), "Konak": (112, 322393, 94, 318522),
        "Menderes": (44, 111443, 21, 98027), "Menemen": (65, 214409, 39, 202151),
        "Narlidere": (11, 61732, 11, 61732), "Seferihisar": (22, 60914, 11, 53460),
        "Torbali": (60, 218744, 40, 201975), "Urla": (37, 79610, 25, 71996),
        "Toplam": (848, 3996462, 692, 3860109),
    },
    "il_karsilastirma_hanehalki_buyuklugu": {
        # TUIK - diger iller
        "Aydin": 2.7, "Izmir": 2.77, "Antalya": 2.9, "Tekirdag": 2.8,
        "Kocaeli": 3.0, "Konya": 3.2, "Erzurum": 3.4, "Diyarbakir": 4.1,
    },
}


def get_reference_text_block() -> str:
    """AI icgoru motoru ve raporlama icin kisa, okunabilir referans ozeti."""
    r = UAP2040_REFERENCE
    lines = [
        f"UAP 2040 raporu {r['meta']['toplam_anket']:,} ankete ({r['meta']['ilce_sayisi']} ilce, "
        f"{r['meta']['mahalle_sayisi']} mahalle) dayanmaktadir.",
        f"Referans ortalama hanehalki buyuklugu: {r['hanehalki_buyuklugu']['ortalama']} kisi "
        f"(TUIK: {r['hanehalki_buyuklugu']['tuik_ortalama']}).",
        f"Referans cinsiyet dagilimi: Erkek %{r['cinsiyet']['uap_erkek_pct']} / "
        f"Kadin %{r['cinsiyet']['uap_kadin_pct']}.",
        f"Referans ortalama yas: {r['yas']['ortalama_yas']}.",
        f"Referans okuryazarlik orani: %{r['okuryazarlik']['uap_pct']}.",
        f"Sabah zirve saati {r['zirve_saatler']['sabah_zirve']}, aksam zirve saati "
        f"{r['zirve_saatler']['aksam_zirve']} olarak tespit edilmistir.",
        r["mod_gelir_iliskisi_notu"],
    ]
    return "\n".join(f"- {l}" for l in lines)
