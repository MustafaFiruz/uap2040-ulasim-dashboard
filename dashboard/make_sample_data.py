# -*- coding: utf-8 -*-
"""
Dashboard'u gercek CSV yuklenmeden once test etmek icin, UAP 2040 raporundaki
yapiya benzer (ilce, cinsiyet, yas, egitim, gelir, arac, yolculuk amaci/mod/
sure/saat) sentetik bir 'buyuk' ham veri seti uretir.

Kullanim:
    python make_sample_data.py            # 300.000 satirlik ornek uretir
    python make_sample_data.py 1000000    # 1 milyon satir
"""
import sys
import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
rng = np.random.default_rng(42)

DISTRICTS = ["Aliaga", "Balcova", "Bayrakli", "Bergama", "Bornova", "Buca", "Cigli",
             "Foca", "Gaziemir", "Guzelbahce", "Karabaglar", "Karsiyaka", "Kemalpasa",
             "Konak", "Menderes", "Menemen", "Narlidere", "Seferihisar", "Torbali", "Urla"]
district_weights = rng.dirichlet(np.ones(len(DISTRICTS)) * 3)

PURPOSES = ["Ev Uclu Is", "Ev Uclu Okul", "Ev Uclu Universite", "Ev Uclu Diger", "Ev Uclu Olmayan"]
purpose_weights = [0.30, 0.15, 0.08, 0.32, 0.15]

MODES = ["Yurume", "Otomobil", "Belediye Otobusu", "Metro/IZBAN", "Servis", "Minibus", "Bisiklet"]
mode_weights = [0.28, 0.27, 0.16, 0.10, 0.09, 0.07, 0.03]

EDUCATION = ["Okuma Yazma Bilmeyen", "Ilkogretim", "Lise", "Yuksekokul veya Fakulte", "Yuksek Lisans/Doktora"]
edu_weights = [0.012, 0.50, 0.27, 0.20, 0.018]

EMPLOYMENT = ["Calisan", "Ogrenci", "Ev Hanimi", "Emekli", "Issiz", "Diger"]
emp_weights = [0.42, 0.20, 0.15, 0.12, 0.07, 0.04]

n_household = max(1000, N // 6)
household_ids = rng.integers(1, n_household + 1, size=N)
household_district = pd.Series(DISTRICTS)[rng.integers(0, len(DISTRICTS), n_household + 1)].values
household_income = np.clip(rng.lognormal(mean=9.9, sigma=0.55, size=n_household + 1), 8000, 250000)
household_vehicle = rng.choice([0, 1, 2, 3], size=n_household + 1, p=[0.28, 0.46, 0.20, 0.06])
household_size = rng.choice([1, 2, 3, 4, 5, 6], size=n_household + 1, p=[0.15, 0.30, 0.24, 0.18, 0.09, 0.04])

df = pd.DataFrame({
    "kisi_no": np.arange(1, N + 1),
    "hane_no": household_ids,
})
df["ilce"] = household_district[df["hane_no"]]
df["hanehalki_geliri"] = household_income[df["hane_no"]] * rng.normal(1, 0.03, N)
df["arac_sayisi"] = household_vehicle[df["hane_no"]]
df["hanehalki_buyuklugu"] = household_size[df["hane_no"]]

df["cinsiyet"] = rng.choice(["Erkek", "Kadin"], size=N, p=[0.501, 0.499])
age = rng.gamma(shape=4.2, scale=9.5, size=N)
df["yas"] = np.clip(age, 0, 95).round(0).astype(int)
df["egitim_duzeyi"] = rng.choice(EDUCATION, size=N, p=edu_weights)
df["calisma_durumu"] = rng.choice(EMPLOYMENT, size=N, p=emp_weights)

df["yolculuk_amaci"] = rng.choice(PURPOSES, size=N, p=purpose_weights)
df["ulasim_turu"] = rng.choice(MODES, size=N, p=mode_weights)

# amaca gore sure dagilimi biraz farklilassin
base_duration = rng.gamma(shape=2.3, scale=9, size=N)
purpose_factor = df["yolculuk_amaci"].map({
    "Ev Uclu Is": 1.15, "Ev Uclu Okul": 0.7, "Ev Uclu Universite": 1.05,
    "Ev Uclu Diger": 0.9, "Ev Uclu Olmayan": 0.8,
}).values
df["yolculuk_suresi_dk"] = np.clip(base_duration * purpose_factor, 2, 150).round(1)

# saatlik dagilim: amaca gore zirve saatler
def sample_hour(purpose):
    if purpose == "Ev Uclu Is":
        return rng.choice([7, 8, 9, 17, 18, 19], p=[0.18, 0.30, 0.12, 0.10, 0.20, 0.10])
    if purpose in ("Ev Uclu Okul", "Ev Uclu Universite"):
        return rng.choice([7, 8, 9, 15, 16], p=[0.20, 0.40, 0.15, 0.15, 0.10])
    if purpose == "Ev Uclu Olmayan":
        return rng.choice([11, 12, 13, 14], p=[0.15, 0.35, 0.30, 0.20])
    return rng.integers(9, 21)

df["baslangic_saati"] = [f"{sample_hour(p):02d}:{rng.integers(0,60):02d}" for p in df["yolculuk_amaci"]]
end_hour = (pd.to_numeric(df["baslangic_saati"].str.slice(0, 2)) +
            (df["yolculuk_suresi_dk"] // 60)).clip(0, 23).astype(int)
df["bitis_saati"] = [f"{h:02d}:{rng.integers(0,60):02d}" for h in end_hour]

out_path = "sample_uap2040_style_data.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Yazildi: {out_path} ({len(df):,} satir, {df.shape[1]} sutun)")
