# UAP 2040 Tarzi Ulasim Veri Dashboard'u

Izmir Ulasim Ana Plani 2040 (UAP 2040) Hanehalki Ulasim Arastirmasi raporunun
(`KSR-045-UAP2040_YENI_BILGILER_RAPORU_CILT 1_R01-17-104.pdf`) analiz mantigini
sizin kendi ham CSV verinize otomatik uygulayan bir Streamlit dashboard'u.

## v2 mimarisi (buyuk/karmasik veri icin)

- **Dosya asla pandas'a tam yuklenmez.** DuckDB, CSV'yi diskten okuyup bir
  KEZ kalici bir `.duckdb` tablosuna isler ("ingest"). Sonraki her analiz
  sorgusu bu tabloyu SQL ile agregasyonla sorgular; sonuclar HER ZAMAN
  dosyadaki TAM veriye gore hesaplanir (kucuk bir ornekleme uzerinden degil).
- **Kodlama guvenligi**: dosyanin gercek kod sayfasi (cp1254, iso-8859-9 vb.)
  Python ile tespit edilip DuckDB'ye verilmeden once UTF-8'e cevrilir (DuckDB
  cp1254 gibi Turkce kod sayfalarini dogrudan desteklemez ve sessizce
  karakterleri bozabilir).
- **Turkce sayi bicimi**: `"12,5"` ya da `"1.234,56"` gibi virgullu/nokta
  bicimli sayilar SQL icinde otomatik ayristirilir.
- **Onbellekleme**: ayni dosya (boyut+degisim zamanina gore) bir daha
  yeniden islenmez; ayrica her analiz sorgusu Streamlit `cache_data` ile
  onbelleklenir — sekmeler arasi gecis ya da tekrar acilislar anlik olur.
- **ML adimlari** (anomali tespiti, ozellik onemi) bellekte calismak
  zorunda oldugundan DuckDB'nin reservoir-sample'iyla temsili bir ornek
  uzerinde calisir; arayuzde bu acikca "ornek/tam veri" olarak belirtilir.
  Kumeleme ve korelasyon gibi geri kalan her sey TAM VERI uzerindedir. Ayrica
  sayfa acilisinda otomatik CALISMAZLAR - "Anomali taramasini calistir" /
  "Ozellik onemini hesapla" butonlarina basinca calisirlar (hiz icin).
- **Numaralandirilmis sutun gruplari** (`aracadet1, aracadet2, ...` ya da
  `yas1, cinsiyet1, yas2, cinsiyet2, ...`) otomatik tespit edilir (bkz.
  `utils/column_groups.py`). Sol menudeki "🧩 Sutun Gruplari" bolumunden:
    - **Tek tabanli** diziler (orn. `aracadet1..3`) TOPLA/ORTALAMA/MAKS ile
      tek bir sayisal sutuna indirgenebilir.
    - **Birden fazla tabanin ayni index kumesini paylastigi** gruplar (orn.
      hanedeki N. kisinin yas/cinsiyet/meslek bilgileri) "uzun formata"
      (bir satir = bir tekrar, bos kayitlar otomatik elenir) cevrilebilir;
      olusan tablo sol menudeki "Aktif tablo" secicisinden analiz icin
      secilebilir.
- **Hata izolasyonu**: her tablo/grafik kendi try/except'i icinde calisir -
  bir bolum hata verirse SADECE o bolumde kucuk bir uyari gorunur, sayfanin
  geri kalani calismaya devam eder. Tum hatalarin dokumu "Genel Bakis" ve
  "Yapay Zeka Icgorulari" sekmelerindeki "🪲 Hatalar" acilir kutusunda
  goruntulenebilir.
- **Sutun eslestirme sadelestirildi**: 21 kavram artik tek uzun liste yerine
  5 kucuk, konuya gore gruba ayrilmis bolum halinde gosterilir (Kimlik/Bolge,
  Kisi & Demografi, Hane/Gelir/Arac, Yolculuk, Gelismis).

## Calistirma

```bash
cd dashboard
streamlit run app.py
```

Tarayicida `http://localhost:8501` acilir (port doluysa Streamlit farkli bir
port secer, terminaldeki adresi kullanin).

## Kullanim

1. **Sol menuden CSV yukleyin.**
   - "Dosya Yolu Gir" (onerilen): her boyutta dosya icin. Dosya hicbir zaman
     tarayiciya/belleğe tam yuklenmez; DuckDB diskten okuyup TAM VERIYI
     kalici bir tabloya isler (ilk seferde islenir, sonraki acilislar aninda).
   - "Dosya Yukle": Streamlit'in kendi yukleyicisi (`.streamlit/config.toml`
     ile 20 GB'a kadar acilmistir), kucuk/orta dosyalar icin pratiktir.
   - Bir sorun yasarsaniz "Onbellegi yoksay, yeniden isle" kutusunu isaretleyin
     ya da sidebar'daki "🗑️ Tum onbellegi temizle" butonunu kullanin.
2. **Sutun eslestirme**: Uygulama sutun adlarini otomatik tahmin etmeye
   calisir (`ilce`, `cinsiyet`, `yas`, `gelir`, `yolculuk_amaci` gibi anahtar
   kelimelerle). Sol menudeki "Sutunlari eslestir / duzelt" bolumunden
   tahminleri kontrol edip gerekirse duzeltin. Bir kavram icin uygun sutun
   yoksa "(Yok)" birakin — o bolumdeki grafik/tablo otomatik gizlenir.
3. **Sekmeler**:
   - 📊 Genel Bakis — veri profili, eksik veri, veri kalitesi.
   - 👥 Demografik Analiz — hane buyuklugu, cinsiyet/yas piramidi, egitim,
     istihdam, gelir/Gini/Lorenz, arac sahipligi (UAP 2040 referanslariyla
     karsilastirmali).
   - 🚗 Yolculuk Analizi — hareketlilik orani, amac/mod dagilimi, sure,
     saatlik dagilim, gelir-yolculuk iliskisi.
   - 🤖 Yapay Zeka Icgorulari — otomatik Turkce anlati, KMeans ile ilce
     profillemesi, IsolationForest ile anomali/veri kalitesi tespiti,
     korelasyon analizi, opsiyonel gercek LLM yorumu.
   - 📖 UAP 2040 Referans — rapordan alinan sabit referans degerleri.

## Yapay Zeka Katmani

- **Yerel (her zaman aktif, internet gerektirmez)**: `utils/ai_local.py`.
  Kural + istatistik tabanli Turkce anlati, scikit-learn `KMeans` ile bolge
  kumeleme, `IsolationForest` ile satir bazli anomali tespiti, korelasyon
  analizi.
- **Opsiyonel gercek LLM**: `utils/ai_llm.py`. Calismasi icin ortam
  degiskeni tanimlayin:

  ```powershell
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  # veya
  $env:OPENAI_API_KEY = "sk-..."
  ```

  Tanimliysa "Yapay Zeka Icgorulari" sekmesinde "LLM ile yorumla" butonu
  aktif olur. Yalnizca zaten hesaplanmis KUCUK ozet istatistikler LLM'e
  gonderilir; ham satirlar asla gonderilmez. API anahtarini kod icine
  yazmayin.

## Ornek Veri ile Test

Gercek CSV'niz hazir olmadan once dashboard'u denemek icin:

```bash
python make_sample_data.py 300000
```

Bu, UAP 2040 rapor yapisina benzer sentetik bir `sample_uap2040_style_data.csv`
uretir (300.000 satir). Ardindan dashboard'da bu dosyayi yukleyin.

## Dosya Yapisi

```
dashboard/
  app.py                  # Streamlit ana uygulama
  make_sample_data.py     # Test icin sentetik veri ureteci
  utils/
    mapping.py             # Sutun -> kavram otomatik eslestirme
    data_io.py              # Buyuk CSV'yi DuckDB ile bellek-dostu okuma
    analytics.py            # Rapor mantigini uygulayan hesaplamalar
    ai_local.py              # Yerel istatistik/ML tabanli AI motoru
    ai_llm.py                 # Opsiyonel gercek LLM baglantisi
    benchmarks.py             # UAP 2040 raporundan referans degerler
```
