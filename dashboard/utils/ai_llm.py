# -*- coding: utf-8 -*-
"""
Opsiyonel gercek LLM baglantisi (Anthropic Claude ya da OpenAI).

Guvenlik/gizlilik notu: bu modul HAM SATIRLARI DEGIL, dashboard'da zaten
hesaplanmis KUCUK OZET istatistikleri (agregatlar) LLM'e gonderir. API
anahtari kod icine YAZILMAZ; yalnizca ortam degiskeninden (ANTHROPIC_API_KEY
veya OPENAI_API_KEY) okunur. Kullanici anahtari kendi ortaminda tanimlamalidir:

    setx ANTHROPIC_API_KEY "sk-ant-..."      (Windows, kalici)
    $env:ANTHROPIC_API_KEY = "sk-ant-..."    (PowerShell, oturum bazli)

Anahtar tanimli degilse bu modul sessizce devre disi kalir ve dashboard
yalnizca yerel (ai_local.py) motorla calismaya devam eder.
"""
from __future__ import annotations
import os
import json
import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = (
    "Sen bir ulasim planlamasi / hanehalki ulasim anketi veri analistisin. "
    "Sana verilen ozet istatistikleri, Izmir Ulasim Ana Plani 2040 (UAP 2040) raporundaki "
    "uslupla (kisa baslik + net, sayisal, karsilastirmali paragraflar) Turkce olarak yorumla. "
    "Sadece sana verilen sayilara dayan, veri uydurma. Cikti Markdown formatinda, "
    "kisa basliklar ve maddeler halinde olsun."
)


def available_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _call_anthropic(user_text: str, system: str = SYSTEM_PROMPT, max_tokens: int = 1500) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-5",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_text}],
    }
    r = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return "".join(block.get("text", "") for block in data.get("content", []))


def _call_openai(user_text: str, system: str = SYSTEM_PROMPT, max_tokens: int = 1500) -> str:
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _call_llm(user_text: str, system: str, max_tokens: int = 1500) -> str:
    """Hangi saglayici (Anthropic/OpenAI) tanimliysa onu kullanarak tek bir
    metin donduren ortak cagri noktasi - yorum uretme ve eslestirme onerisi
    gibi FARKLI gorevler ayni saglayici secim mantigini tekrar etmesin diye."""
    provider = available_provider()
    if provider is None:
        raise ValueError(
            "Ortam degiskenlerinde ANTHROPIC_API_KEY ya da OPENAI_API_KEY bulunamadi. "
            "LLM destekli ozellik icin bu degiskenlerden birini tanimlayin."
        )
    if provider == "anthropic":
        return _call_anthropic(user_text, system=system, max_tokens=max_tokens)
    return _call_openai(user_text, system=system, max_tokens=max_tokens)


def generate_llm_commentary(summary_dict: dict) -> str:
    """summary_dict: dashboard'da zaten hesaplanmis kucuk agregat sozluk.
    Donen deger Markdown metindir. API anahtari yoksa ValueError firlatir."""
    summary_text = (
        "Asagida bir hanehalki ulasim anketi CSV'sinden hesaplanmis ozet istatistikler var. "
        "Bunlari UAP 2040 rapor uslubunda yorumla:\n\n"
        + json.dumps(summary_dict, ensure_ascii=False, indent=2, default=str)
    )
    return _call_llm(summary_text, system=SYSTEM_PROMPT)


# =========================================================================
# SUTUN ESLESTIRME ONERISI (herhangi bir sehrin FARKLI isimlendirilmis CSV'si
# icin) - anahtar-kelime tabanli otomatik tahmin (mapping.auto_guess_mapping)
# bir sutun COK FARKLI/beklenmedik bir ad tasiyorsa (orn. "income_lvl",
# "hnhlk_kzn") bos kalabilir. Bu fonksiyon, SADECE sutun ADLARINI ve HER
# birinden birkac KUCUK ORNEK DEGERI (ham veri satirlarinin TAMAMI DEGIL)
# LLM'e gonderip "bu sutun hangi kavrama karsilik geliyor" tahmini ister.
# Tamamen OPSIYONELDIR (yalnizca kullanici butona basinca calisir), SADECE
# mevcut bos/eslesmemis kavramlari doldurmak icin kullanilir - var olan bir
# eslestirmeyi ASLA sessizce degistirmez (bkz. app.py'daki cagri noktasi).
MAPPING_SYSTEM_PROMPT = (
    "Sen hanehalki ulasim anketi (Turkiye'deki UAP 2040 tarzi il ulasim ana "
    "plani anketleri) CSV dosyalarinin sutun yapisini taniyan bir veri "
    "muhendisisin. Farkli sehirlerin anket dosyalari FARKLI sutun adlandirma "
    "kurallari kullanabilir (kisaltmalar, Ingilizce adlar, farkli diller). "
    "Gorevin: verilen KAVRAM listesindeki her kavram icin, verilen SUTUN "
    "listesinden (adi ve birkac ornek degeriyle) EN UYGUN olani secmek. "
    "SADECE verilen sutun adlarindan birini kullan, ASLA yeni bir ad uydurma. "
    "Emin degilsen ya da hicbir sutun uygun degilse o kavram icin null don. "
    "Cikti SADECE gecerli JSON olmali, baska hicbir aciklama/metin ekleme."
)


def suggest_column_mapping(candidates: dict[str, list[str]], concepts: dict) -> dict[str, str | None]:
    """candidates: {sutun_adi: [ornek_deger, ...]} - adaylarin adi ve
    (varsa) birkac gercek ornek degeri. concepts: mapping.CONCEPTS sozlugu
    (concept_key -> (etiket, aciklama, zorunlu_mu, tip, grup)).

    Doner: {concept_key: onerilen_sutun_adi ya da None}. Yalnizca
    `candidates` icinde GERCEKTEN var olan sutun adlari kabul edilir -
    LLM'in uydurabilecegi/yanlis yazabilecegi bir ad varsa o kavram icin
    None'a dusurulur (guvenlik/dogruluk icin). API anahtari yoksa
    ValueError firlatir."""
    concept_lines = [
        f"- {key} | etiket: {label} | aciklama: {desc}"
        for key, (label, desc, _req, _ctype, _grp) in concepts.items()
    ]
    col_lines = []
    for col, samples in candidates.items():
        sv = ", ".join(str(s) for s in samples[:5] if s not in (None, ""))
        col_lines.append(f"- {col}" + (f"  (ornek degerler: {sv})" if sv else ""))

    user_text = (
        "KAVRAMLAR (her biri icin en uygun sutunu sec):\n"
        + "\n".join(concept_lines)
        + "\n\nSUTUNLAR (aday listesi):\n"
        + "\n".join(col_lines)
        + "\n\nSADECE su formatta bir JSON nesnesi don (aciklama/metin EKLEME):\n"
        + '{"concept_key": "sutun_adi_veya_null", ...}'
    )
    raw = _call_llm(user_text, system=MAPPING_SYSTEM_PROMPT, max_tokens=2000)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {}

    result: dict[str, str | None] = {}
    for key in concepts:
        val = parsed.get(key)
        # LLM'in var olmayan/uydurma bir sutun adi donmesine karsi: sadece
        # GERCEKTEN aday listesinde olan degerler kabul edilir.
        if isinstance(val, str) and val in candidates:
            result[key] = val
    return result


def _extract_json(raw: str):
    """LLM yaniti bazen JSON'un disinda ek metin (orn. bir markdown kod
    blogu icinde donebilir) - ilk { ile son } arasini ayiklayip parse eder.
    Parse basarisiz olursa None doner (cagiran taraf bunu 'AI oneri
    uretemedi' olarak yorumlar - hicbir sey ZORLA uygulanmaz)."""
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# =========================================================================
# KOD -> ETIKET ONERISI: "Kod Cozme" panelindeki AYNI mekanizmayi (bkz.
# app.py decode_key) besler. Kategorik bir sutunda gozlenen HAM kodlarin
# (orn. cinsiyet icin 1/2, surucu/yolcu icin 1/2/0) GERCEK anlamini,
# genel bilgisiyle (bu tur anketlerde 1=Erkek/2=Kadin gibi YERLESIK
# kaliplar) tahmin eder. SADECE kullanici butona basinca calisir, SADECE
# GERCEKTEN gozlenen kodlar icin etiket doner (uydurma kod KABUL EDILMEZ),
# ve sonuc HER ZAMAN normal 'Kod Cozme' metin kutusunda goruntulenip
# degistirilebilir - hicbir sey sessizce/geri donusumsuz uygulanmaz.
CODE_LABEL_SYSTEM_PROMPT = (
    "Sen Turkiye'deki hanehalki ulasim anketlerinin (UAP 2040 tarzi) "
    "standart kodlama kaliplarini iyi bilen bir veri analistisin. Sana her "
    "biri icin KAVRAM ACIKLAMASI ve GOZLENEN SAYISAL/KISA KODLAR verilen "
    "sutunlar var. Her kod icin, bu tur anketlerde EN YAYGIN/mantikli "
    "karsiligi olan KISA Turkce etiketi tahmin et (orn. cinsiyet icin "
    "1=Erkek, 2=Kadin gibi). Emin degilsen o kod icin tahmin YAPMA (o kodu "
    "sonuca dahil etme) - yanlis/uydurma bir etiket vermek, hic etiket "
    "vermemekten KOTUDUR. SADECE verilen kodlari kullan, yeni kod uydurma. "
    "Cikti SADECE gecerli JSON olmali, baska hicbir aciklama/metin ekleme."
)


def suggest_code_labels(fields: dict[str, dict]) -> dict[str, dict[str, str]]:
    """fields: {sutun_adi: {"concept_label": str, "codes": [kod, ...]}}.

    Doner: {sutun_adi: {kod: etiket}} - SADECE `fields`'taki GERCEKTEN
    gozlenen kodlar icin (LLM'in uydurdugu bir kod varsa o REDDEDILIR).
    API anahtari yoksa ValueError firlatir."""
    field_lines = []
    for col, info in fields.items():
        codes_str = ", ".join(str(c) for c in info.get("codes", []))
        field_lines.append(f"- {col} | kavram: {info.get('concept_label', col)} | kodlar: [{codes_str}]")

    user_text = (
        "Asagidaki her sutun icin, gozlenen HER KOD'un en olasi Turkce "
        "karsiligini tahmin et:\n\n"
        + "\n".join(field_lines)
        + "\n\nSADECE su formatta bir JSON nesnesi don (aciklama/metin EKLEME):\n"
        + '{"sutun_adi": {"kod": "etiket", ...}, ...}'
    )
    raw = _call_llm(user_text, system=CODE_LABEL_SYSTEM_PROMPT, max_tokens=2000)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {}

    result: dict[str, dict[str, str]] = {}
    for col, info in fields.items():
        observed_codes = {str(c) for c in info.get("codes", [])}
        suggestion = parsed.get(col)
        if not isinstance(suggestion, dict):
            continue
        # DOGRULUK ICIN: SADECE gercekten gozlenen kodlar kabul edilir -
        # LLM'in uydurdugu/yanlis yazdigi bir kod varsa o tek tek elenir,
        # geri kalan gecerli kodlar yine de uygulanir.
        clean = {k: str(v) for k, v in suggestion.items()
                 if k in observed_codes and isinstance(v, str) and v.strip()}
        if clean:
            result[col] = clean
    return result
