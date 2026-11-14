# 🎮 ReplikAI — Yapay Zeka Destekli Gerçek Zamanlı Oyun Çeviri Motoru

> **Güvensiz `.exe` yamalarına ve virüs riskine son!**  
> **ReplikAI**, oyun dosyalarına hiçbir şekilde müdahale etmeden (Anti-Cheat %100 güvenli), ekrandaki altyazı, diyalog ve menüleri gerçek zamanlı olarak yakalayan, oyun bağlamına uygun şekilde yapay zeka ile Türkçeye çeviren ve tıklamaları engellemeyen **Hayalet Katman (Click-Through Ghost Overlay)** üzerinde gösteren açık kaynaklı bir masaüstü uygulamasıdır.

---

## 🚀 Neden ReplikAI?

Avrupa ve Asya merkezli birçok oyun stüdyosu oyunlarına resmi Türkçe dil desteği eklemiyor. Oyuncular ise:
1. Yarım yamalak yabancı dille hikayeyi kaçırarak oynamak zorunda kalıyor,
2. İnternetteki denetimsiz forum ve sitelerden `.exe` uzantılı yama indirirken bilgisayarlarına **trojan / zararlı yazılım** bulaştırıyor,
3. Oyun dosyaları değiştirildiği için **Online / Anti-Cheat** korumalı oyunlarda ban riskiyle karşılaşıyor.

**ReplikAI** bu sorunu kökten çözer:
- 🛡️ **Sıfır Dosya Müdahalesi:** Oyun dosyalarına dokunmaz, RAM okumaz, sadece ekran piksellerini okur.
- 🧠 **Bağlamsal Yapay Zeka Çevirisi:** Kelime kelime robotik çeviri yerine, oynadığınız oyunun türüne (RPG, Karanlık Fantezi, Bilim Kurgu, Aksiyon vb.) ve önceki diyalog geçmişine göre doğal Türkçe çeviri üretir.
- ⚡ **Akıllı Piksel Fark Algılama & SQLite Önbellek:** Altyazı değişmediği sürece işlemci veya API kotası harcamaz. Daha önce çevrilen bir cümle tekrar ekrana geldiğinde **0 ms** gecikmeyle yerel önbellekten (`SQLite`) getirir.
- 👻 **Hayalet Overlay (Click-Through):** Altyazı katmanı oyunun üstünde durur ancak fare tıklamalarınız doğrudan arkadaki oyuna geçer (`WS_EX_TRANSPARENT`).

---

## ✨ Temel Özellikler

| Özellik | Açıklama |
| :--- | :--- |
| **🎯 İnteraktif Bölge Seçici** | Fare ile ekranda altyazı veya diyalog kutusunun çıktığı alanı saniyeler içinde seçin (`Alt + R`). |
| **🔍 Çift Katmanlı Hızlı OCR** | Windows yerleşik **WinRT Native OCR** (ekstra kurulum gerektirmez, ultra hızlı) ve **RapidOCR / Tesseract** desteği. |
| **🤖 Çoklu AI Motoru** | **Google Gemini** (`gemini-2.5-flash`), **OpenAI** (`gpt-4o-mini`), **Yerel Ollama** (`llama3`, `gemma2`, `qwen2.5` - %100 ücretsiz & çevrimdışı) ve **Ücretsiz Hızlı Web Çeviri** yedek katmanı. |
| **🎭 Oyun Türü & Karakter Tonu** | RPG, Bilim Kurgu, Korku, Orta Çağ veya Genel modlarıyla oyunun atmosferine uygun çeviri. |
| **💾 Kalıcı SQLite Önbellek** | Çevrilen her replik yerel veritabanına kaydedilir; aynı diyalog veya menü tekrar açıldığında anında gösterilir. |
| **🎨 Özelleştirilebilir Arayüz** | Yazı boyutu, arka plan saydamlığı, metin rengi ve hayalet mod (tıklama geçirgenliği) anlık ayarlanabilir. |

---

## 🏗️ Mimari Yapı

```
ReplikAI/
├── main.py                      # Uygulama giriş noktası
├── requirements.txt             # Python bağımlılıkları
├── replikai/
│   ├── __init__.py
│   ├── config.py                # Ayar yönetimi (JSON tabanlı kalıcı yapılandırma)
│   ├── core/
│   │   ├── capture.py           # MSS tabanlı ultra hızlı ekran yakalama & piksel değişim algılayıcı
│   │   ├── ocr_engine.py        # Windows Native OCR + RapidOCR/Tesseract hibrit metin okuma motoru
│   │   ├── translator.py        # Gemini, OpenAI, Ollama ve Ücretsiz Fallback destekli AI çeviri motoru
│   │   ├── cache.py             # SQLite tabanlı sıfır gecikmeli çeviri önbelleği
│   │   └── pipeline.py          # Yakalama -> OCR -> Önbellek -> AI Çeviri -> Overlay asenkron iş hattı
│   └── ui/
│       ├── dashboard.py         # Modern karanlık tema kontrol paneli & canlı çeviri günlüğü
│       ├── overlay.py           # Win32 Click-Through destekli Hayalet Altyazı Katmanı
│       └── region_selector.py   # Tam ekran yarı saydam interaktif OCR bölge seçici
└── tests/
    └── test_core.py             # Önbellek, metin temizleme ve çeviri motoru birim testleri
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/BerkeKeklicek/ReplikAI.git
cd ReplikAI
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
python main.py
```

---

## ⌨️ Kısayollar ve Kullanım

1. **Bölge Seç (`Alt + R`):** Oyundaki altyazı alanını fareyle sürükleyerek seçin.
2. **Çeviriyi Başlat / Durdur (`Alt + T`):** Canlı yakalama ve çeviri motorunu açıp kapatın.
3. **Hayalet Mod (`Alt + L`):** Overlay penceresini kilitleyin (fare tıklamaları doğrudan oyuna geçer) veya kilidi açıp sürükleyin.
4. **Anlık Tek Kare Çevir (`Alt + C`):** Sürekli tarama kapalıyken ekrandaki metni tek tuşla anında çevirin.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında açık kaynaklı olarak sunulmaktadır.
