"""Kalıcı uygulama yapılandırması ve Oyun Profilleri (API anahtarları yerel kalır)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.local.json"

GAME_PROFILES: Dict[str, Dict[str, str]] = {
    "Grand Theft Auto V (GTA 5 - Los Santos)": {
        "genre": "Modern Suç, Sokak & Aksiyon Draması (GTA V)",
        "style_guide": (
            "GTA 5 Los Santos atmosferine uygun, sokak dilini, argo ve esprileri robotikleştirmeden "
            "doğal ve akıcı bir Türkçeyle çevir. Michael, Franklin, Trevor, Lamar, Lester, Los Santos, "
            "Vinewood, Blaine County, FIB, LSPD gibi özel isimleri koru. 'Homie/Dog' -> 'dostum/kanka', "
            "'Cops/Heat' -> 'aynasızlar/polisler', 'Score/Heist' -> 'vurgun/soygun' gibi doğal karşılıklar kullan."
        ),
        "accent_color": "#2EC4B6",
        "subtitle_color": "#FFFFFF",
    },
    "Red Dead Redemption 2 (Vahşi Batı 1899)": {
        "genre": "1899 Vahşi Batı / Kanun Kaçağı Draması (Red Dead Redemption 2)",
        "style_guide": (
            "1899 yılı Amerikan Vahşi Batı atmosferine uygun, doğal ve sert dönem Türkçesi kullan. "
            "Arthur Morgan, Dutch, John, Sadie, Micah, Pinkerton, O'Driscoll, Valentine, Saint Denis, "
            "Blackwater, Dead Eye gibi özel isimleri çevirme. 'Partner' -> 'ortak/dostum', "
            "'Lawmen' -> 'kanun adamları', 'Bounty' -> 'kelle ödülü', 'Outlaw' -> 'kanun kaçağı', "
            "'Reckon' -> 'sanırım/hesabıma göre', 'Folks' -> 'millet/ahali' gibi atmosfere uygun karşılıklar seç."
        ),
        "accent_color": "#E63946",
        "subtitle_color": "#FFF3B0",
    },
    "RPG / Fantastik (BG3, Witcher, Elder Scrolls)": {
        "genre": "Karanlık Fantezi & Orta Çağ RPG",
        "style_guide": (
            "Orta çağ ve fantastik RPG atmosferine uygun, edebi ve akıcı bir Türkçe kullan. "
            "Şövalye, büyü, lonca, diyar, kadim metin dilini koru; modern sokak ağzı kullanma."
        ),
        "accent_color": "#D4AF37",
        "subtitle_color": "#FFE66D",
    },
    "Bilim Kurgu / Cyberpunk (Cyberpunk 2077, Starfield)": {
        "genre": "Bilim Kurgu & Siberpunk Aksiyon",
        "style_guide": (
            "Siberpunk ve gelecek temasına uygun, keskin, modern ve sokak/teknoloji terminolojisine "
            "hakim bir Türkçe kullan. Özel şirket, sibernetik ve şehir adlarını koru."
        ),
        "accent_color": "#00F5D4",
        "subtitle_color": "#00F5D4",
    },
    "Aksiyon / Hikaye Odaklı (GTA, Mafia, Uncharted)": {
        "genre": "Sinematik Aksiyon & Suç Draması",
        "style_guide": (
            "Doğal, akıcı, sinema filmi altyazısı kalitesinde konuşma dili kullan. "
            "Argo ve deyimleri birebir kelime çevirisi yerine Türkçe doğal karşılıklarıyla ver."
        ),
        "accent_color": "#3B82F6",
        "subtitle_color": "#FFFFFF",
    },
    "Korku / Gerilim (Resident Evil, Silent Hill, Alan Wake)": {
        "genre": "Psikolojik Korku & Hayatta Kalma",
        "style_guide": (
            "Gerilimli, ciddi ve karanlık atmosferi bozmayan bir Türkçe kullan. "
            "Bulmaca ipuçlarını ve notları anlam kaybı olmadan net çevir."
        ),
        "accent_color": "#9B5DE5",
        "subtitle_color": "#E2E8F0",
    },
}


@dataclass
class CaptureRegion:
    left: int = 260
    top: int = 790
    width: int = 1400
    height: int = 220

    def to_dict(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": max(100, int(self.width)),
            "height": max(40, int(self.height)),
        }


@dataclass
class AppConfig:
    # Yakalama ve OCR ayarları
    capture_region: Dict[str, int] = field(
        default_factory=lambda: CaptureRegion().to_dict()
    )
    scan_interval_ms: int = 550
    pixel_diff_threshold: float = 2.2
    preprocess_contrast: bool = True

    # Çeviri Motoru Ayarları
    # Seçenekler: "gemini", "ollama", "openai", "free_fast"
    provider: str = "free_fast"
    source_lang: str = "Auto"
    target_lang: str = "Türkçe"
    game_genre: str = "Grand Theft Auto V (GTA 5 - Los Santos)"

    # API Anahtarları ve Model İsimleri (config.local.json içinde yerel tutulur)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    # Overlay (Altyazı Katmanı) Ayarları
    overlay_font_size: int = 21
    overlay_opacity: float = 0.88
    overlay_text_color: str = "#FFF3B0"
    overlay_bg_color: str = "#0B0E14"
    overlay_click_through: bool = True
    show_original_text: bool = True

    # Kısayollar
    hotkey_toggle_scan: str = "alt+t"
    hotkey_select_region: str = "alt+r"
    hotkey_toggle_lock: str = "alt+l"
    hotkey_single_capture: str = "alt+c"
    hotkey_borderless: str = "alt+b"

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception:
            return cls()

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
