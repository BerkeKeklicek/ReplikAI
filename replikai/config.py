"""Kalıcı uygulama yapılandırması (API anahtarları yerel kalır, Git'e gönderilmez)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.local.json"


@dataclass
class CaptureRegion:
    left: int = 360
    top: int = 780
    width: int = 1200
    height: int = 200

    def to_dict(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": max(50, int(self.width)),
            "height": max(30, int(self.height)),
        }


@dataclass
class AppConfig:
    # Yakalama ve OCR ayarları
    capture_region: Dict[str, int] = field(
        default_factory=lambda: CaptureRegion().to_dict()
    )
    scan_interval_ms: int = 450
    pixel_diff_threshold: float = 2.5
    preprocess_contrast: bool = True

    # Çeviri Motoru Ayarları
    # Seçenekler: "gemini", "ollama", "openai", "free_fast"
    provider: str = "free_fast"
    source_lang: str = "Auto"
    target_lang: str = "Türkçe"
    game_genre: str = "RPG / Hikaye Odaklı"

    # API Anahtarları ve Model İsimleri (config.local.json içinde yerel tutulur)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    # Overlay (Altyazı Katmanı) Ayarları
    overlay_font_size: int = 20
    overlay_opacity: float = 0.85
    overlay_text_color: str = "#FFE66D"
    overlay_bg_color: str = "#111318"
    overlay_click_through: bool = True
    show_original_text: bool = False

    # Kısayollar
    hotkey_toggle_scan: str = "alt+t"
    hotkey_select_region: str = "alt+r"
    hotkey_toggle_lock: str = "alt+l"
    hotkey_single_capture: str = "alt+c"

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
