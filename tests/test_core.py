"""ReplikAI çekirdek modülleri birim ve entegrasyon testleri."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from replikai.config import AppConfig
from replikai.core.cache import TranslationCache, is_similar_text, normalize_ocr_text
from replikai.core.ocr_engine import OCREngine
from replikai.core.translator import GameTranslator


def test_normalize_and_similarity() -> None:
    assert normalize_ocr_text("  Hello\n\nworld!  ") == "Hello world!"
    assert normalize_ocr_text("...") == ""
    assert is_similar_text("The gate is locked.", "The gate is locked!")
    assert not is_similar_text("The gate is locked.", "Open your inventory now.")


def test_sqlite_translation_cache(tmp_path: Path) -> None:
    db_file = tmp_path / "test_cache.db"
    cache = TranslationCache(db_path=db_file)
    assert cache.count() == 0

    cache.set(
        source_text="War never changes.",
        translated_text="Savaş asla değişmez.",
        provider="Gemini",
        target_lang="Türkçe",
        genre="RPG / Hikaye Odaklı",
    )
    assert cache.count() == 1

    hit = cache.get(
        "war never changes.",
        target_lang="Türkçe",
        genre="RPG / Hikaye Odaklı",
    )
    assert hit == "Savaş asla değişmez."
    assert cache.hits == 1


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = AppConfig(provider="gemini", overlay_font_size=24)
    cfg.save(cfg_path)

    loaded = AppConfig.load(cfg_path)
    assert loaded.provider == "gemini"
    assert loaded.overlay_font_size == 24


def test_ocr_engine_reads_game_subtitle() -> None:
    img = Image.new("RGB", (680, 110), color=(15, 18, 24))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((28, 36), "The northern kingdom has fallen.", fill=(255, 235, 140), font=font)

    ocr = OCREngine()
    extracted = ocr.extract_text(img)
    assert "northern" in extracted.lower() or "kingdom" in extracted.lower()


def test_translator_prompt_and_fallback() -> None:
    cfg = AppConfig(provider="free_fast", game_genre="Karanlık Fantezi / Orta Çağ")
    translator = GameTranslator(cfg)
    prompt = translator.build_system_prompt()
    assert "Karanlık Fantezi" in prompt

    translated, provider = translator.translate("Stay awhile and listen!")
    assert len(translated) > 2
    assert provider != "none"
