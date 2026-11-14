"""Çoklu Yapay Zeka Destekli Bağlamsal Oyun Çeviri Motoru.

Desteklenen Motorlar:
1. Google Gemini API (gemini-2.5-flash vb.)
2. OpenAI API (gpt-4o-mini vb.)
3. Yerel Ollama (Ücretsiz / Çevrimdışı LLM)
4. Ücretsiz Hızlı Çeviri (Anahtarsız varsayılan & otomatik yedek katman)
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Deque, Optional, Tuple
import requests

from replikai.config import AppConfig, GAME_PROFILES
from replikai.core.cache import normalize_ocr_text

logger = logging.getLogger(__name__)


class GameTranslator:
    """Oyun türüne ve son diyalog bağlamına duyarlı çeviri motoru."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._context_history: Deque[Tuple[str, str]] = deque(maxlen=4)
        self._session = requests.Session()

    def build_system_prompt(self) -> str:
        profile_name = self.config.game_genre or "Red Dead Redemption 2 (Vahşi Batı 1899)"
        profile = GAME_PROFILES.get(profile_name, {})
        genre = profile.get("genre", profile_name)
        style_guide = profile.get(
            "style_guide",
            "Oyun atmosferine uygun, doğal ve akıcı bir Türkçe kullan.",
        )
        target = self.config.target_lang or "Türkçe"
        history_block = ""
        if self._context_history:
            lines = [f"- Orijinal: {src} -> Çeviri: {tr}" for src, tr in self._context_history]
            history_block = "\nÖnceki Diyalog Bağlamı:\n" + "\n".join(lines)

        return (
            f"Sen profesyonel bir AAA video oyunu yerelleştirme (localization) uzmanısın. "
            f"Oyun Profili: {genre}. Hedef Dil: {target}.\n"
            f"Üslup Rehberi: {style_guide}\n"
            f"Kurallar:\n"
            f"1. Ekrandan OCR ile okunan oyun altyazısını veya menü metnini {target} diline çevir.\n"
            f"2. Kelime kelime robotik çeviri yapma; karakterin duygu durumunu, oyun atmosferini ve doğal {target} deyimlerini kullan.\n"
            f"3. Özel karakter/yer/eşya isimlerini bozma, OCR kaynaklı ufak harf hatalarını bağlamdan onar.\n"
            f"4. SADECE çevrilmiş metni döndür. Açıklama, tırnak işareti veya ek yorum YAZMA."
            f"{history_block}"
        )

    def translate(self, text: str) -> Tuple[str, str]:
        """
        Metni seçili sağlayıcı ile çevirir.
        Dönüş: `(cevrilmis_metin, kullanilan_motor_adi)`
        """
        cleaned = normalize_ocr_text(text)
        if not cleaned:
            return "", "none"

        provider = (self.config.provider or "free_fast").lower()

        try:
            if provider == "gemini" and self.config.gemini_api_key.strip():
                result = self._translate_gemini(cleaned)
                if result:
                    self._record_context(cleaned, result)
                    return result, f"Gemini ({self.config.gemini_model})"

            elif provider == "openai" and self.config.openai_api_key.strip():
                result = self._translate_openai(cleaned)
                if result:
                    self._record_context(cleaned, result)
                    return result, f"OpenAI ({self.config.openai_model})"

            elif provider == "ollama":
                result = self._translate_ollama(cleaned)
                if result:
                    self._record_context(cleaned, result)
                    return result, f"Ollama ({self.config.ollama_model})"
        except Exception as exc:
            logger.warning("%s motorunda hata oluştu, yedek motora geçiliyor: %s", provider, exc)

        # Yedek veya varsayılan: Ücretsiz Hızlı Motor
        fallback = self._translate_free_fast(cleaned)
        if fallback:
            self._record_context(cleaned, fallback)
            return fallback, "Hızlı Çeviri (Ücretsiz)"

        return cleaned, "Hata"

    def _record_context(self, source: str, translated: str) -> None:
        self._context_history.append((source, translated))

    def _translate_gemini(self, text: str) -> Optional[str]:
        api_key = self.config.gemini_api_key.strip()
        model = self.config.gemini_model.strip() or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": self.build_system_prompt()}]},
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 256,
            },
        }
        resp = self._session.post(url, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        out = "".join(p.get("text", "") for p in parts).strip()
        return out or None

    def _translate_openai(self, text: str) -> Optional[str]:
        api_key = self.config.openai_api_key.strip()
        model = self.config.openai_model.strip() or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": text},
            ],
            "temperature": 0.25,
            "max_tokens": 256,
        }
        resp = self._session.post(url, headers=headers, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        out = choices[0].get("message", {}).get("content", "").strip()
        return out or None

    def _translate_ollama(self, text: str) -> Optional[str]:
        base_url = (self.config.ollama_url or "http://localhost:11434").rstrip("/")
        model = self.config.ollama_model.strip() or "gemma2:9b"
        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "system": self.build_system_prompt(),
            "prompt": text,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = self._session.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        out = data.get("response", "").strip()
        return out or None

    def _translate_free_fast(self, text: str) -> Optional[str]:
        """Anahtarsız, sıfır kurulum gerektiren ücretsiz hızlı çeviri katmanı."""
        lang_map = {
            "Türkçe": "tr",
            "İngilizce": "en",
            "Almanca": "de",
            "İspanyolca": "es",
            "Fransızca": "fr",
        }
        tl = lang_map.get(self.config.target_lang, "tr")
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": tl,
            "dt": "t",
            "q": text,
        }
        resp = self._session.get(url, params=params, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data[0], list):
            return None
        translated_chunks = [chunk[0] for chunk in data[0] if chunk and chunk[0]]
        return "".join(translated_chunks).strip() or None
