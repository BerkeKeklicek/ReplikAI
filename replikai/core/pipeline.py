"""Gerçek zamanlı Yakalama -> 3D Altyazı Filtresi -> Turbo OCR -> Önbellek -> AI Çeviri iş hattı."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional

from replikai.config import AppConfig
from replikai.core.cache import TranslationCache, is_similar_text, normalize_ocr_text
from replikai.core.capture import ScreenCapturer
from replikai.core.game_window import auto_detect_running_game, set_process_gaming_friendly_priority
from replikai.core.ocr_engine import OCREngine
from replikai.core.translator import GameTranslator


@dataclass
class TranslationEvent:
    source_text: str
    translated_text: str
    provider: str
    from_cache: bool
    latency_ms: float
    ocr_ms: float = 0.0
    translate_ms: float = 0.0


class TranslationPipeline:
    """Arka planda çalışan, ön-ısıtmalı (warm-start) ve sıfır FPS-drop odaklı çeviri yöneticisi."""

    def __init__(
        self,
        config: AppConfig,
        on_translation: Optional[Callable[[TranslationEvent], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.on_translation = on_translation
        self.on_status = on_status

        self.capturer = ScreenCapturer()
        self.ocr = OCREngine()
        self.cache = TranslationCache()
        self.translator = GameTranslator(config)

        self.target_hwnd: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ocr_text: str = ""
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmed_up = False

        # Uygulama açılır açılmaz ONNX modelini ve HTTPS TLS soketini arka planda ısıt
        threading.Thread(target=self.warmup, daemon=True).start()

    def warmup(self) -> None:
        """İlk tıklamada 3000ms soğuk başlangıç yaşanmaması için motorları tek seferde önceden ısıtır."""
        if self._warmed_up:
            return
        with self._warmup_lock:
            if self._warmed_up:
                return
            self.ocr.warmup()
            self.translator.warmup()
            self._warmed_up = True

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        set_process_gaming_friendly_priority()
        if not self.target_hwnd:
            detected = auto_detect_running_game()
            if detected:
                self.target_hwnd = detected[0]
        self._running = True
        self.capturer.reset_state()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.on_status:
            self.on_status("Canlı Çeviri Aktif (Sıfır-Drop Modu)")

    def stop(self) -> None:
        self._running = False
        if self.on_status:
            self.on_status("Duraklatıldı")

    def toggle(self) -> bool:
        if self._running:
            self.stop()
        else:
            self.start()
        return self._running

    def process_single_frame(self, force: bool = True) -> Optional[TranslationEvent]:
        """Tek bir ekran karesini yakalar, okur ve çevirir."""
        t0 = time.perf_counter()
        img, _ = self.capturer.capture_region(
            region=self.config.capture_region,
            diff_threshold=self.config.pixel_diff_threshold,
            preprocess=self.config.preprocess_contrast,
            force=force,
            target_hwnd=self.target_hwnd,
        )
        if img is None:
            return None

        t_ocr_start = time.perf_counter()
        raw_text = self.ocr.extract_text(img)
        ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

        cleaned = normalize_ocr_text(raw_text)
        if not cleaned:
            return None

        if not force and is_similar_text(cleaned, self._last_ocr_text, threshold=0.88):
            return None

        self._last_ocr_text = cleaned

        # 1. Önbellek kontrolü (0ms çeviri süresi)
        cached = self.cache.get(
            cleaned,
            target_lang=self.config.target_lang,
            genre=self.config.game_genre,
        )
        if cached:
            latency = (time.perf_counter() - t0) * 1000.0
            event = TranslationEvent(
                source_text=cleaned,
                translated_text=cached,
                provider="SQLite Önbellek ⚡",
                from_cache=True,
                latency_ms=latency,
                ocr_ms=ocr_ms,
                translate_ms=0.0,
            )
            if self.on_translation:
                self.on_translation(event)
            return event

        # 2. Yapay Zeka Çevirisi
        t_tr_start = time.perf_counter()
        translated, provider_label = self.translator.translate(cleaned)
        tr_ms = (time.perf_counter() - t_tr_start) * 1000.0
        if not translated:
            return None

        self.cache.set(
            source_text=cleaned,
            translated_text=translated,
            provider=provider_label,
            target_lang=self.config.target_lang,
            genre=self.config.game_genre,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        event = TranslationEvent(
            source_text=cleaned,
            translated_text=translated,
            provider=provider_label,
            from_cache=False,
            latency_ms=latency,
            ocr_ms=ocr_ms,
            translate_ms=tr_ms,
        )
        if self.on_translation:
            self.on_translation(event)
        return event

    def _loop(self) -> None:
        while self._running:
            t_start = time.perf_counter()
            try:
                with self._lock:
                    self.process_single_frame(force=False)
            except Exception as exc:
                if self.on_status:
                    self.on_status(f"Hata: {exc}")
            elapsed = time.perf_counter() - t_start
            interval_sec = max(0.30, float(self.config.scan_interval_ms) / 1000.0)
            sleep_for = max(0.10, interval_sec - elapsed)
            time.sleep(sleep_for)
