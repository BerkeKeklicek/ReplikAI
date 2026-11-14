"""Gerçek zamanlı Otomatik Altyazı Radar -> Turbo OCR -> Önbellek -> AI Çeviri iş hattı."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional, Set

from replikai.config import AppConfig
from replikai.core.cache import TranslationCache, is_similar_text, normalize_ocr_text
from replikai.core.capture import ScreenCapturer
from replikai.core.game_window import (
    auto_detect_running_game,
    make_window_borderless_fullscreen,
    set_process_gaming_friendly_priority,
)
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
    """
    Sıfır ayar gerektiren otomatik oyun altyazı takipçisi:
    - Oyun açıldığında otomatik algılar ve DWM uyumlu çerçevesiz moda geçirir.
    - Altyazı çıktığında ~35ms içinde okur ve çevirir.
    - Konuşma bittiğinde (altyazı ekrandan kalktığında) Türkçe altyazıyı otomatik gizler.
    """

    def __init__(
        self,
        config: AppConfig,
        on_translation: Optional[Callable[[TranslationEvent], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.on_translation = on_translation
        self.on_status = on_status
        self.on_clear = on_clear

        self.capturer = ScreenCapturer()
        self.ocr = OCREngine()
        self.cache = TranslationCache()
        self.translator = GameTranslator(config)

        self.target_hwnd: Optional[int] = None
        self._borderless_applied_hwnds: Set[int] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ocr_text: str = ""
        self._prev_clause_text: str = ""
        self._prev_clause_time: float = 0.0
        self._empty_ticks: int = 0
        self._subtitle_shown: bool = False
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmed_up = False

        threading.Thread(target=self.warmup, daemon=True).start()

    def warmup(self) -> None:
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

    def _auto_prepare_game_window(self) -> None:
        """Çalışan bir oyun (GTA5, RDR2 vb.) varsa kullanıcıya sormadan otomatik hazırlar."""
        detected = auto_detect_running_game()
        if not detected:
            return
        hwnd, _title, exe_name = detected
        self.target_hwnd = hwnd
        if hwnd not in self._borderless_applied_hwnds:
            ok, _ = make_window_borderless_fullscreen(hwnd)
            if ok:
                self._borderless_applied_hwnds.add(hwnd)
                if self.on_status:
                    self.on_status(f"Otomatik Bağlandı: {exe_name}")

    def start(self) -> None:
        if self._running:
            return
        set_process_gaming_friendly_priority()
        self._auto_prepare_game_window()
        self._running = True
        self._empty_ticks = 0
        self.capturer.reset_state()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.on_status:
            self.on_status("Otomatik Altyazı Takibi Aktif")

    def stop(self) -> None:
        self._running = False
        if self.on_clear:
            self.on_clear()
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

        # Ekranda uzun süre (>= 12 kare / ~2.5 saniye) hiç altyazı kalmadıysa Türkçe altyazıyı gizle
        if not force and not self.capturer.subtitle_currently_visible:
            self._empty_ticks += 1
            if self._empty_ticks >= 12 and self._subtitle_shown:
                self._subtitle_shown = False
                self._last_ocr_text = ""
                self._prev_clause_text = ""
                if self.on_clear:
                    self.on_clear()
            return None

        if img is None:
            return None

        t0 = time.perf_counter()
        t_ocr_start = t0
        raw_text = self.ocr.extract_text(img)
        ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

        cleaned = normalize_ocr_text(raw_text)
        if not cleaned:
            return None

        self._empty_ticks = 0
        if not force and is_similar_text(cleaned, self._last_ocr_text, threshold=0.88):
            return None

        # Dinamik Kurgulu Cümle Birleştirici ("Whoever wins this" -> "let's team up"):
        # Sadece ilk kart yarım bir yan cümleyse veya ikinci kart küçük harfle / "let's" ile devam ediyorsa birleştir;
        # birleştirme yapıldıktan sonra zincirleme 3. cümlenin yapışmaması için _prev_clause_text'i sıfırla!
        now_ts = time.perf_counter()
        query_text = cleaned
        is_joined_clause = False
        prev_clean = self._prev_clause_text.strip()
        if (
            prev_clean
            and (0.10 <= (now_ts - self._prev_clause_time) <= 3.4)
            and len(prev_clean) <= 52
            and len(cleaned) <= 52
            and not prev_clean.endswith((".", "!", "?"))
        ):
            prev_low = prev_clean.lower()
            cur_low = cleaned.lower()
            is_extension = cur_low.startswith(prev_low[: min(7, len(prev_low))]) or prev_low.startswith(
                cur_low[: min(7, len(cur_low))]
            )
            subordinate_starts = (
                "whoever",
                "whatever",
                "whenever",
                "wherever",
                "if ",
                "when ",
                "while ",
                "unless ",
                "until ",
                "because ",
                "every time",
                "as soon as",
            )
            dangling_ends = (
                ",",
                " to",
                " with",
                " for",
                " of",
                " in",
                " on",
                " and",
                " or",
                " that",
                " the",
                " a",
            )
            is_continuation = (
                prev_low.startswith(subordinate_starts)
                or prev_low.endswith(dangling_ends)
                or cleaned[0].islower()
                or cur_low.startswith(("let's", "lets"))
            )
            if not is_extension and is_continuation:
                query_text = f"{prev_clean.rstrip(',')}, {cleaned}"
                is_joined_clause = True

        self._last_ocr_text = cleaned
        if is_joined_clause:
            # İki yarım kart birleştirildikten sonra 3. kartın yanına yapışmaması için sıfırla
            self._prev_clause_text = ""
            self._prev_clause_time = 0.0
        else:
            self._prev_clause_text = cleaned
            self._prev_clause_time = now_ts

        # 1. Önbellek kontrolü (0ms)
        cached = self.cache.get(
            query_text,
            target_lang=self.config.target_lang,
            genre=self.config.game_genre,
        )
        if cached:
            if not force and is_similar_text(query_text, cached, threshold=0.84):
                return None
            self._subtitle_shown = True
            latency = (time.perf_counter() - t0) * 1000.0
            event = TranslationEvent(
                source_text=query_text,
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
        translated, provider_label = self.translator.translate(query_text)
        tr_ms = (time.perf_counter() - t_tr_start) * 1000.0
        if not translated or provider_label == "tr_ignored":
            return None

        # Eğer çeviri orijinal metinle aynıysa (zaten Türkçe veya özel isimse) ekrana basma
        if not force and is_similar_text(query_text, translated, threshold=0.84):
            return None

        self._subtitle_shown = True

        self.cache.set(
            source_text=query_text,
            translated_text=translated,
            provider=provider_label,
            target_lang=self.config.target_lang,
            genre=self.config.game_genre,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        event = TranslationEvent(
            source_text=query_text,
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
        tick_counter = 0
        while self._running:
            t_start = time.perf_counter()
            tick_counter += 1
            if tick_counter % 15 == 1:
                self._auto_prepare_game_window()
            if tick_counter % 55 == 0:
                threading.Thread(target=self.translator.warmup, daemon=True).start()
            try:
                with self._lock:
                    self.process_single_frame(force=False)
            except Exception as exc:
                if self.on_status:
                    self.on_status(f"Hata: {exc}")
            elapsed = time.perf_counter() - t_start
            interval_sec = max(0.18, float(self.config.scan_interval_ms) / 1000.0)
            sleep_for = max(0.05, interval_sec - elapsed)
            time.sleep(sleep_for)
