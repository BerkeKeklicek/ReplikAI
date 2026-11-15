"""Gerçek zamanlı Otomatik Altyazı Radar -> Turbo OCR -> Önbellek -> AI Çeviri iş hattı."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, List, Optional, Tuple

from replikai.config import AppConfig
from replikai.core.cache import TranslationCache, is_similar_text, normalize_ocr_text
from replikai.core.capture import ScreenCapturer
from replikai.core.dialogue_grouper import DialogueTextGrouper
from replikai.core.game_window import (
    auto_detect_running_game,
    is_escape_key_pressed,
    is_game_active_in_foreground,
    map_region_to_game_window,
    set_process_gaming_friendly_priority,
)
from replikai.core.hud_radar import HUDTextPatch, UniversalHUDRadar
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
    Sıfır FPS-Drop Çift Kanal (Dual-Thread) Otomatik Oyun Altyazı & HUD Takipçisi:
    1. `_loop` (Altyazı Kanalı): Sadece küçük altyazı şeridini (~130ms) izler; HUD taraması
       tarafından ASLA bloklanmaz, 2 satırlık ardışık altyazıları bile anında yakalar.
    2. `_hud_loop` (Arka Plan Oyun HUD/Menü Kanalı): Oyun penceresine müdahale etmeden,
       FPS düşürmeden telefon (`DIALING...` -> `CONNECTED`), sol üst bildirim ve ESC menülerini
       sıkı harf sınırlarıyla (`tight_bounds`) yerinde çevirir.
    """

    def __init__(
        self,
        config: AppConfig,
        on_translation: Optional[Callable[[TranslationEvent], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_hud_patches: Optional[Callable[[List[HUDTextPatch]], None]] = None,
    ) -> None:
        self.config = config
        self.on_translation = on_translation
        self.on_status = on_status
        self.on_clear = on_clear
        self.on_hud_patches = on_hud_patches

        self.capturer = ScreenCapturer()
        self.ocr = OCREngine()
        self.cache = TranslationCache()
        self.translator = GameTranslator(config)
        self.hud_radar = UniversalHUDRadar()

        self.target_hwnd: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hud_thread: Optional[threading.Thread] = None
        self._last_ocr_text: str = ""
        self._prev_clause_text: str = ""
        self._prev_clause_time: float = 0.0
        self._empty_ticks: int = 0
        self._idle_cycles: int = 0
        self._subtitle_shown: bool = False
        self._last_esc_time: float = 0.0
        self._sub_lock = threading.Lock()
        self._hud_lock = threading.Lock()
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

    def _auto_detect_game_hwnd(self) -> None:
        """
        Çalışan oyunu (GTA5, RDR2 vb.) sadece HWND olarak tespit eder;
        oyunun pencere modunu veya çözünürlüğünü ASLA kendi kendine değiştirmez!
        """
        detected = auto_detect_running_game()
        if not detected:
            return
        hwnd, _title, exe_name = detected
        if self.target_hwnd != hwnd:
            self.target_hwnd = hwnd
            if self.on_status:
                self.on_status(f"Oyun Algılandı: {exe_name}")

    def start(self) -> None:
        if self._running:
            return
        set_process_gaming_friendly_priority()
        self._auto_detect_game_hwnd()
        self._running = True
        self._empty_ticks = 0
        self._idle_cycles = 0
        self.capturer.reset_state()
        self.hud_radar.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._hud_thread = threading.Thread(target=self._hud_loop, daemon=True)
        self._thread.start()
        self._hud_thread.start()
        if self.on_status:
            self.on_status("Otomatik Altyazı & Oyun HUD Takibi Aktif")

    def stop(self) -> None:
        self._running = False
        self.hud_radar.reset()
        if self.on_clear:
            self.on_clear()
        if self.on_hud_patches:
            self.on_hud_patches([])
        if self.on_status:
            self.on_status("Duraklatıldı")

    def toggle(self) -> bool:
        if self._running:
            self.stop()
        else:
            self.start()
        return self._running

    def _translate_preserving_speaker(self, query_text: str) -> Tuple[str, str, bool, float]:
        """
        OverTranslate `SpeakerLineStart` / `SpeakerLabel` korumalı çeviri:
        Metnin başında konuşmacı adı varsa (`"Michael: What are you doing?"`),
        sadece konuşma gövdesini çevirir ve konuşmacı adını koruyarak birleştirir.
        """
        speaker, body = DialogueTextGrouper.split_speaker_prefix(query_text)
        target_body = body if speaker else query_text

        cached = self.cache.get(
            target_body,
            target_lang=self.config.target_lang,
            genre=self.config.game_genre,
        )
        if cached:
            final_tr = f"{speaker}: {cached}" if speaker else cached
            return final_tr, "SQLite Önbellek ⚡", True, 0.0

        t_tr_start = time.perf_counter()
        translated_body, provider_label = self.translator.translate(target_body)
        tr_ms = (time.perf_counter() - t_tr_start) * 1000.0
        if not translated_body or provider_label == "tr_ignored":
            return "", provider_label, False, tr_ms

        if not is_similar_text(target_body, translated_body, threshold=0.84):
            self.cache.set(
                source_text=target_body,
                translated_text=translated_body,
                provider=provider_label,
                target_lang=self.config.target_lang,
                genre=self.config.game_genre,
            )

        final_tr = f"{speaker}: {translated_body}" if speaker else translated_body
        return final_tr, provider_label, False, tr_ms

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
            self._idle_cycles += 1
            if self._empty_ticks >= 12 and self._subtitle_shown:
                self._subtitle_shown = False
                self._last_ocr_text = ""
                self._prev_clause_text = ""
                if self.on_clear:
                    self.on_clear()
            return None

        if img is None:
            self._idle_cycles += 1
            return None

        self._idle_cycles = 0
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

        # Dinamik Kurgulu Cümle Birleştirici ("Whoever wins this" -> "let's team up")
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
            self._prev_clause_text = ""
            self._prev_clause_time = 0.0
        else:
            self._prev_clause_text = cleaned
            self._prev_clause_time = now_ts

        translated, provider_label, from_cache, tr_ms = self._translate_preserving_speaker(query_text)
        if not translated or provider_label == "tr_ignored":
            return None

        if not force and is_similar_text(query_text, translated, threshold=0.84):
            return None

        self._subtitle_shown = True
        latency = (time.perf_counter() - t0) * 1000.0
        event = TranslationEvent(
            source_text=query_text,
            translated_text=translated,
            provider=provider_label,
            from_cache=from_cache,
            latency_ms=latency,
            ocr_ms=ocr_ms,
            translate_ms=tr_ms,
        )
        if self.on_translation:
            self.on_translation(event)
        return event

    def process_game_hud_frame(self, force_full_menu: bool = False) -> List[HUDTextPatch]:
        """
        3 Filtreli + OverTranslate Doğal Doku Yamalı Evrensel Oyun HUD, Bildirim, Telefon ve ESC Menü Çeviricisi.
        Ayrı arka plan iş parçacığında (`_hud_loop`) çalışır; altyazı çevirisini ve oyun FPS'ini ASLA etkilemez!
        """
        with self._hud_lock:
            if not force_full_menu:
                if not self.config.enable_game_hud_radar or not is_game_active_in_foreground(self.target_hwnd):
                    if self.hud_radar.active_patches:
                        self.hud_radar.reset()
                        if self.on_hud_patches:
                            self.on_hud_patches([])
                    return []

            full_rgb = self.capturer.capture_fullscreen_rgb()
            eff_sub_reg = map_region_to_game_window(self.config.capture_region, self.target_hwnd)
            max_cands = 14 if force_full_menu else 3
            candidates, patches_changed = self.hud_radar.scan_for_new_hud_boxes(
                full_rgb=full_rgb,
                subtitle_region=eff_sub_reg,
                force_full_menu=force_full_menu,
                max_candidates=max_cands,
            )

            for box in candidates:
                raw_hud, tight_bounds, num_lines = self.ocr.extract_hud_crop_text_and_bounds(box.crop_rgb)
                if not raw_hud:
                    self.hud_radar.lock_dead_zone(box)
                    continue

                # 1. Anlık 0.0 ms Oyun Menü & Telefon Sözlüğü ("MAP" -> "HARİTA", "DIALING..." -> "ARANIYOR..." vb.)
                instant_tr = self.hud_radar.lookup_instant_glossary(raw_hud)
                if instant_tr:
                    self.hud_radar.register_patch(
                        box,
                        raw_hud,
                        instant_tr,
                        tight_bounds=tight_bounds,
                        num_lines=num_lines,
                    )
                    patches_changed = True
                    continue

                # 2. Genel HUD / Telefon / İpucu cümlesi kontrolü
                cleaned = normalize_ocr_text(raw_hud)
                if not cleaned:
                    self.hud_radar.lock_dead_zone(box)
                    continue

                translated, prov, _from_cache, _tr_ms = self._translate_preserving_speaker(cleaned)
                if not translated or prov == "tr_ignored" or is_similar_text(cleaned, translated, threshold=0.86):
                    self.hud_radar.lock_dead_zone(box)
                    continue

                self.hud_radar.register_patch(
                    box,
                    cleaned,
                    translated,
                    tight_bounds=tight_bounds,
                    num_lines=num_lines,
                )
                patches_changed = True

            active_list = list(self.hud_radar.active_patches.values())
            if patches_changed and self.on_hud_patches:
                self.on_hud_patches(active_list)
            return active_list

    def _loop(self) -> None:
        """Hızlı, Sıfır-Blokajlı Altyazı Kanalı."""
        tick_counter = 0
        while self._running:
            t_start = time.perf_counter()
            tick_counter += 1
            if tick_counter % 25 == 1:
                self._auto_detect_game_hwnd()
            if tick_counter % 65 == 0:
                threading.Thread(target=self.translator.warmup, daemon=True).start()
            try:
                with self._sub_lock:
                    self.process_single_frame(force=False)
            except Exception as exc:
                if self.on_status:
                    self.on_status(f"Hata: {exc}")

            elapsed = time.perf_counter() - t_start
            base_interval = max(0.12, float(self.config.scan_interval_ms) / 1000.0)
            if self._idle_cycles >= 20:
                adaptive_interval = min(0.24, base_interval * 1.3)
            else:
                adaptive_interval = base_interval
            sleep_for = max(0.03, adaptive_interval - elapsed)
            time.sleep(sleep_for)

    def _hud_loop(self) -> None:
        """
        Ayrı Arka Plan Oyun HUD & ESC Menü Kanalı:
        Altyazı kanalını bloklamaz ve 0.9 saniyelik yumuşak periyotla çalışarak
        tam ekran 3D oyunlarda %0 FPS düşüşü sağlar.
        """
        while self._running:
            t_start = time.perf_counter()
            try:
                if self.config.enable_game_hud_radar and is_game_active_in_foreground(self.target_hwnd):
                    now_t = time.perf_counter()
                    if (now_t - self._last_esc_time) > 1.0 and is_escape_key_pressed():
                        self._last_esc_time = now_t
                        time.sleep(0.20)
                        self.process_game_hud_frame(force_full_menu=True)
                    else:
                        self.process_game_hud_frame(force_full_menu=False)
                elif self.hud_radar.active_patches:
                    self.hud_radar.reset()
                    if self.on_hud_patches:
                        self.on_hud_patches([])
            except Exception:
                pass

            elapsed = time.perf_counter() - t_start
            # ESC tuşunu hızlı yakalamak için kısa adımlarla ~0.85s bekle
            remaining = max(0.20, 0.85 - elapsed)
            steps = max(1, int(remaining / 0.10))
            for _ in range(steps):
                if not self._running:
                    break
                if (time.perf_counter() - self._last_esc_time) > 1.0 and is_escape_key_pressed():
                    break
                time.sleep(0.10)
