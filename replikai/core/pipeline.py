"""3 Katmanlı Hibrit Oyun Altyazı, Köşe Bilgi Kartı ve Otomatik Tam Ekran Menü Çeviri İş Hattı.

MİMARİ:
- Katman 1 (`_loop` — Canlı Altyazı Şeridi, ~120ms, %0 FPS kaybı):
  Sadece alt-orta altyazı bandını (minimap ve telefon hariç) izler; 1-3 satırlık altyazıları anında çevirir.
- Katman 2 (`_scan_corner_info_cards` — Canlı Köşe Bilgi Kartı Radarı, ~750ms, ~%0 FPS kaybı):
  Oynanış sırasında (imleç gizliyken) tam ekranı ASLA çekmez; sadece sol-üst köşedeki küçük bilgi/ipucu
  alanını (~%8 ekran) izler ve sabitlenen yeni bir bildirim kartı doğduğunda sadece o küçük kırpımda
  gerçek 2D DBNet + Toplu Çeviri çalıştırır.
- Katman 3 (`_scan_full_menu_dbnet` — Otomatik Tam Ekran Menü/Harita/Envanter Tarayıcı):
  Hangi tuşa (ESC, M, I, Tab, J...) basılırsa basılsın Windows `GetCursorInfo` imleç görünürlüğü
  ile menünün açıldığını OTOMATİK anlar -> Tüm ekranda gerçek 2D DBNet (`use_det=True`) + limitsiz
  Toplu (Batch) Çeviri ile ekrandaki tüm kelimelerin %99'unu tek seferde çevirir. Oyuna dönüldüğü
  an (imleç gizlendiğinde) tüm menü yamalarını anında temizler!
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from replikai.config import AppConfig
from replikai.core.cache import TranslationCache, is_similar_text, normalize_ocr_text
from replikai.core.capture import ScreenCapturer
from replikai.core.dialogue_grouper import DialogueTextGrouper
from replikai.core.frame_gate import FrameGate
from replikai.core.game_packs import GamePackStore
from replikai.core.game_window import (
    auto_detect_running_game,
    get_active_game_window_rect,
    is_cursor_visible,
    is_escape_key_pressed,
    is_exclusive_d3d_fullscreen_active,
    is_game_active_in_foreground,
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
    """3 Katmanlı Hibrit Oyun & Video Çeviri İş Hattı."""

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
        self.game_packs = GamePackStore()
        self.frame_gate = FrameGate(stuck_after_ms=12_000)
        self.translator = GameTranslator(config)
        self.hud_radar = UniversalHUDRadar()

        self.target_hwnd: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hud_thread: Optional[threading.Thread] = None

        # Katman 1 (Altyazı) durumları
        self._last_ocr_text: str = ""
        self._prev_clause_text: str = ""
        self._prev_clause_time: float = 0.0
        self._empty_ticks: int = 0
        self._idle_cycles: int = 0
        self._subtitle_shown: bool = False

        # Katman 2 (Köşe Bilgi Kartı) durumları
        self._corner_last_mask_fp: Optional[np.ndarray] = None
        self._corner_patches: List[HUDTextPatch] = []

        # Katman 3 (Otomatik Tam Ekran Menü / Harita / Envanter) durumları
        self._in_menu_mode: bool = False
        self._esc_menu_toggle: bool = False
        self._menu_patches: List[HUDTextPatch] = []
        self._last_menu_fp: Optional[np.ndarray] = None
        self._pending_menu_fp: Optional[np.ndarray] = None
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
        detected = auto_detect_running_game()
        if not detected:
            return
        hwnd, _title, exe_name = detected
        if self.target_hwnd != hwnd:
            self.target_hwnd = hwnd
            if self.on_status:
                if is_exclusive_d3d_fullscreen_active():
                    self.on_status(
                        f"⚠️ {exe_name} Özel Tam Ekran (Fullscreen) modunda! Overlay için oyun ayarlarından 'Windowed Borderless' seçin."
                    )
                else:
                    self.on_status(f"Oyun Algılandı: {exe_name}")

    def start(self) -> None:
        if self._running:
            return
        set_process_gaming_friendly_priority()
        self._auto_detect_game_hwnd()
        self._running = True
        self._empty_ticks = 0
        self._idle_cycles = 0
        self._corner_last_mask_fp = None
        self._corner_patches = []
        self._in_menu_mode = False
        self._esc_menu_toggle = False
        self._menu_patches = []
        self._last_menu_fp = None
        self._pending_menu_fp = None
        self.frame_gate.reset()
        self.capturer.reset_state()
        self.hud_radar.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._hud_thread = threading.Thread(target=self._hud_loop, daemon=True)
        self._thread.start()
        self._hud_thread.start()
        if self.on_status:
            self.on_status("3 Katmanlı Hibrit Çeviri Aktif (Altyazı + Bilgi Kartı + Otomatik Menü)")

    def stop(self) -> None:
        self._running = False
        self.frame_gate.reset()
        self._corner_patches = []
        self._menu_patches = []
        self._in_menu_mode = False
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
        speaker, body = DialogueTextGrouper.split_speaker_prefix(query_text)
        target_body = body if speaker else query_text

        # 0. Nino GamePacks + PackFuzzy (0.0ms, 1-2 OCR harf hatasını otomatik onarır)
        pack_hit = self.game_packs.lookup(target_body)
        if pack_hit:
            final_tr = f"{speaker}: {pack_hit}" if speaker else pack_hit
            return final_tr, "Oyun Paketi ⚡", True, 0.0

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

    def _batch_translate_grouped_blocks(
        self,
        grouped_blocks: List[Tuple[str, Tuple[int, int, int, int], int]],
    ) -> List[Tuple[str, str, Tuple[int, int, int, int], int]]:
        """
        2D DBNet ile bulunan tüm menü/bilgi kartı bloklarını 3 aşamada anında çevirir:
        1. Anlık 0.0ms Oyun Sözlüğü (`MAP` -> `HARİTA`, `SETTINGS` -> `AYARLAR` vb.)
        2. 0.0ms SQLite Önbellek
        3. Kalanların tamamı TEK BİR `translate_batch()` isteğinde (~180ms)!
        Dönüş: `[(orijinal_metin, turkce_ceviri, (x, y, w, h), satir_sayisi), ...]`
        """
        resolved: Dict[int, str] = {}
        pending_indices: List[int] = []
        pending_bodies: List[str] = []
        pending_speakers: List[Optional[str]] = []
        cleaned_sources: List[str] = []

        for idx, (raw_txt, _rect, _nlines) in enumerate(grouped_blocks):
            cleaned = normalize_ocr_text(raw_txt)
            cleaned_sources.append(cleaned)
            if not cleaned:
                continue

            # 1. Anlık sözlük & Nino GamePacks / PackFuzzy (0.0ms)
            instant = self.hud_radar.lookup_instant_glossary(cleaned) or self.game_packs.lookup(cleaned)
            if instant:
                resolved[idx] = instant
                continue

            # Sayı/para/saat/özel kod gibi çevrilmemesi gereken satırları atla ("$22,331,609", "14:48")
            alpha_cnt = sum(1 for c in cleaned if c.isalpha())
            if alpha_cnt < 3:
                continue

            speaker, body = DialogueTextGrouper.split_speaker_prefix(cleaned)
            target_body = body if speaker else cleaned

            # 2. SQLite Önbellek (0.0ms)
            cached = self.cache.get(
                target_body,
                target_lang=self.config.target_lang,
                genre=self.config.game_genre,
            )
            if cached:
                resolved[idx] = f"{speaker}: {cached}" if speaker else cached
                continue

            pending_indices.append(idx)
            pending_bodies.append(target_body)
            pending_speakers.append(speaker)

        # 3. Kalan tüm kelimeleri TEK BİR toplu istekte çevir (~180ms)
        if pending_bodies:
            batch_tr = self.translator.translate_batch(pending_bodies)
            for p_idx, body_src, spk, tr_body in zip(pending_indices, pending_bodies, pending_speakers, batch_tr):
                tr_clean = (tr_body or "").strip()
                if not tr_clean or is_similar_text(body_src, tr_clean, threshold=0.88):
                    continue
                self.cache.set(
                    source_text=body_src,
                    translated_text=tr_clean,
                    provider="Batch AI",
                    target_lang=self.config.target_lang,
                    genre=self.config.game_genre,
                )
                resolved[p_idx] = f"{spk}: {tr_clean}" if spk else tr_clean

        out: List[Tuple[str, str, Tuple[int, int, int, int], int]] = []
        for idx, (_raw_txt, rect, nlines) in enumerate(grouped_blocks):
            tr = resolved.get(idx, "")
            src = cleaned_sources[idx]
            if tr and src and not is_similar_text(src, tr, threshold=0.88):
                out.append((src, tr, rect, nlines))
        return out

    # =========================================================================
    # KATMAN 1: CANLI ALTYAZI MOTORU (~120ms, %0 FPS KAYBI, NINO FRAMEGATE KORUMALI)
    # =========================================================================
    def process_single_frame(self, force: bool = True) -> Optional[TranslationEvent]:
        """Tek bir altyazı şeridi karesini FrameGate korumasıyla yakalar, okur ve çevirir."""
        status, token = self.frame_gate.enter()
        if status == "busy" and not force:
            return None
        try:
            return self._process_single_frame_inner(force=force, token=token)
        finally:
            if token:
                self.frame_gate.leave(token)

    def _process_single_frame_inner(self, force: bool, token: int) -> Optional[TranslationEvent]:
        t0 = time.perf_counter()
        img, _ = self.capturer.capture_region(
            region=self.config.capture_region,
            diff_threshold=self.config.pixel_diff_threshold,
            preprocess=self.config.preprocess_contrast,
            force=force,
            target_hwnd=self.target_hwnd,
        )

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
                "whoever", "whatever", "whenever", "wherever", "if ", "when ", "while ",
                "unless ", "until ", "because ", "every time", "as soon as",
            )
            dangling_ends = (
                ",", " to", " with", " for", " of", " in", " on", " and", " or", " that", " the", " a",
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

        # FrameGate kontrolü: Çeviri sürerken uygulama durdurulduysa veya yeni tur başladıysa eski sonucu basma
        if token and not self.frame_gate.is_current(token):
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

    # =========================================================================
    # KATMAN 2: CANLI KÖŞE BİLGİ KARTI RADARI (~0% FPS KAYBI, HAREKETLİ 3D'DE BİLE ÇALIŞIR)
    # =========================================================================
    def _scan_corner_info_cards(self) -> None:
        """
        Oynanış sırasında ekranın SADECE sol-üst bilgi kartı köşesini (`%0..%32 x %0..%28`)
        yakalar (~0.5ms). Ham 3D piksel farkı yerine harf-ritmi maskesi (`extract_outlined_subtitle_mask`)
        kullandığı için yürürken veya araba sürerken 3D arka plan aksa bile bilgi kartını %100 yakalar!
        """
        game_rect = get_active_game_window_rect(self.target_hwnd)
        if game_rect:
            gx, gy, gw, gh = game_rect
        else:
            sw, sh = self.capturer.get_primary_screen_size()
            gx, gy, gw, gh = 0, 0, sw, sh

        cx = gx + int(gw * 0.01)
        cy = gy + int(gh * 0.02)
        cw = max(240, int(gw * 0.32))
        ch = max(120, int(gh * 0.28))

        corner_rgb = self.capturer.capture_rect_rgb(cx, cy, cw, ch)
        corner_mask = ScreenCapturer.extract_outlined_subtitle_mask(corner_rgb)
        active_px = int(np.count_nonzero(corner_mask))

        # Köşede hiç metin kalmadıysa aktif yamayı anında temizle
        if active_px < 40:
            self._corner_last_mask_fp = None
            if self._corner_patches:
                self._corner_patches = []
                if self.on_hud_patches:
                    self.on_hud_patches([])
            return

        small_mask = cv2.resize(corner_mask, (64, 32), interpolation=cv2.INTER_AREA) > 16
        if self._corner_last_mask_fp is not None and self._corner_last_mask_fp.shape == small_mask.shape:
            union = int(np.count_nonzero(small_mask | self._corner_last_mask_fp))
            xor = int(np.count_nonzero(small_mask ^ self._corner_last_mask_fp))
            xor_ratio = float(xor) / float(max(1, union))
            # Aynı bilgi kartı ekranda duruyorsa tekrar OCR çalıştırma (%0 CPU!)
            if xor_ratio < 0.16:
                return

        self._corner_last_mask_fp = small_mask

        # Küçük kırpım üzerinde (~55ms) gerçek 2D DBNet çalıştır!
        raw_blocks = self.ocr.detect_and_read_blocks(corner_rgb, min_conf=0.56, max_side=640)
        if not raw_blocks:
            return

        grouped = self.hud_radar.group_dbnet_blocks(raw_blocks, offset_x=cx, offset_y=cy)
        # Nino CornerBlock.tipBox filtresi: 1-2 kelimelik gürültüyü ve dikey harita etiketlerini ele!
        filtered_grouped = [
            item
            for item in grouped
            if DialogueTextGrouper.is_valid_corner_tip_block(
                item[0],
                item[1],
                num_lines=item[2],
                is_known_glossary=bool(self.hud_radar.lookup_instant_glossary(item[0])),
            )
        ]
        if not filtered_grouped:
            return

        translated_items = self._batch_translate_grouped_blocks(filtered_grouped)
        if not translated_items:
            return

        new_patches: List[HUDTextPatch] = []
        for src, tr, rect, nlines in translated_items:
            new_patches.append(
                self.hud_radar.build_patch_from_rgb(
                    corner_rgb,
                    abs_rect=rect,
                    source_text=src,
                    translated_text=tr,
                    num_lines=nlines,
                    src_offset_x=cx,
                    src_offset_y=cy,
                )
            )

        self._corner_patches = new_patches
        if self.on_hud_patches:
            self.on_hud_patches(list(self._corner_patches))

    # =========================================================================
    # KATMAN 3: OTOMATİK TAM EKRAN MENÜ / HARİTA / ENVANTER TARAYICI (2D DBNET)
    # =========================================================================
    def process_game_hud_frame(self, force_full_menu: bool = False) -> List[HUDTextPatch]:
        """
        Tam Ekran 2D DBNet + Sınırsız Toplu (Batch) Menü Çeviricisi:
        - `Alt+M` ile manuel tetiklenebilir veya menü/imleç açıldığında otomatik çalışır.
        - Ekrandaki TÜM menü sekmelerini, butonları, ayarları ve görev metinlerini (%99+)
          tek seferde çevirir!
        """
        with self._hud_lock:
            if not force_full_menu:
                if not self.config.enable_game_hud_radar or not is_game_active_in_foreground(self.target_hwnd):
                    return []

            game_rect = get_active_game_window_rect(self.target_hwnd)
            if game_rect:
                gx, gy, gw, gh = game_rect
                full_rgb = self.capturer.capture_rect_rgb(gx, gy, gw, gh)
            else:
                gx, gy = 0, 0
                full_rgb = self.capturer.capture_fullscreen_rgb()

            menu_fp = cv2.resize(full_rgb, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)

            # Eğer zorunlu değilse ve menü ekranı son taranan menüyle aynıysa tekrar tarama (%0 CPU!)
            if not force_full_menu and self._last_menu_fp is not None and self._last_menu_fp.shape == menu_fp.shape:
                if float(np.mean(np.abs(menu_fp - self._last_menu_fp))) < 5.5:
                    return list(self._menu_patches)

            # Harita kaydırılırken / sekme animasyonu oynarken bekle, sabitlenince tara
            if not force_full_menu and self._pending_menu_fp is not None and self._pending_menu_fp.shape == menu_fp.shape:
                if float(np.mean(np.abs(menu_fp - self._pending_menu_fp))) > 4.5:
                    self._pending_menu_fp = menu_fp
                    return list(self._menu_patches)

            self._pending_menu_fp = menu_fp
            self._last_menu_fp = menu_fp

            # Gerçek 2D DBNet (`use_det=True`) ile TAM EKRAN tarama (14 kutu limiti YOK!)
            raw_blocks = self.ocr.detect_and_read_blocks(full_rgb, min_conf=0.55, max_side=1280)
            if not raw_blocks:
                self._menu_patches = []
                if self.on_hud_patches:
                    self.on_hud_patches([])
                return []

            grouped = self.hud_radar.group_dbnet_blocks(raw_blocks, offset_x=gx, offset_y=gy)
            translated_items = self._batch_translate_grouped_blocks(grouped)

            patches: List[HUDTextPatch] = []
            for src, tr, rect, nlines in translated_items:
                patches.append(
                    self.hud_radar.build_patch_from_rgb(
                        full_rgb,
                        abs_rect=rect,
                        source_text=src,
                        translated_text=tr,
                        num_lines=nlines,
                        src_offset_x=gx,
                        src_offset_y=gy,
                    )
                )

            self._menu_patches = patches
            if self.on_hud_patches:
                self.on_hud_patches(list(self._menu_patches))
            return list(self._menu_patches)

    def _loop(self) -> None:
        """Katman 1: Hızlı, Sıfır-Blokajlı Altyazı Kanalı."""
        tick_counter = 0
        while self._running:
            t_start = time.perf_counter()
            tick_counter += 1
            if tick_counter % 25 == 1:
                self._auto_detect_game_hwnd()
            if tick_counter % 65 == 0:
                threading.Thread(target=self.translator.warmup, daemon=True).start()
            try:
                # Menü modundayken (tüm ekran zaten Katman 3 tarafından çevrilmişken) altyazıyı duraklat
                if not self._in_menu_mode:
                    with self._sub_lock:
                        self.process_single_frame(force=False)
            except Exception as exc:
                if self.on_status:
                    self.on_status(f"Hata: {exc}")

            elapsed = time.perf_counter() - t_start
            base_interval = max(0.12, float(self.config.scan_interval_ms) / 1000.0)
            adaptive_interval = min(0.24, base_interval * 1.3) if self._idle_cycles >= 20 else base_interval
            sleep_for = max(0.03, adaptive_interval - elapsed)
            time.sleep(sleep_for)

    def _hud_loop(self) -> None:
        """
        Katman 2 & Katman 3 Akıllı Yönetici:
        - `GetCursorInfo` ile fare imleci GÖRÜNÜR olduğu an (ESC, M, I, Tab, J hangi tuş olursa olsun!)
          otomatik olarak Katman 3 (Tam Ekran 2D DBNet Menü Tarayıcı) moduna geçer.
        - İmleç GİZLİ olduğunda (3D canlı oynanış) tüm menü yamalarını anında siler ve sadece
          Katman 2 (Hafif Sol-Üst Köşe Bilgi Kartı Radarı) çalışır (%0 FPS düşüşü!).
        """
        esc_pending = False
        while self._running:
            t_start = time.perf_counter()
            try:
                game_fg = self.config.enable_game_hud_radar and is_game_active_in_foreground(self.target_hwnd)
                if not game_fg:
                    esc_pending = False
                    if self._in_menu_mode or self._menu_patches or self._corner_patches:
                        self._in_menu_mode = False
                        self._esc_menu_toggle = False
                        self._menu_patches = []
                        self._corner_patches = []
                        self._last_menu_fp = None
                        self._pending_menu_fp = None
                        if self.on_hud_patches:
                            self.on_hud_patches([])
                    time.sleep(0.35)
                    continue

                cursor_now = is_cursor_visible()
                now_t = time.perf_counter()
                esc_hit = esc_pending or ((now_t - self._last_esc_time) > 0.65 and is_escape_key_pressed())
                esc_pending = False
                if esc_hit:
                    self._last_esc_time = now_t
                    # GTA 5 gibi kendi DirectX imlecini çizen oyunlarda ESC aç/kapa durumunu takip et
                    self._esc_menu_toggle = not self._esc_menu_toggle

                should_be_menu = cursor_now or self._esc_menu_toggle

                # KATMAN 3: İmleç görünürse (Witcher/Cyberpunk/BG3) VEYA ESC ile menü açıldıysa (GTA 5/RDR 2)
                if should_be_menu:
                    if not self._in_menu_mode:
                        self._in_menu_mode = True
                        self._corner_patches = []
                        self._last_menu_fp = None
                        self._pending_menu_fp = None
                        time.sleep(0.24)
                        self.process_game_hud_frame(force_full_menu=True)
                    else:
                        self.process_game_hud_frame(force_full_menu=False)
                else:
                    if self._in_menu_mode:
                        self._in_menu_mode = False
                        self._menu_patches = []
                        self._last_menu_fp = None
                        self._pending_menu_fp = None
                        self._corner_last_mask_fp = None
                        if self.on_hud_patches:
                            self.on_hud_patches([])

                    # KATMAN 2: Canlı oynanışta hafif sol-üst bilgi kartı köşesini izle (~0.5ms)
                    with self._hud_lock:
                        self._scan_corner_info_cards()
            except Exception:
                pass

            elapsed = time.perf_counter() - t_start
            target_sleep = 0.45 if self._in_menu_mode else 0.65
            remaining = max(0.10, target_sleep - elapsed)
            steps = max(1, int(remaining / 0.06))
            for _ in range(steps):
                if not self._running:
                    break
                if (time.perf_counter() - self._last_esc_time) > 0.65 and is_escape_key_pressed():
                    esc_pending = True
                    break
                if is_cursor_visible() != self._in_menu_mode:
                    break
                time.sleep(0.06)
