"""Sıfır FPS-Drop Ekran Yakalayıcı ve Yüksek Hassasiyetli 3D Oyun Altyazı Dedektörü."""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image

from replikai.core.game_window import (
    is_game_active_in_foreground,
    map_region_to_game_window,
)


class ScreenCapturer:
    """
    DirectX / DWM uyumlu, kalıcı oturumlu ekran yakalayıcı.
    - Hem siyah konturlu hem de GTA 5 / RDR 2 gibi sadece hafif sağ-alt gölgeli (drop-shadow)
      ince beyaz/renkli altyazıları (kısa 'Hey!', 'Wait!' replikleri dahil) %100 yakalar.
    - 320x80 yüksek çözünürlüklü maske farkı sayesinde ardışık benzer uzunluktaki
      altyazı değişimlerini asla kaçırmaz.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._last_subtitle_mask_small: Optional[np.ndarray] = None
        self._unrecognized_retries: int = 0
        self.subtitle_currently_visible: bool = False
        self.last_effective_region: Optional[Dict[str, int]] = None

    def _get_mss(self):
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._tls.sct = sct
        return sct

    def reset_state(self) -> None:
        self._last_subtitle_mask_small = None
        self._unrecognized_retries = 0
        self.subtitle_currently_visible = False

    def allow_fadein_recheck(self) -> None:
        """
        Altyazı ekrana yarı saydam (fade-in) girerken ilk karede OCR metni okuyamazsa,
        `_last_subtitle_mask_small` yüzünden cümlenin 4-5 saniye boyunca kilitli kalmasını önler
        (en fazla 2 kez yeniden denemeye izin verir; sabit yol parlamasında CPU tüketmez).
        """
        if self._unrecognized_retries < 2:
            self._unrecognized_retries += 1
            self._last_subtitle_mask_small = None

    def mark_recognized_ok(self) -> None:
        self._unrecognized_retries = 0

    @staticmethod
    def extract_outlined_subtitle_mask(rgb_arr: np.ndarray) -> np.ndarray:
        """
        Her renkteki (Beyaz, Sarı, Camgöbeği, Yeşil, Kırmızı) konturlu veya hafif gölgeli
        altyazı harflerini yakalar:
        - `th_max` & `th_min`: Hem renkli hem güneşli sarı yol üzerindeki beyaz harfleri geçirir.
        - `local_drop > 28`: Açık/parlak zemindeki yarı saydam oyun altyazı gölgelerini kaçırmaz.
        - `edge_density >= 3`: 3D yoldaki tekil parlak noktaları eler, harf dizilerini korur.
        """
        max_c = np.max(rgb_arr, axis=2)
        min_c = np.min(rgb_arr, axis=2)

        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        th_max = cv2.morphologyEx(max_c, cv2.MORPH_TOPHAT, kernel_th)
        th_min = cv2.morphologyEx(min_c, cv2.MORPH_TOPHAT, kernel_th)
        tophat = np.maximum(th_max, th_min)

        local_min = cv2.erode(min_c, np.ones((5, 5), np.uint8), iterations=1)
        local_drop = max_c.astype(np.int16) - local_min.astype(np.int16)

        dx = np.abs(np.diff(max_c.astype(np.int16), axis=1, append=0))
        horiz_edges = (dx > 20).astype(np.uint8)
        edge_density = cv2.boxFilter(horiz_edges, ddepth=-1, ksize=(11, 3), normalize=False)

        subtitle_mask = (
            (tophat > 24)
            & (max_c > 142)
            & (local_drop > 28)
            & (edge_density >= 3)
        ).astype(np.uint8) * 255
        return subtitle_mask

    def capture_region(
        self,
        region: Dict[str, int],
        diff_threshold: float = 1.4,
        preprocess: bool = True,
        force: bool = False,
        target_hwnd: Optional[int] = None,
    ) -> Tuple[Optional[Image.Image], float]:
        eff = map_region_to_game_window(region, target_hwnd)
        self.last_effective_region = eff

        left = max(0, int(eff.get("left", 250)))
        top = max(0, int(eff.get("top", 790)))
        width = max(100, int(eff.get("width", 1420)))
        height = max(40, int(eff.get("height", 280)))

        sct = self._get_mss()
        monitor = {"left": left, "top": top, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        rgb_arr = bgra[:, :, [2, 1, 0]].copy()

        sub_mask = self.extract_outlined_subtitle_mask(rgb_arr)
        active_pixels = int(np.count_nonzero(sub_mask))

        row_counts = np.sum(sub_mask > 0, axis=1).astype(np.float32)
        band_energy = (
            float(np.max(np.convolve(row_counts, np.ones(12, dtype=np.float32), mode="same")))
            if len(row_counts) >= 12
            else float(active_pixels)
        )

        # Kısa altyazıları ("Hey!", "Wait!", "Yeah.") kaçırmayacak hassas eşik
        if not force and (active_pixels < 26 or band_energy < 22.0):
            self.subtitle_currently_visible = False
            self._last_subtitle_mask_small = None
            self._unrecognized_retries = 0
            return None, 0.0

        self.subtitle_currently_visible = True

        # 320x80 çözünürlükte harf şekillerini koruyarak fark ölç (160x40'ta iki farklı cümle 3px bulanık
        # çizgiye dönüşüp birbirinin aynısı sanılıyordu!)
        small_mask = cv2.resize(sub_mask, (320, 80), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff_score = 999.0
        if (
            self._last_subtitle_mask_small is not None
            and self._last_subtitle_mask_small.shape == small_mask.shape
        ):
            cur_bin = small_mask > 14.0
            prev_bin = self._last_subtitle_mask_small > 14.0
            union_count = int(np.count_nonzero(cur_bin | prev_bin))
            xor_count = int(np.count_nonzero(cur_bin ^ prev_bin))
            xor_ratio = float(xor_count) / float(max(1, union_count))
            l1_active = float(np.sum(np.abs(small_mask - self._last_subtitle_mask_small))) / float(
                max(1, union_count)
            )
            diff_score = max(xor_ratio * 26.0, l1_active / 6.5)

        eff_threshold = min(1.5, float(diff_threshold))
        if not force and diff_score < eff_threshold:
            return None, diff_score

        if not force and not is_game_active_in_foreground(target_hwnd):
            time.sleep(0.045)
            sct_next = sct.grab(monitor)
            bgra_next = np.asarray(sct_next, dtype=np.uint8)
            rgb_next = bgra_next[:, :, [2, 1, 0]].copy()
            mask_next = self.extract_outlined_subtitle_mask(rgb_next)
            if int(np.count_nonzero(mask_next)) >= 26:
                rgb_arr = rgb_next
                small_mask = cv2.resize(mask_next, (320, 80), interpolation=cv2.INTER_AREA).astype(np.float32)

        self._last_subtitle_mask_small = small_mask
        return Image.fromarray(rgb_arr), diff_score

    def get_primary_screen_size(self) -> Tuple[int, int]:
        sct = self._get_mss()
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        return int(mon["width"]), int(mon["height"])

    def capture_rect_rgb(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        sct = self._get_mss()
        mon = {
            "left": max(0, int(left)),
            "top": max(0, int(top)),
            "width": max(16, int(width)),
            "height": max(16, int(height)),
        }
        sct_img = sct.grab(mon)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        return bgra[:, :, [2, 1, 0]].copy()

    def capture_fullscreen_rgb(self) -> np.ndarray:
        sct = self._get_mss()
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        sct_img = sct.grab(mon)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        return bgra[:, :, [2, 1, 0]].copy()
