"""Sıfır FPS-Drop Ekran Yakalayıcı ve Otomatik 3D Oyun Altyazı Dedektörü."""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image


class ScreenCapturer:
    """
    DirectX / DWM uyumlu, kalıcı oturumlu ekran yakalayıcı.
    3D oyunlarda (GTA 5, RDR 2) altyazı arkasındaki hareketli yolu/arabaları
    yoksayarak SADECE siyah konturlu parlak altyazı piksellerinin değişimini izler.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._last_subtitle_mask_small: Optional[np.ndarray] = None
        self.subtitle_currently_visible: bool = False

    def _get_mss(self):
        """Her karede yeni GDI/DWM handle açıp kapatmak yerine kalıcı MSS nesnesi kullanır."""
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._tls.sct = sct
        return sct

    def reset_state(self) -> None:
        self._last_subtitle_mask_small = None
        self.subtitle_currently_visible = False

    @staticmethod
    def extract_outlined_subtitle_mask(rgb_arr: np.ndarray) -> np.ndarray:
        """
        GTA 5, RDR 2 vb. oyunlardaki parlak beyaz/sarı/renkli ve koyu dış konturlu (drop-shadow)
        altyazı harflerini arka plandaki hareketli 3D dünyadan (yol, gökyüzü, far ışığı) ayırır.
        """
        r = rgb_arr[:, :, 0]
        g = rgb_arr[:, :, 1]
        b = rgb_arr[:, :, 2]

        # Parlak beyaz, sarı veya açık renkli altyazı pikselleri
        white_text = (r > 175) & (g > 175) & (b > 170)
        yellow_text = (r > 185) & (g > 155) & (b > 50)
        bright = (white_text | yellow_text).astype(np.uint8) * 255

        # Altyazıların etrafındaki siyah kontur, gölge veya yarı saydam video altyazı kutusu (YouTube/Sinematik)
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        dark_stroke = (gray < 105).astype(np.uint8) * 255

        # Sadece yakınında (5x5 komşulukta) koyu kontur/gölge/kutu bulunan parlak pikselleri tut
        near_dark = cv2.dilate(dark_stroke, np.ones((5, 5), np.uint8), iterations=1)
        subtitle_mask = cv2.bitwise_and(bright, near_dark)
        return subtitle_mask

    def capture_region(
        self,
        region: Dict[str, int],
        diff_threshold: float = 2.2,
        preprocess: bool = True,
        force: bool = False,
        target_hwnd: Optional[int] = None,
    ) -> Tuple[Optional[Image.Image], float]:
        """
        Fiziksel ekran koordinatlarındaki bölgeyi yakalar.
        Ekranda altyazı yoksa `subtitle_currently_visible = False` işaretler ve `(None, 0.0)` döner.
        """
        left = max(0, int(region.get("left", 260)))
        top = max(0, int(region.get("top", 790)))
        width = max(100, int(region.get("width", 1400)))
        height = max(40, int(region.get("height", 220)))

        sct = self._get_mss()
        monitor = {"left": left, "top": top, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        rgb_arr = bgra[:, :, [2, 1, 0]].copy()

        # 0.3ms içinde altyazı maskesini çıkar
        sub_mask = self.extract_outlined_subtitle_mask(rgb_arr)
        active_pixels = int(np.count_nonzero(sub_mask))

        # Ekranda hiç altyazı yoksa (ör. GTA 5'te kimse konuşmuyorsa)
        if not force and active_pixels < 42:
            self.subtitle_currently_visible = False
            self._last_subtitle_mask_small = None
            return None, 0.0

        self.subtitle_currently_visible = True

        # Küçük 160x40 altyazı maskesi üzerinden fark ölç (arka plan hareketinden etkilenmez!)
        small_mask = cv2.resize(sub_mask, (160, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff_score = 999.0
        if (
            self._last_subtitle_mask_small is not None
            and self._last_subtitle_mask_small.shape == small_mask.shape
        ):
            diff_score = float(np.mean(np.abs(small_mask - self._last_subtitle_mask_small)))

        if not force and diff_score < (diff_threshold * 2.2):
            return None, diff_score

        self._last_subtitle_mask_small = small_mask
        return Image.fromarray(rgb_arr), diff_score
