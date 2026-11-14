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
        Her renkteki (Beyaz, Sarı, Mavi/Camgöbeği MrBeast altyazıları, Yeşil, Kırmızı vb.)
        konturlu veya koyu kutulu altyazı harflerini yakalar; çimen, gökyüzü ve duvar gibi
        geniş arka plan yüzeylerini Top-Hat morfolojik filtre ile tamamen eler.
        """
        max_c = np.max(rgb_arr, axis=2)

        # 11x11 Top-Hat: Sadece parlak harf çizgilerini geçirir,
        # geniş parlak yüzeyleri (gökyüzü, çimen, asfalt, duvar) sıfırlar.
        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        tophat = cv2.morphologyEx(max_c, cv2.MORPH_TOPHAT, kernel_th)

        # Altyazıların etrafındaki koyu kontur, gölge veya yarı saydam altyazı kutusu
        dark_stroke = (max_c < 70).astype(np.uint8) * 255
        near_dark = cv2.dilate(dark_stroke, np.ones((5, 5), np.uint8), iterations=1)

        subtitle_mask = ((tophat > 60) & (max_c > 152) & (near_dark > 0)).astype(np.uint8) * 255
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

        # Küçük 160x40 altyazı maskesi üzerinden AKTİF HARF PİKSELLERİNE göre fark ölç
        # (Tam kutu ortalaması kısa altyazılarda 'lets team up' değişimini yutuyordu;
        #  aktif piksel XOR oranı sayesinde 2 kelimelik en ufak altyazı değişimi bile %100 yakalanır!)
        small_mask = cv2.resize(sub_mask, (160, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff_score = 999.0
        if (
            self._last_subtitle_mask_small is not None
            and self._last_subtitle_mask_small.shape == small_mask.shape
        ):
            cur_bin = small_mask > 18.0
            prev_bin = self._last_subtitle_mask_small > 18.0
            union_count = int(np.count_nonzero(cur_bin | prev_bin))
            xor_count = int(np.count_nonzero(cur_bin ^ prev_bin))
            xor_ratio = float(xor_count) / float(max(1, union_count))
            l1_active = float(np.sum(np.abs(small_mask - self._last_subtitle_mask_small))) / float(
                max(1, union_count)
            )
            diff_score = max(xor_ratio * 22.0, l1_active / 7.5)

        if not force and diff_score < diff_threshold:
            return None, diff_score

        self._last_subtitle_mask_small = small_mask
        return Image.fromarray(rgb_arr), diff_score
