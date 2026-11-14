"""Ultra hızlı ekran yakalama (MSS) ve akıllı piksel değişim algılama."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mss
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class ScreenCapturer:
    """Seçili ekran bölgesini yakalar ve önceki kareyle farkını ölçer."""

    def __init__(self) -> None:
        self._last_gray_small: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        """Son yakalanan kare önbelleğini sıfırlar (zorunlu yeniden okuma için)."""
        self._last_gray_small = None

    def capture_region(
        self,
        region: Dict[str, int],
        diff_threshold: float = 2.5,
        preprocess: bool = True,
        force: bool = False,
    ) -> Tuple[Optional[Image.Image], float]:
        """
        Belirtilen ekran bölgesini yakalar.
        Eğer önceki kareye göre ortalama piksel değişimi `diff_threshold` altındaysa
        (ve `force=False` ise), `(None, diff_score)` döndürerek gereksiz OCR yükünü önler.
        """
        monitor = {
            "left": int(region.get("left", 360)),
            "top": int(region.get("top", 780)),
            "width": max(50, int(region.get("width", 1200))),
            "height": max(30, int(region.get("height", 200))),
        }

        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Hızlı piksel farkı için küçük gri tonlamalı örneklem
        thumb = img.resize((160, 48), Image.Resampling.BILINEAR).convert("L")
        arr = np.asarray(thumb, dtype=np.float32)

        diff_score = 999.0
        if self._last_gray_small is not None and self._last_gray_small.shape == arr.shape:
            diff_score = float(np.mean(np.abs(arr - self._last_gray_small)))

        if not force and diff_score < diff_threshold:
            return None, diff_score

        self._last_gray_small = arr

        if preprocess:
            img = self.enhance_for_ocr(img)

        return img, diff_score

    @staticmethod
    def enhance_for_ocr(img: Image.Image) -> Image.Image:
        """Oyun altyazılarının arka plandan ayrışması için kontrast ve keskinlik iyileştirmesi."""
        # Çok küçük bölgeleri 1.5x büyütmek OCR doğruluğunu ciddi artırır
        w, h = img.size
        if h < 120:
            img = img.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.BICUBIC)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = ImageEnhance.Contrast(img).enhance(1.35)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
        return img
