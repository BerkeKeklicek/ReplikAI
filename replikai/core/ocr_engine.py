"""Hibrit OCR Motoru (RapidOCR ONNX Runtime + Otomatik Satır Birleştirme)."""

from __future__ import annotations

import logging
import threading
from typing import List, Tuple

import numpy as np
from PIL import Image

from replikai.core.cache import normalize_ocr_text

logger = logging.getLogger(__name__)


class OCREngine:
    """Oyun ekranındaki altyazı ve diyalog kutularını yüksek hızda metne dönüştürür."""

    def __init__(self, min_confidence: float = 0.48) -> None:
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._rapid_ocr = None
        self.backend_name = "Hazırlanıyor..."

    def _ensure_initialized(self) -> None:
        if self._rapid_ocr is not None:
            return
        with self._lock:
            if self._rapid_ocr is not None:
                return
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._rapid_ocr = RapidOCR()
                self.backend_name = "RapidOCR (ONNX Yerel)"
            except Exception as exc:
                logger.warning("RapidOCR başlatılamadı: %s", exc)
                self._rapid_ocr = False
                self.backend_name = "Devre Dışı"

    def extract_text(self, image: Image.Image) -> str:
        """Görüntüdeki metinleri yukarıdan aşağıya ve soldan sağa sıralayarak birleştirir."""
        self._ensure_initialized()
        if not self._rapid_ocr:
            return ""

        try:
            rgb_arr = np.asarray(image.convert("RGB"))
            result, _ = self._rapid_ocr(rgb_arr)
            if not result:
                return ""

            # result elemanları: [box, text, score]
            # box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            items: List[Tuple[float, float, str]] = []
            for entry in result:
                if len(entry) < 3:
                    continue
                box, text, score = entry[0], entry[1], float(entry[2])
                if score < self.min_confidence or not text:
                    continue
                top_y = float(min(pt[1] for pt in box))
                left_x = float(min(pt[0] for pt in box))
                items.append((top_y, left_x, str(text).strip()))

            # Satırları dikey hizaya (y) ve yatay sıraya (x) göre diz
            items.sort(key=lambda item: (round(item[0] / 18.0), item[1]))
            joined = " ".join(item[2] for item in items if item[2])
            return normalize_ocr_text(joined)
        except Exception as exc:
            logger.error("OCR okuma hatası: %s", exc)
            return ""
