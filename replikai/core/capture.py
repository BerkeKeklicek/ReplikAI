"""Sıfır FPS-Drop Ekran Yakalayıcı ve 3D Oyun Arka Plan (Asfalt/Çevre) Filtresi."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
import threading
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image


PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020


class ScreenCapturer:
    """
    DirectX / DWM uyumlu, kalıcı oturumlu ekran yakalayıcı.
    3D oyunlarda (GTA 5, RDR 2) altyazı arkasındaki hareketli yolu/arabaları
    yoksayarak SADECE siyah konturlu parlak altyazı piksellerinin değişimini izler.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._last_subtitle_mask_small: Optional[np.ndarray] = None
        self._had_subtitle_last_frame: bool = False
        self._kernel_5x5 = np.ones((5, 5), dtype=np.uint8)

    def _get_mss(self):
        """Her karede yeni GDI/DWM handle açıp kapatmak yerine kalıcı MSS nesnesi kullanır."""
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._tls.sct = sct
        return sct

    def reset_state(self) -> None:
        self._last_subtitle_mask_small = None
        self._had_subtitle_last_frame = False

    @staticmethod
    def extract_outlined_subtitle_mask(rgb_arr: np.ndarray) -> np.ndarray:
        """
        GTA 5, RDR 2 vb. oyunlardaki parlak beyaz/sarı/renkli ve koyu dış konturlu (drop-shadow)
        altyazı harflerini arka plandaki hareketli 3D dünyadan (yol, gökyüzü, far ışığı) ayırır.
        """
        r = rgb_arr[:, :, 0]
        g = rgb_arr[:, :, 1]
        b = rgb_arr[:, :, 2]

        # Parlak beyaz veya parlak sarı/kehribar altyazı pikselleri
        white_text = (r > 185) & (g > 185) & (b > 180)
        yellow_text = (r > 195) & (g > 165) & (b > 60)
        bright = (white_text | yellow_text).astype(np.uint8) * 255

        # Altyazıların etrafındaki siyah kontur / gölge pikselleri
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        dark_stroke = (gray < 78).astype(np.uint8) * 255

        # Sadece yakınında (5x5 komşulukta) koyu kontur/gölge bulunan parlak pikselleri tut
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
        Ekran bölgesini yakalar ve 3D arka plan hareketinden bağımsız olarak
        SADECE altyazı maskesi değiştiğinde görüntüyü OCR'a gönderir.
        """
        left = int(region.get("left", 360))
        top = int(region.get("top", 780))
        width = max(50, int(region.get("width", 1200)))
        height = max(30, int(region.get("height", 200)))

        rgb_arr: Optional[np.ndarray] = None

        # Eğer belirli bir oyun penceresi hedeflenmişse ve Windows'taysak önce doğrudan HWND dene
        if target_hwnd and sys.platform == "win32":
            rgb_arr = self._capture_hwnd_region(target_hwnd, left, top, width, height)

        if rgb_arr is None:
            sct = self._get_mss()
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            bgra = np.asarray(sct_img, dtype=np.uint8)
            rgb_arr = bgra[:, :, [2, 1, 0]].copy()

        # 0.3ms içinde altyazı maskesini çıkar
        sub_mask = self.extract_outlined_subtitle_mask(rgb_arr)
        active_pixels = int(np.count_nonzero(sub_mask))

        # Ekranda hiç altyazı yoksa (ör. GTA 5'te sadece asfalt/araba görünüyorsa)
        # OCR motorunu HİÇ çalıştırma -> %0 CPU kullanımı!
        if not force and active_pixels < 38:
            self._had_subtitle_last_frame = False
            self._last_subtitle_mask_small = None
            return None, 0.0

        # Küçük 160x40 altyazı maskesi üzerinden fark ölç (arka plan hareketinden etkilenmez!)
        small_mask = cv2.resize(sub_mask, (160, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff_score = 999.0
        if (
            self._last_subtitle_mask_small is not None
            and self._last_subtitle_mask_small.shape == small_mask.shape
        ):
            diff_score = float(np.mean(np.abs(small_mask - self._last_subtitle_mask_small)))

        if not force and self._had_subtitle_last_frame and diff_score < (diff_threshold * 2.2):
            return None, diff_score

        self._last_subtitle_mask_small = small_mask
        self._had_subtitle_last_frame = True

        return Image.fromarray(rgb_arr), diff_score

    @staticmethod
    def _capture_hwnd_region(
        hwnd: int, left: int, top: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Hedef oyun penceresinin (HWND) DC'sinden doğrudan bölge yakalar."""
        if sys.platform != "win32":
            return None
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        if not user32.IsWindow(wintypes.HWND(hwnd)) or user32.IsIconic(wintypes.HWND(hwnd)):
            return None

        rect = wintypes.RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None

        rel_x = max(0, left - rect.left)
        rel_y = max(0, top - rect.top)

        hwnd_dc = user32.GetWindowDC(wintypes.HWND(hwnd))
        if not hwnd_dc:
            return None

        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        old_bmp = gdi32.SelectObject(mem_dc, bmp)

        try:
            ok = gdi32.BitBlt(mem_dc, 0, 0, width, height, hwnd_dc, rel_x, rel_y, SRCCOPY)
            if not ok:
                return None

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD),
                    ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = width
            bmi.biHeight = -height  # Top-down
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0

            buf = ctypes.create_string_buffer(width * height * 4)
            lines = gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, ctypes.byref(bmi), 0)
            if lines != height:
                return None

            arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            # Eğer tamamen siyah döndüyse (Exclusive Fullscreen) None dön ki MSS yedek devreye girsin
            if int(np.max(arr[:, :, :3])) == 0:
                return None
            return arr[:, :, [2, 1, 0]].copy()
        except Exception:
            return None
        finally:
            gdi32.SelectObject(mem_dc, old_bmp)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(wintypes.HWND(hwnd), hwnd_dc)
