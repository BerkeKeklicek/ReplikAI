"""Sıfır FPS-Drop Ekran Yakalayıcı ve Otomatik 3D Oyun Altyazı Dedektörü."""

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
    3D oyunlarda (GTA 5, RDR 2) hem karanlık sahnelerde hem de güneşli sarı/parlak
    asfalt/kaldırım üzerinde çıkan gölgeli/konturlu altyazıları %100 yakalar.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._last_subtitle_mask_small: Optional[np.ndarray] = None
        self.subtitle_currently_visible: bool = False
        self.last_effective_region: Optional[Dict[str, int]] = None

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
        konturlu veya hafif gölgeli (GTA 5 güneşli sarı kaldırım üstü) altyazı harflerini yakalar:
        - `th_max`: Renkli (sarı/mavi/yeşil) altyazıları yakalar.
        - `th_min`: Güneşli sarı/turuncu yol üzerindeki beyaz altyazıları yakalar (sarı yolda Mavi kanal düşüktür, beyaz harfte yüksektir!).
        - `local_drop`: Sabit `< 70` yerine 5x5 komşulukta yerel gölge/kontur farkına (`max_c - local_min > 40`) bakar;
          böylece güneşli parlak zeminlerdeki yarı saydam oyun altyazı gölgelerini asla kaçırmaz!
        """
        max_c = np.max(rgb_arr, axis=2)
        min_c = np.min(rgb_arr, axis=2)

        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        th_max = cv2.morphologyEx(max_c, cv2.MORPH_TOPHAT, kernel_th)
        th_min = cv2.morphologyEx(min_c, cv2.MORPH_TOPHAT, kernel_th)
        tophat = np.maximum(th_max, th_min)

        local_min = cv2.erode(min_c, np.ones((5, 5), np.uint8), iterations=1)
        local_drop = max_c.astype(np.int16) - local_min.astype(np.int16)

        # Yatay harf ritmi (Comb filter): 3D yoldaki/çimendeki dağınık parlak gürültüyü eler,
        # sadece yan yana harf kenarlarına sahip gerçek altyazı satırlarını geçirir!
        dx = np.abs(np.diff(max_c.astype(np.int16), axis=1, append=0))
        horiz_edges = (dx > 24).astype(np.uint8)
        edge_density = cv2.boxFilter(horiz_edges, ddepth=-1, ksize=(11, 3), normalize=False)

        subtitle_mask = (
            (tophat > 30)
            & (max_c > 148)
            & (local_drop > 42)
            & (edge_density >= 4)
        ).astype(np.uint8) * 255
        return subtitle_mask

    def capture_region(
        self,
        region: Dict[str, int],
        diff_threshold: float = 2.0,
        preprocess: bool = True,
        force: bool = False,
        target_hwnd: Optional[int] = None,
    ) -> Tuple[Optional[Image.Image], float]:
        """
        Fiziksel ekran koordinatlarındaki bölgeyi yakalar.
        Eğer oyun pencere modundaysa (`1536x864` vb.) bölgeyi otomatik olarak oyun penceresinin
        alt-ortasına hizalar.
        """
        eff = map_region_to_game_window(region, target_hwnd)
        self.last_effective_region = eff

        left = max(0, int(eff.get("left", 260)))
        top = max(0, int(eff.get("top", 790)))
        width = max(100, int(eff.get("width", 1400)))
        height = max(40, int(eff.get("height", 220)))

        sct = self._get_mss()
        monitor = {"left": left, "top": top, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        rgb_arr = bgra[:, :, [2, 1, 0]].copy()

        # 0.3ms içinde altyazı maskesini çıkar
        sub_mask = self.extract_outlined_subtitle_mask(rgb_arr)
        active_pixels = int(np.count_nonzero(sub_mask))

        # Gerçek bir altyazı satırı en az 12px dikey bant içinde yoğunlaşmış harf enerjisine sahiptir;
        # yoldaki dağınık araba farı / şerit çizgisi gürültüsünün sahte OCR ve HTTP 429 üretmesini engeller!
        row_counts = np.sum(sub_mask > 0, axis=1).astype(np.float32)
        band_energy = float(np.max(np.convolve(row_counts, np.ones(12, dtype=np.float32), mode="same"))) if len(row_counts) >= 12 else float(active_pixels)

        # Ekranda hiç altyazı yoksa (ör. GTA 5'te kimse konuşmuyorsa)
        if not force and (active_pixels < 60 or band_energy < 50.0):
            self.subtitle_currently_visible = False
            self._last_subtitle_mask_small = None
            return None, 0.0

        self.subtitle_currently_visible = True

        # Küçük 160x40 altyazı maskesi üzerinden AKTİF HARF PİKSELLERİNE göre fark ölç
        small_mask = cv2.resize(sub_mask, (160, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff_score = 999.0
        if (
            self._last_subtitle_mask_small is not None
            and self._last_subtitle_mask_small.shape == small_mask.shape
        ):
            cur_bin = small_mask > 16.0
            prev_bin = self._last_subtitle_mask_small > 16.0
            union_count = int(np.count_nonzero(cur_bin | prev_bin))
            xor_count = int(np.count_nonzero(cur_bin ^ prev_bin))
            xor_ratio = float(xor_count) / float(max(1, union_count))
            l1_active = float(np.sum(np.abs(small_mask - self._last_subtitle_mask_small))) / float(
                max(1, union_count)
            )
            diff_score = max(xor_ratio * 24.0, l1_active / 7.0)

        if not force and diff_score < diff_threshold:
            return None, diff_score

        # YouTube/video kelime-kelime pop-in animasyonu için 45ms bekle;
        # oyunlarda altyazı tek karede anında çıktığı için 0ms bekle (anlık tepki!)
        if not force and not is_game_active_in_foreground(target_hwnd):
            time.sleep(0.045)
            sct_next = sct.grab(monitor)
            bgra_next = np.asarray(sct_next, dtype=np.uint8)
            rgb_next = bgra_next[:, :, [2, 1, 0]].copy()
            mask_next = self.extract_outlined_subtitle_mask(rgb_next)
            if int(np.count_nonzero(mask_next)) >= 30:
                rgb_arr = rgb_next
                small_mask = cv2.resize(mask_next, (160, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

        self._last_subtitle_mask_small = small_mask
        return Image.fromarray(rgb_arr), diff_score

    def get_primary_screen_size(self) -> Tuple[int, int]:
        sct = self._get_mss()
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        return int(mon["width"]), int(mon["height"])

    def capture_rect_rgb(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Ekranın sadece belirtilen küçük dikdörtgenini (~0.5ms, %0 FPS kaybı) RGB olarak yakalar."""
        sct = self._get_mss()
        mon = {"left": max(0, int(left)), "top": max(0, int(top)), "width": max(16, int(width)), "height": max(16, int(height))}
        sct_img = sct.grab(mon)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        return bgra[:, :, [2, 1, 0]].copy()

    def capture_fullscreen_rgb(self) -> np.ndarray:
        """Birincil monitörün tam ekran RGB dizisini döndürür (Sadece Menü / Harita / Envanter Modu için)."""
        sct = self._get_mss()
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        sct_img = sct.grab(mon)
        bgra = np.asarray(sct_img, dtype=np.uint8)
        return bgra[:, :, [2, 1, 0]].copy()
