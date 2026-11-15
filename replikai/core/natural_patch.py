"""OverTranslate Mimarisi: Doğal Arka Plan Doku Yaması (Inpainting) ve WCAG/HSL Harf Rengi Örnekleyici.

Hon-Lu/OverTranslate projesindeki iki temel görsel mühendislik modülünün
(`RealtimeNaturalBackground.cs`, `SourceTextColorSampler.cs`, `OverlayTextColor.cs`)
Python / NumPy / OpenCV vektörel karşılığıdır:

1. `RealtimeNaturalBackground`:
   Orijinal İngilizce yazının üstüne düz/yapay bir dikdörtgen kutu çizmek yerine,
   harf piksellerini (`excluded` maskesi) tespit eder ve kutunun üst/alt (`RowBand`) veya
   sol/sağ (`ColumnBand`) temiz kenar piksellerinden 3-derinlikli medyan enterpolasyonu yaparak
   orijinal arka plan dokusunu/gradyanını birebir yeniden üretir.
2. `SourceTextColorSampler` & `OverlayTextColor`:
   Harflerin kenar yumuşatması (anti-aliasing) yüzünden soluklaşan örneklenmiş harf rengini
   HSL uzayında doygunlaştırır (`Tune`) ve WCAG 2.0 bağıl parlaklık (`RelativeLuminance`)
   ikili araması (`EnsureContrast`) ile arka plana karşı en az 3.5 kontrast oranını garantiler.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np

RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class SampledTextStyle:
    """Örneklenen metin katmanı renkleri (Arka plan yüzeyi, harf gövde rengi ve varsa dış kontur)."""

    background_rgb: RGBColor
    text_rgb: RGBColor
    outline_rgb: Optional[RGBColor] = None


class OverlayTextColor:
    """
    `OverTranslate/Services/OverlayTextColor.cs` portudur:
    Ekrandan örneklenen harf rengini anti-aliasing solukluğundan kurtarır ve
    WCAG 2.0 kontrast oranına göre okunabilirliği garanti altına alır.
    """

    MINIMUM_CONTRAST: float = 3.4

    @staticmethod
    def perceived_luminance(rgb: RGBColor) -> float:
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0

    @staticmethod
    def relative_luminance(rgb: RGBColor) -> float:
        """WCAG 2.0 bağıl parlaklık (0.0 - 1.0)."""

        def _linear(channel: int) -> float:
            c = channel / 255.0
            return c / 12.92 if c <= 0.03928 else math.pow((c + 0.055) / 1.055, 2.4)

        return 0.2126 * _linear(rgb[0]) + 0.7152 * _linear(rgb[1]) + 0.0722 * _linear(rgb[2])

    @classmethod
    def contrast_ratio(cls, a: RGBColor, b: RGBColor) -> float:
        """WCAG 2.0 kontrast oranı (1.0 - 21.0)."""
        x = cls.relative_luminance(a)
        y = cls.relative_luminance(b)
        return (max(x, y) + 0.05) / (min(x, y) + 0.05)

    @staticmethod
    def _rgb_to_hsl(rgb: RGBColor) -> Tuple[float, float, float]:
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        # colorsys HLS döndürür (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, s, l

    @staticmethod
    def _hsl_to_rgb(h: float, s: float, l: float) -> RGBColor:
        h = h - math.floor(h)
        s = max(0.0, min(1.0, s))
        l = max(0.0, min(1.0, l))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return (
            int(round(max(0.0, min(1.0, r)) * 255)),
            int(round(max(0.0, min(1.0, g)) * 255)),
            int(round(max(0.0, min(1.0, b)) * 255)),
        )

    @classmethod
    def tune(cls, text_rgb: RGBColor, bg_rgb: RGBColor) -> RGBColor:
        """
        Anti-aliasing yüzünden grileşen/soluklaşan harf renginin doygunluğunu (`saturation`)
        ve parlaklığını (`lightness`) orijinal oyun fontuna yaklaştırır.
        """
        bg_lum = cls.perceived_luminance(bg_rgb)
        text_lum = cls.perceived_luminance(text_rgb)
        h, s, l = cls._rgb_to_hsl(text_rgb)
        is_near_neutral = s < 0.18
        is_near_black = text_lum < 0.16

        if is_near_neutral:
            if is_near_black:
                return text_rgb
            if bg_lum >= 0.55:
                max_allowed = max(0.08, bg_lum - 0.24)
                boosted_lum = min(max_allowed, text_lum + 0.12)
                boosted_l = min(0.74, l + 0.10)
                return cls._hsl_to_rgb(h, min(0.22, s * 1.08), max(boosted_l, boosted_lum))

            min_allowed = min(0.92, bg_lum + 0.30)
            lifted_lum = max(min_allowed, text_lum + 0.08)
            lifted_l = max(l, min(0.90, l + 0.08))
            return cls._hsl_to_rgb(h, min(0.22, s * 1.08), max(lifted_l, lifted_lum))

        # Renkli metinlerde (GTA 5 sarı/kırmızı/mavi/yeşil vurgular) tonu (Hue) koru, doygunluğu güçlendir
        target_s = min(1.0, max(s + 0.08, s * 1.15))
        if bg_lum >= 0.55:
            max_allowed = max(0.08, bg_lum - 0.24)
            adj_lum = min(max_allowed, text_lum + 0.01)
            adj_l = min(0.64, max(l, l + 0.01))
            return cls._hsl_to_rgb(h, target_s, max(adj_l, adj_lum))

        min_allowed_color = min(0.90, bg_lum + 0.22)
        color_lum = max(min_allowed_color, text_lum + 0.01)
        color_l = max(l, min(0.82, l + 0.01))
        return cls._hsl_to_rgb(h, target_s, max(color_l, color_lum))

    @classmethod
    def ensure_contrast(cls, text_rgb: RGBColor, bg_rgb: RGBColor, min_ratio: float = MINIMUM_CONTRAST) -> RGBColor:
        """
        16 adımlı ikili arama (bisection) ile harf renginin tonunu ve doygunluğunu bozmadan
        arka plana karşı WCAG kontrast oranını (`min_ratio`) sağlayacak en yakın parlaklığa taşır.
        """
        if cls.contrast_ratio(text_rgb, bg_rgb) >= min_ratio:
            return text_rgb

        h, s, l = cls._rgb_to_hsl(text_rgb)
        lighter_first = cls.relative_luminance(text_rgb) >= cls.relative_luminance(bg_rgb)

        for lighter in (lighter_first, not lighter_first):
            limit = 1.0 if lighter else 0.0
            if cls.contrast_ratio(cls._hsl_to_rgb(h, s, limit), bg_rgb) < min_ratio:
                continue

            near, far = l, limit
            for _ in range(16):
                mid = (near + far) / 2.0
                if cls.contrast_ratio(cls._hsl_to_rgb(h, s, mid), bg_rgb) >= min_ratio:
                    far = mid
                else:
                    near = mid
            return cls._hsl_to_rgb(h, s, far)

        black: RGBColor = (0, 0, 0)
        white: RGBColor = (255, 255, 255)
        return black if cls.contrast_ratio(black, bg_rgb) >= cls.contrast_ratio(white, bg_rgb) else white


class SourceTextColorSampler:
    """
    `OverTranslate/Services/SourceTextColorSampler.cs` portudur:
    Kutunun dış halkasından (`outer ring`) gerçek arka plan rengini, iç kısmındaki en yüksek
    kontrastlı piksellerden ise harf gövde rengini ve varsa kontur (outline) rengini çıkarır.
    """

    @classmethod
    def sample(cls, crop_rgb: np.ndarray) -> SampledTextStyle:
        if crop_rgb is None or crop_rgb.size == 0:
            return SampledTextStyle((14, 18, 26), (248, 250, 252), (0, 0, 0))

        h, w = crop_rgb.shape[:2]
        if h < 4 or w < 4:
            return SampledTextStyle((14, 18, 26), (248, 250, 252), (0, 0, 0))

        # 1. Arka Plan Yüzeyi (Outer Ring): Kutunun dış çerçevesindeki 2-3 piksellik halkadan örnek al
        ring_y = max(1, min(4, int(round(h * 0.18))))
        ring_x = max(1, min(6, int(round(w * 0.10))))
        ring_mask = np.zeros((h, w), dtype=bool)
        ring_mask[:ring_y, :] = True
        ring_mask[h - ring_y :, :] = True
        ring_mask[:, :ring_x] = True
        ring_mask[:, w - ring_x :] = True

        ring_pixels = crop_rgb[ring_mask]
        if len(ring_pixels) > 0:
            bg_med = np.median(ring_pixels, axis=0)
            bg_rgb: RGBColor = (int(bg_med[0]), int(bg_med[1]), int(bg_med[2]))
        else:
            bg_rgb = (14, 18, 26)

        # 2. İç bölgedeki piksellerin arka plana olan Manhattan renk uzaklığını hesapla
        inner_rgb = crop_rgb[ring_y : max(ring_y + 1, h - ring_y), ring_x : max(ring_x + 1, w - ring_x)]
        if inner_rgb.size == 0:
            inner_rgb = crop_rgb

        bg_arr = np.array(bg_rgb, dtype=np.int16)
        color_dist = np.sum(np.abs(inner_rgb.astype(np.int16) - bg_arr), axis=2)

        # Arka plandan belirgin şekilde ayrılan pikseller (harf gövdesi)
        fg_threshold = max(75.0, float(np.percentile(color_dist, 80)))
        fg_mask = color_dist >= fg_threshold

        if np.count_nonzero(fg_mask) >= 5:
            fg_pixels = inner_rgb[fg_mask]
            # En parlak/belirgin %60'lık dilimin medyanını al (kenar anti-aliasing piksellerini dışla)
            fg_lum = 0.299 * fg_pixels[:, 0] + 0.587 * fg_pixels[:, 1] + 0.114 * fg_pixels[:, 2]
            bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
            if np.median(fg_lum) >= bg_lum:
                core_mask = fg_lum >= np.percentile(fg_lum, 40)
            else:
                core_mask = fg_lum <= np.percentile(fg_lum, 60)
            core_pixels = fg_pixels[core_mask] if np.any(core_mask) else fg_pixels
            raw_text_med = np.median(core_pixels, axis=0)
            raw_text_rgb: RGBColor = (int(raw_text_med[0]), int(raw_text_med[1]), int(raw_text_med[2]))
        else:
            raw_text_rgb = (248, 250, 252) if OverlayTextColor.perceived_luminance(bg_rgb) < 0.55 else (16, 20, 28)

        # 3. Harf dış konturu (Outline) kontrolü: Harflerin hemen bitişiğinde koyu bir kontur katmanı var mı?
        fg_u8 = fg_mask.astype(np.uint8) * 255
        dilated = cv2.dilate(fg_u8, np.ones((3, 3), np.uint8), iterations=1)
        rim_mask = (dilated > 0) & (~fg_mask)
        outline_rgb: Optional[RGBColor] = None
        if np.count_nonzero(rim_mask) >= 6:
            rim_pixels = inner_rgb[rim_mask]
            rim_med = np.median(rim_pixels, axis=0)
            cand_outline: RGBColor = (int(rim_med[0]), int(rim_med[1]), int(rim_med[2]))
            if OverlayTextColor.contrast_ratio(raw_text_rgb, cand_outline) >= 2.2:
                outline_rgb = cand_outline

        # 4. OverTranslate `Tune` + `EnsureContrast` ile rengi canlandır ve okunabilirliği garantile
        tuned = OverlayTextColor.tune(raw_text_rgb, bg_rgb)
        final_text = OverlayTextColor.ensure_contrast(tuned, bg_rgb, OverlayTextColor.MINIMUM_CONTRAST)

        if outline_rgb is None:
            # Kontur yoksa metin parlaklığına zıt yumuşak bir okuma konturu ata
            outline_rgb = (10, 14, 20) if OverlayTextColor.perceived_luminance(final_text) >= 0.45 else (248, 250, 252)

        return SampledTextStyle(
            background_rgb=bg_rgb,
            text_rgb=final_text,
            outline_rgb=outline_rgb,
        )


class RealtimeNaturalBackground:
    """
    `OverTranslate/Services/Realtime/RealtimeNaturalBackground.cs` portudur:
    Oyun içi HUD, telefon ve menü kutularındaki orijinal İngilizce harfleri
    etraftaki temiz arka plan dokusundan (`RowBand` / `ColumnBand` 3-derinlikli medyan)
    enterpolasyon yaparak siler ve doğal bir arka plan yaması (`np.ndarray` RGB) üretir.
    """

    BAND_DEPTH: int = 3

    @classmethod
    def build_natural_patch(cls, crop_rgb: np.ndarray, bg_rgb: RGBColor) -> np.ndarray:
        """
        Girdi olarak aldığı `crop_rgb` içindeki harf ve kontur piksellerini tespit eder,
        temiz arka plan piksellerinden dikey (`FillFromRows`) ve yatay (`FillFromColumns`)
        doğrusal enterpolasyon yaparak harfleri dokuyu bozmadan siler.
        Süre: ~0.4 - 0.7 ms!
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return np.zeros((16, 32, 3), dtype=np.uint8)

        h, w = crop_rgb.shape[:2]
        if h < 6 or w < 6:
            patch = np.empty_like(crop_rgb)
            patch[:] = bg_rgb
            return patch

        # 1. Silinecek harf + kontur piksellerini (`excluded` maskesi) tespit et
        bg_arr = np.array(bg_rgb, dtype=np.int16)
        dist = np.sum(np.abs(crop_rgb.astype(np.int16) - bg_arr), axis=2)

        # Ayrıca yüksek yerel kontrastlı harf darbelerini (Top-Hat / Black-Hat) yakala
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_morph)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_morph)

        glyph_mask = ((dist > 52) | (tophat > 28) | (blackhat > 28)).astype(np.uint8) * 255

        # OverTranslate'deki `ErasedArea` gibi harf konturlarını ve anti-aliasing kırıntılarını
        # tam kapsamak için maskeyi genişlet (`MinErasePad`)
        glyph_mask = cv2.dilate(
            glyph_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        # Dış çerçevedeki 2px sınırını temiz donör olarak koru (eğer tamamen harfle kaplı değilse)
        clean_mask = glyph_mask == 0
        if np.count_nonzero(clean_mask) < max(12, int(h * w * 0.12)):
            # Kutu neredeyse tamamen harfle doluysa yumuşak gradyanlı arka plan üret
            patch = np.empty_like(crop_rgb)
            patch[:] = bg_rgb
            return patch

        # 2. OverTranslate `FillFromRows`: Her sütun (x) için üst ve alt temiz donör bantlarını bul
        # ve dikey eksende doğrusal enterpolasyon (Lerp) yap
        repaired = crop_rgb.astype(np.float32).copy()

        top_band_h = min(cls.BAND_DEPTH + 2, max(2, h // 4))
        bot_band_h = min(cls.BAND_DEPTH + 2, max(2, h // 4))

        # Üst ve alt şeritlerdeki temiz piksellerin sütun bazlı medyanını hesapla
        top_strip = crop_rgb[:top_band_h, :].astype(np.float32)
        bot_strip = crop_rgb[h - bot_band_h :, :].astype(np.float32)
        top_clean = clean_mask[:top_band_h, :]
        bot_clean = clean_mask[h - bot_band_h :, :]

        # Varsayılan olarak genel temiz medyanı baz al
        global_clean_med = np.median(crop_rgb[clean_mask], axis=0).astype(np.float32)

        top_colors = np.tile(global_clean_med, (w, 1))
        bot_colors = np.tile(global_clean_med, (w, 1))

        for x in range(w):
            tc = top_strip[top_clean[:, x], x]
            if len(tc) > 0:
                top_colors[x] = np.median(tc, axis=0)
            bc = bot_strip[bot_clean[:, x], x]
            if len(bc) > 0:
                bot_colors[x] = np.median(bc, axis=0)

        # Yatayda komşu sütunlar arasında yumuşak geçiş (1D Gaussian/Box yumuşatma)
        top_colors = cv2.GaussianBlur(top_colors.reshape(1, w, 3), (9, 1), 0).reshape(w, 3)
        bot_colors = cv2.GaussianBlur(bot_colors.reshape(1, w, 3), (9, 1), 0).reshape(w, 3)

        # Dikey enterpolasyon ağırlık matrisi t in [0, 1], boyut: (h, 1, 1)
        t_y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        interpolated_bg = (1.0 - t_y) * top_colors.reshape(1, w, 3) + t_y * bot_colors.reshape(1, w, 3)

        # Sadece harf/kontur (`glyph_mask > 0`) olan pikselleri doğal enterpolasyonlu arka planla değiştir;
        # geçiş kenarlarını (feathering) yumuşat ki yama izi belli olmasın!
        alpha = cv2.GaussianBlur(glyph_mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None]
        blended = repaired * (1.0 - alpha) + interpolated_bg * alpha

        return np.clip(blended, 0, 255).astype(np.uint8)
