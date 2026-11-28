"""Birleşik Tam Ekran Oyun Üstü Canlı Altyazı, Köşe Bilgi Kartı ve Alt+M Menü Çeviri Katmanı.

Neden Birleşik Katman?
- Windows 11 DWM üzerinde birden fazla tam ekran/topmost şeffaf pencere (`HWND_TOPMOST`) aynı anda
  açık olduğunda birbirleriyle Z-sırası yarışı yapar.
- `Alt + M` ile oyun üstünde %100 çalıştığı kanıtlanan `SpatialHUDOverlay` doğrudan kendi `paintEvent`
  metodu içinde hem sinematik konturlu canlı altyazıyı (`QPainterPath`) hem de yerinde (`In-Place`)
  menü/HUD doku yamalarını tek bir birleşik yüzeyde çizer!
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QRectF, QTimer, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from replikai.config import AppConfig
from replikai.core.game_window import (
    ensure_game_window_allows_overlay,
    exclude_window_from_capture,
    map_region_to_game_window,
)
from replikai.core.hud_radar import HUDTextPatch

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020


class SpatialHUDOverlay(QWidget):
    """
    Oyun ve tam ekran videoların üstünde çalışan tekil, birleşik, tıklama-geçirgen
    (`WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST`) çizim katmanı:
    1. Canlı Altyazı (`_live_translated`): Alt-orta altyazı bölgesine vektörel konturlu temiz metin çizer.
    2. HUD & Alt+M Menü Yamaları (`_patches`): Doğal arka plan dokusuyla yerinde (`In-Place`) çizer.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__(None)
        self.config = config
        self.target_hwnd: Optional[int] = None
        self._patches: List[HUDTextPatch] = []
        self._live_translated: str = ""
        self._live_original: str = ""

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._sub_hide_timer = QTimer(self)
        self._sub_hide_timer.setSingleShot(True)
        self._sub_hide_timer.timeout.connect(self.clear_live_subtitle)

        self._sync_fullscreen_geometry()
        self.show()
        self._apply_win32_passthrough_and_capture_exclusion()

        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self.ensure_topmost)
        self._topmost_timer.start(450)

    @staticmethod
    def get_dpr() -> float:
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def _sync_fullscreen_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen:
            g = screen.geometry()
            self.setGeometry(g.x(), g.y(), g.width(), max(400, g.height() - 1))
        else:
            self.setGeometry(0, 0, 1920, 1079)

    def ensure_topmost(self) -> None:
        if sys.platform != "win32":
            return
        if not self._patches and not self._live_translated:
            return
        try:
            ensure_game_window_allows_overlay()
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                wintypes.HWND(hwnd),
                wintypes.HWND(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    def _apply_win32_passthrough_and_capture_exclusion(self) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            exclude_window_from_capture(hwnd)
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            ctypes.windll.user32.SetWindowPos(
                wintypes.HWND(hwnd),
                wintypes.HWND(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
        except Exception:
            pass

    def set_live_subtitle(self, translated: str, original: Optional[str] = None) -> None:
        """Canlı altyazıyı doğrudan oyun üstü birleşik katmanda gösterir."""
        cleaned = (translated or "").strip()
        if not cleaned:
            return
        self._live_translated = cleaned
        self._live_original = (original or "").strip()
        self.ensure_topmost()
        self._sub_hide_timer.start(7000)
        self.update()

    def clear_live_subtitle(self) -> None:
        if self._live_translated or self._live_original:
            self._live_translated = ""
            self._live_original = ""
            self.update()

    def set_patches(self, patches: List[HUDTextPatch]) -> None:
        self._patches = list(patches)
        if self._patches:
            self.ensure_topmost()
        self.update()

    def clear_patches(self) -> None:
        if self._patches:
            self._patches = []
            self.update()

    @staticmethod
    def _wrap_text_lines(text: str, max_width: float, fm: QFontMetricsF) -> List[str]:
        words = (text or "").split()
        if not words:
            return []
        lines: List[str] = []
        current = words[0]
        for w in words[1:]:
            cand = f"{current} {w}"
            if fm.horizontalAdvance(cand) <= max_width:
                current = cand
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return lines

    @staticmethod
    def _numpy_rgb_to_qimage(arr: np.ndarray) -> QImage:
        contiguous = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w, _ = contiguous.shape
        qimg = QImage(contiguous.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return qimg.copy()

    def _draw_live_subtitle(self, painter: QPainter, dpr: float) -> None:
        if not self._live_translated or self.config is None:
            return
        # Kilit açıkken (kullanıcı yeşil çerçeveyi sürüklerken) SubtitleOverlay zaten çiziyor
        if not self.config.overlay_click_through:
            return

        reg = map_region_to_game_window(self.config.capture_region, self.target_hwnd)
        lx = float(reg.get("left", 250)) / dpr
        ly = float(reg.get("top", 790)) / dpr
        lw = max(240.0, float(reg.get("width", 1420)) / dpr)
        lh = max(70.0, float(reg.get("height", 280)) / dpr)

        sub_rect = QRectF(lx, ly, lw, lh)

        # İsteğe bağlı yarı saydam arka plan kutusu (varsayılan 0.0 = tamamen kutusuz!)
        opac = float(self.config.overlay_opacity)
        if opac > 0.02:
            bg = QColor(self.config.overlay_bg_color)
            bg.setAlpha(int(max(0.05, min(1.0, opac)) * 220))
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(sub_rect.adjusted(8.0, 6.0, -8.0, -6.0), 12.0, 12.0)

        font_sz = int(self.config.overlay_font_size)
        font = QFont("Segoe UI", font_sz, QFont.Weight.Black)
        fm = QFontMetricsF(font)
        avail_w = max(140.0, lw - 32.0)
        lines = self._wrap_text_lines(self._live_translated, avail_w, fm)
        if not lines:
            return

        text_col = QColor(self.config.overlay_text_color)
        if not text_col.isValid():
            text_col = QColor("#FFFFFF")
        lum = 0.299 * text_col.red() + 0.587 * text_col.green() + 0.114 * text_col.blue()
        stroke_col = QColor(255, 255, 255, 250) if lum < 115 else QColor(0, 0, 0, 250)
        stroke_w = 5.8 if lum < 115 else 5.6

        line_h = fm.height() * 1.10
        total_h = line_h * len(lines)
        # Türkçe altyazıyı şeridin üst-orta hizasına yerleştir ki alttaki orijinal İngilizce satırla üst üste binip karışmasın
        start_y = ly + max(fm.ascent() + 10.0, (lh - total_h) * 0.26 + fm.ascent())

        # Orijinal İngilizce metin de açıksa üstte küçük italik göster
        if self.config.show_original_text and self._live_original:
            orig_font = QFont("Segoe UI Semibold", max(11, font_sz - 7), QFont.Weight.DemiBold)
            orig_font.setItalic(True)
            ofm = QFontMetricsF(orig_font)
            orig_lines = self._wrap_text_lines(self._live_original, avail_w, ofm)[:2]
            orig_path = QPainterPath()
            oy_base = start_y - (len(orig_lines) * ofm.height() * 1.05) - 6.0
            for i, ol in enumerate(orig_lines):
                ow = ofm.horizontalAdvance(ol)
                ox = lx + max(8.0, (lw - ow) / 2.0)
                orig_path.addText(ox, oy_base + i * ofm.height() * 1.05, orig_font, ol)
            painter.setPen(QPen(QColor(0, 0, 0, 235), 3.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(orig_path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#E2E8F0"))
            painter.drawPath(orig_path)

        path = QPainterPath()
        for idx, line in enumerate(lines):
            line_w = fm.horizontalAdvance(line)
            tx = lx + max(8.0, (lw - line_w) / 2.0)
            ty = start_y + idx * line_h
            path.addText(tx, ty, font, line)

        pen = QPen(
            stroke_col,
            stroke_w,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(text_col)
        painter.drawPath(path)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if not self._patches and not self._live_translated:
            painter.fillRect(0, 0, 1, 1, QColor(0, 0, 0, 1))
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        dpr = self.get_dpr()

        # 1. Yerinde (In-Place) HUD & Alt+M Menü Yamaları
        for patch in self._patches:
            lx = float(patch.x) / dpr
            ly = float(patch.y) / dpr
            lw = max(18.0, float(patch.w) / dpr)
            lh = max(11.0, float(patch.h) / dpr)

            rect = QRectF(lx, ly, lw, lh)

            if patch.natural_bg_patch is not None and patch.natural_bg_patch.size > 0:
                qimg = self._numpy_rgb_to_qimage(patch.natural_bg_patch)
                painter.drawImage(rect, qimg)
            else:
                bg_col = QColor(patch.bg_rgb[0], patch.bg_rgb[1], patch.bg_rgb[2], 238)
                painter.setBrush(bg_col)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 3.0, 3.0)

            text = patch.translated_text.strip()
            if not text:
                continue

            avail_w = max(16.0, lw - 4.0)
            avail_h = max(10.0, lh - 2.0)

            single_line_lh = lh / float(max(1, patch.num_lines))
            font_px = max(8, min(18, int(single_line_lh * 0.56)))
            font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
            fm = QFontMetricsF(font)
            lines = self._wrap_text_lines(text, avail_w, fm)

            while font_px > 8 and (
                (len(lines) * fm.height() * 1.02 > avail_h)
                or any(fm.horizontalAdvance(line) > avail_w for line in lines)
            ):
                font_px -= 1
                font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
                fm = QFontMetricsF(font)
                lines = self._wrap_text_lines(text, avail_w, fm)

            line_h = fm.height() * 1.02
            total_h = line_h * len(lines)
            start_y = ly + max(fm.ascent(), (lh - total_h) / 2.0 + fm.ascent())

            path = QPainterPath()
            is_multi_line = len(lines) > 1 or patch.num_lines > 1
            for idx, line in enumerate(lines):
                line_w = fm.horizontalAdvance(line)
                tx = (lx + 3.0) if is_multi_line else (lx + max(2.0, (lw - line_w) / 2.0))
                ty = start_y + idx * line_h
                path.addText(tx, ty, font, line)

            outline_col = QColor(patch.outline_rgb[0], patch.outline_rgb[1], patch.outline_rgb[2], 235)
            stroke_pen = QPen(
                outline_col,
                2.2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(stroke_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            text_col = QColor(patch.text_rgb[0], patch.text_rgb[1], patch.text_rgb[2], 255)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(text_col)
            painter.drawPath(path)

        # 2. Canlı Sinematik Altyazı (Aynı Kanıtlanmış Oyun-Üstü Katman Üzerinde!)
        self._draw_live_subtitle(painter, dpr)
