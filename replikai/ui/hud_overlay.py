"""Tam Ekran Yerinde (In-Place Spatial) Oyun HUD, Bildirim, Telefon ve ESC Menü Çeviri Katmanı.

OverTranslate Mimarisi Entegrasyonu:
1. `RealtimeNaturalBackground` Doğal Doku Yaması:
   Orijinal İngilizce yazının üstüne yapay/düz kutu çizmek yerine, `natural_bg_patch` ile
   harfleri oyunun kendi arka plan dokusu/gradyanı içinde eriterek siler (`QImage` doku kaplama).
2. `SourceTextColorSampler` & `OverlayTextColor`:
   Oyunun kendi harf rengini (ör. GTA 5 sarı/kırmızı/mavi görev vurguları veya beyaz menü metni)
   ve dış konturunu (`outline_rgb`) `QPainterPath` ile birebir koruyarak çizer.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import List

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

from replikai.core.game_window import (
    ensure_game_window_allows_overlay,
    exclude_window_from_capture,
)
from replikai.core.hud_radar import HUDTextPatch

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


class SpatialHUDOverlay(QWidget):
    """
    Oyun içindeki bildirim, ipucu, telefon ve ESC menüsü yazılarını
    TAM OLARAK orijinal İngilizce kelimenin bulunduğu `(x, y, w, h)` koordinatının
    üstüne (In-Place) doğal arka plan yaması (`RealtimeNaturalBackground`) ve
    orijinal oyun font rengiyle çizen şeffaf katman.
    """

    def __init__(self) -> None:
        super().__init__(None)
        self._patches: List[HUDTextPatch] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._sync_fullscreen_geometry()
        self.show()
        self._apply_win32_passthrough_and_capture_exclusion()

        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self.ensure_topmost)
        self._topmost_timer.start(300)

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
        if sys.platform != "win32" or not self._patches:
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
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
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
            new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            ctypes.windll.user32.SetWindowPos(
                wintypes.HWND(hwnd),
                wintypes.HWND(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        except Exception:
            pass

    def set_patches(self, patches: List[HUDTextPatch]) -> None:
        """Aktif yerinde (In-Place) Türkçe HUD/menü yamalarını günceller."""
        self._patches = list(patches)
        if self._patches:
            self._apply_win32_passthrough_and_capture_exclusion()
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

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if not self._patches:
            painter.fillRect(0, 0, 1, 1, QColor(0, 0, 0, 1))
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        dpr = self.get_dpr()

        for patch in self._patches:
            lx = float(patch.x) / dpr
            ly = float(patch.y) / dpr
            lw = max(32.0, float(patch.w) / dpr)
            lh = max(16.0, float(patch.h) / dpr)

            rect = QRectF(lx, ly, lw, lh)

            # 1. OverTranslate `RealtimeNaturalBackground`:
            # Düz kutu yerine orijinal harflerin etraftaki temiz piksellerle silindiği doğal doku yamasını çiz!
            if patch.natural_bg_patch is not None and patch.natural_bg_patch.size > 0:
                qimg = self._numpy_rgb_to_qimage(patch.natural_bg_patch)
                painter.drawImage(rect, qimg)
            else:
                bg_col = QColor(patch.bg_rgb[0], patch.bg_rgb[1], patch.bg_rgb[2], 238)
                painter.setBrush(bg_col)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 4.0, 4.0)

            # 2. Türkçe çeviriyi kutunun boyutuna (`lw x lh`) tam sığacak font boyutuyla hesapla
            text = patch.translated_text.strip()
            if not text:
                continue

            avail_w = max(24.0, lw - 8.0)
            avail_h = max(12.0, lh - 4.0)

            # Kutunun satır sayısına göre başlangıç font boyutunu belirle
            font_px = max(10, min(20, int(lh * 0.52)))
            font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
            fm = QFontMetricsF(font)
            lines = self._wrap_text_lines(text, avail_w, fm)

            while font_px > 9 and (
                (len(lines) * fm.height() * 1.05 > avail_h)
                or any(fm.horizontalAdvance(line) > avail_w for line in lines)
            ):
                font_px -= 1
                font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
                fm = QFontMetricsF(font)
                lines = self._wrap_text_lines(text, avail_w, fm)

            line_h = fm.height() * 1.05
            total_h = line_h * len(lines)
            start_y = ly + max(fm.ascent() + 1.0, (lh - total_h) / 2.0 + fm.ascent())

            path = QPainterPath()
            is_multi_line = len(lines) > 1
            for idx, line in enumerate(lines):
                line_w = fm.horizontalAdvance(line)
                if is_multi_line:
                    # Çok satırlı bildirim kutularında (GTA 5 sol üst bilgi kutusu) sola hizalı
                    tx = lx + 5.0
                else:
                    # Tek satırlı menü/buton etiketlerinde ortalı
                    tx = lx + max(3.0, (lw - line_w) / 2.0)
                ty = start_y + idx * line_h
                path.addText(tx, ty, font, line)

            # 3. OverTranslate `SourceTextColorSampler` + `OverlayTextColor`:
            # Orijinal harf konturu (`outline_rgb`) + WCAG kontrast ayarlı orijinal harf rengi (`text_rgb`)
            outline_col = QColor(patch.outline_rgb[0], patch.outline_rgb[1], patch.outline_rgb[2], 235)
            stroke_pen = QPen(
                outline_col,
                2.6,
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
