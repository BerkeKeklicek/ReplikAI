"""Tam Ekran Yerinde (In-Place Spatial) Oyun HUD, Bildirim, Telefon ve ESC Menü Çeviri Katmanı."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import List

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from replikai.core.game_window import exclude_window_from_capture
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
    üstüne (In-Place) çizen şeffaf, tıklanabilirliği arkaya geçiren katman.
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

    @staticmethod
    def get_dpr() -> float:
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def _sync_fullscreen_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
        else:
            self.setGeometry(0, 0, 1920, 1080)

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

    def paintEvent(self, _event) -> None:
        if not self._patches:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        dpr = self.get_dpr()

        for patch in self._patches:
            lx = float(patch.x) / dpr
            ly = float(patch.y) / dpr
            lw = max(32.0, float(patch.w) / dpr)
            lh = max(16.0, float(patch.h) / dpr)

            rect = QRectF(lx, ly, lw, lh)

            # 1. Alttaki İngilizce kelimeyi oyunun kendi HUD arka plan rengiyle temizce ört
            bg_col = QColor(patch.bg_rgb[0], patch.bg_rgb[1], patch.bg_rgb[2], 236)
            painter.setBrush(bg_col)
            painter.setPen(QPen(QColor(0, 245, 212, 70), 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            # 2. Türkçe çeviriyi kutunun boyutuna (`lw x lh`) tam sığacak font boyutuyla üstüne yaz
            font_px = max(10, min(22, int(lh * 0.56)))
            font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
            fm = QFontMetricsF(font)

            # Tek satıra sığmıyorsa fontu küçült veya 2 satıra sar
            text = patch.translated_text.strip()
            while font_px > 9 and fm.horizontalAdvance(text) > (lw - 10.0) and " " not in text:
                font_px -= 1
                font = QFont("Segoe UI", font_px, QFont.Weight.Bold)
                fm = QFontMetricsF(font)

            painter.setFont(font)
            text_col = QColor(patch.text_rgb[0], patch.text_rgb[1], patch.text_rgb[2])
            painter.setPen(text_col)
            painter.drawText(
                rect.adjusted(4.0, 1.0, -4.0, -1.0),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                text,
            )
