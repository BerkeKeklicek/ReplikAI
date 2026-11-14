"""Qt6 Sinematik Hayalet Altyazı Katmanı (Per-Pixel Alpha + Win32 Click-Through)."""

from __future__ import annotations

import ctypes
import sys
from typing import Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from replikai.config import AppConfig

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020


class SubtitleOverlay(QWidget):
    """Oyunların üstünde çalışan, tıklama geçirgen (Ghost) sinematik altyazı katmanı."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(None)
        self.config = config
        self._drag_offset: Optional[QPoint] = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        screen = QGuiApplication.primaryScreen()
        scr_w, scr_h = (1920, 1080)
        if screen:
            g = screen.geometry()
            scr_w, scr_h = g.width(), g.height()

        width = min(1140, int(scr_w * 0.70))
        height = 132
        left = (scr_w - width) // 2
        top = int(scr_h * 0.81)
        self.setGeometry(left, top, width, height)

        self._build_layout()
        self.show()
        self.apply_config()

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 10, 22, 14)
        root.setSpacing(4)

        # Üst mini HUD barı
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        self.badge_label = QLabel("🔒 HAYALET MOD AKTİF  •  ALT+L: TAŞI / KİLİTLE")
        self.badge_label.setFont(QFont("Segoe UI Semibold", 8))
        self.badge_label.setStyleSheet("color: rgba(0, 245, 212, 190); background: transparent;")
        top_row.addWidget(self.badge_label, alignment=Qt.AlignmentFlag.AlignLeft)

        self.engine_tag = QLabel("REPLIK AI PRO")
        self.engine_tag.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.engine_tag.setStyleSheet("color: rgba(148, 163, 184, 150); background: transparent;")
        top_row.addWidget(self.engine_tag, alignment=Qt.AlignmentFlag.AlignRight)

        root.addLayout(top_row)

        # Orijinal yabancı metin satırı
        self.source_label = QLabel("")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color: rgba(203, 213, 225, 195); background: transparent;")
        root.addWidget(self.source_label)

        # Ana Türkçe Sinematik Altyazı
        self.translated_label = QLabel(
            "ReplikAI Pro Hazır — Oyundaki altyazı bölgesini seçin (Alt+R) ve çeviriyi başlatın (Alt+T)"
        )
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translated_label.setWordWrap(True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setColor(QColor(0, 0, 0, 240))
        shadow.setOffset(0, 2)
        self.translated_label.setGraphicsEffect(shadow)

        root.addWidget(self.translated_label, stretch=1)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        alpha = int(max(0.25, min(1.0, float(self.config.overlay_opacity))) * 235)
        bg = QColor(self.config.overlay_bg_color)
        bg.setAlpha(alpha)

        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.setBrush(bg)

        if self.config.overlay_click_through:
            painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
        else:
            painter.setPen(QPen(QColor("#FFB703"), 2.0))

        painter.drawRoundedRect(rect, 16, 16)

    def set_click_through(self, enabled: bool) -> None:
        self.config.overlay_click_through = enabled
        if enabled:
            self.badge_label.setText("🔒 HAYALET MOD (TIKLAMALAR OYUNA GEÇER)  •  ALT+L")
            self.badge_label.setStyleSheet("color: rgba(0, 245, 212, 175); background: transparent;")
        else:
            self.badge_label.setText("🔓 TAŞIMA MODU AÇIK (FARE İLE SÜRÜKLEYİN)  •  ALT+L: KİLİTLE")
            self.badge_label.setStyleSheet("color: #FFB703; background: transparent;")

        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if enabled:
                    new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT
                else:
                    new_style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            except Exception:
                pass
        self.update()

    def apply_config(self) -> None:
        font_sz = int(self.config.overlay_font_size)
        self.translated_label.setFont(QFont("Segoe UI Semibold", font_sz))
        self.translated_label.setStyleSheet(
            f"color: {self.config.overlay_text_color}; background: transparent;"
        )

        orig_sz = max(10, font_sz - 7)
        orig_font = QFont("Segoe UI", orig_sz)
        orig_font.setItalic(True)
        self.source_label.setFont(orig_font)
        self.source_label.setVisible(bool(self.config.show_original_text))

        self.set_click_through(bool(self.config.overlay_click_through))
        self.update()

    def update_subtitle(
        self,
        translated: str,
        original: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        if self.config.show_original_text and original:
            self.source_label.setText(original)
            self.source_label.setVisible(True)
        else:
            self.source_label.setVisible(False)

        if provider:
            self.engine_tag.setText(provider.upper())

        self.translated_label.setText(translated)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.config.overlay_click_through and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.config.overlay_click_through and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag_offset = None
