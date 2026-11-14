"""Qt6 Birleşik Otomatik Altyazı Katmanı (Sessizken %100 Görünmez + DPI Uyumlu Ortak Bölge)."""

from __future__ import annotations

import ctypes
import sys
from typing import Callable, Dict, Optional

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from replikai.config import AppConfig
from replikai.core.game_window import exclude_window_from_capture

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080


class SubtitleOverlay(QWidget):
    """
    Birleşik Otomatik Altyazı Katmanı:
    1. Okuma alanı ile altyazı gösterme alanı BİRE BİR AYNI kutudur (farklı yeri okumaz!).
    2. Windows %125 / %150 DPI ölçeklendirmesini otomatik hesaplar (`devicePixelRatio`).
    3. Oyunda kimse konuşmuyorken (altyazı yokken) ekranda HİÇBİR DİKDÖRTGEN GÖSTERMEZ (%100 görünmezdir).
    4. Konuşma bittiğinde veya 5.5 saniye sonra altyazıyı otomatik gizler.
    """

    def __init__(
        self,
        config: AppConfig,
        on_region_moved: Optional[Callable[[Dict[str, int]], None]] = None,
    ) -> None:
        super().__init__(None)
        self.config = config
        self.on_region_moved = on_region_moved
        self._drag_offset: Optional[QPoint] = None
        self._has_active_text: bool = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Konuşma bittiğinde altyazıyı otomatik temizleyen zamanlayıcı
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self.clear_subtitle)

        self._build_layout()
        self.sync_geometry_from_config()
        self.show()
        exclude_window_from_capture(int(self.winId()))
        self.apply_config()

    @staticmethod
    def get_dpr() -> float:
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def sync_geometry_from_config(self) -> None:
        """Fiziksel ekran piksellerini (`1920x1080`) Qt mantıksal koordinatlarına (`1536x864` @ %125 DPI) çevirir."""
        dpr = self.get_dpr()
        reg = self.config.capture_region
        lx = int(round(reg.get("left", 260) / dpr))
        ly = int(round(reg.get("top", 790) / dpr))
        lw = max(200, int(round(reg.get("width", 1400) / dpr)))
        lh = max(60, int(round(reg.get("height", 220) / dpr)))
        self.setGeometry(lx, ly, lw, lh)

    def _save_geometry_to_config(self) -> None:
        """Kullanıcı kutuyu taşıdığında okuma bölgesini (capture_region) kutunun yeni yerine eşitler!"""
        dpr = self.get_dpr()
        geom = self.geometry()
        new_region = {
            "left": max(0, int(round(geom.x() * dpr))),
            "top": max(0, int(round(geom.y() * dpr))),
            "width": max(120, int(round(geom.width() * dpr))),
            "height": max(50, int(round(geom.height() * dpr))),
        }
        self.config.capture_region = new_region
        self.config.save()
        if self.on_region_moved:
            self.on_region_moved(new_region)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 8, 20, 12)
        root.setSpacing(2)

        # Sadece "Konum Ayarlama Modu (Kilitsiz)" açıkken görünen rehber başlığı
        self.guide_label = QLabel(
            "🎯 OTOMATİK ALTYAZI BÖLGESİ — Bu çerçevenin içindeki altyazılar okunur ve burada gösterilir (Taşıyıp Alt+L ile kilitleyin)"
        )
        self.guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guide_label.setFont(QFont("Segoe UI Semibold", 9))
        self.guide_label.setStyleSheet("color: #00F5D4; background: transparent;")
        root.addWidget(self.guide_label)

        self.source_label = QLabel("")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color: rgba(226, 232, 240, 210); background: transparent;")
        root.addWidget(self.source_label)

        self.translated_label = QLabel("")
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translated_label.setWordWrap(True)
        root.addWidget(self.translated_label, stretch=1)

    def paintEvent(self, _event) -> None:
        # Kilitli (Oyun) modundayken ve ekranda aktif altyazı yokken HİÇBİR ŞEY ÇİZME (%100 Görünmez!)
        if self.config.overlay_click_through and not self._has_active_text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)

        if not self.config.overlay_click_through:
            # Konum/Bölge Ayarlama Modu: Neon vizör çerçevesi göster
            painter.setBrush(QColor(11, 15, 25, 180))
            painter.setPen(QPen(QColor("#00F5D4"), 2.0, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect, 12, 12)
            return

        # Aktif Altyazı Varken: Sadece yazının arkasında yumuşak sinematik bant çiz
        alpha = int(max(0.25, min(1.0, float(self.config.overlay_opacity))) * 215)
        bg = QColor(self.config.overlay_bg_color)
        bg.setAlpha(alpha)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 14, 14)

    def set_click_through(self, enabled: bool) -> None:
        self.config.overlay_click_through = enabled
        self.guide_label.setVisible(not enabled)

        if not enabled:
            if not self._has_active_text:
                self.translated_label.setText("Altyazı Alanını Oyundaki Altyazının Üzerine Sürükleyin")
                self.translated_label.setVisible(True)
        else:
            if not self._has_active_text:
                self.translated_label.setText("")
                self.translated_label.setVisible(False)
                self.source_label.setVisible(False)

        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                exclude_window_from_capture(hwnd)
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                base_style = style | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
                if enabled:
                    new_style = base_style | WS_EX_TRANSPARENT
                else:
                    new_style = base_style & ~WS_EX_TRANSPARENT
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            except Exception:
                pass
        self.update()

    def apply_config(self) -> None:
        font_sz = int(self.config.overlay_font_size)
        self.translated_label.setFont(QFont("Segoe UI Semibold", font_sz, QFont.Weight.Bold))
        self.translated_label.setStyleSheet(
            f"color: {self.config.overlay_text_color}; background: transparent;"
        )

        orig_sz = max(10, font_sz - 7)
        orig_font = QFont("Segoe UI", orig_sz)
        orig_font.setItalic(True)
        self.source_label.setFont(orig_font)

        self.set_click_through(bool(self.config.overlay_click_through))
        self.update()

    def update_subtitle(
        self,
        translated: str,
        original: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        cleaned_tr = (translated or "").strip()
        if not cleaned_tr:
            self.clear_subtitle()
            return

        self._has_active_text = True
        if self.config.show_original_text and original:
            self.source_label.setText(original)
            self.source_label.setVisible(True)
        else:
            self.source_label.setVisible(False)

        self.translated_label.setText(cleaned_tr)
        self.translated_label.setVisible(True)
        # 5.5 saniye yeni altyazı gelmezse ekrandan otomatik kaldır
        self._auto_hide_timer.start(5500)
        self.update()

    def clear_subtitle(self) -> None:
        """Konuşma bittiğinde ekrandaki altyazı kutusunu tamamen gizler."""
        self._has_active_text = False
        self.source_label.setText("")
        self.source_label.setVisible(False)
        if self.config.overlay_click_through:
            self.translated_label.setText("")
            self.translated_label.setVisible(False)
        else:
            self.translated_label.setText("Altyazı Alanını Oyundaki Altyazının Üzerine Sürükleyin")
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.config.overlay_click_through and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.config.overlay_click_through and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        if not self.config.overlay_click_through and self._drag_offset is not None:
            self._drag_offset = None
            self._save_geometry_to_config()
