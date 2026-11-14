"""Qt6 Tam Ekran Video & Oyun Üstü Birleşik Altyazı Katmanı (DWM DirectFlip Korumalı + HWND_TOPMOST)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import Callable, Dict, Optional

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget

from replikai.config import AppConfig
from replikai.core.game_window import exclude_window_from_capture

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


class SubtitleOverlay(QWidget):
    """
    Tam Ekran Video (YouTube, Netflix, VLC) ve Oyunlarda Kapanmayan Birleşik Altyazı Katmanı:
    1. DWM DirectFlip Koruması: Tam ekran videolarda Windows'un katmanı arkaya atmaması için
       1x1 görünmez (alpha=1) piksel ile DWM kompozisyonunu canlı tutar.
    2. Her 500ms'de bir ve her yeni altyazıda `SetWindowPos(HWND_TOPMOST)` çağırarak tam ekran
       tarayıcı/oyun pencerelerinin her zaman üstünde kalır.
    3. Win32 `GetWindowRect` ile fiziksel ekran koordinatlarını doğrudan işletim sisteminden alır.
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

        # Tool bayrağını kullanmıyoruz ki tam ekran videoların (Chrome/Edge Fullscreen) üstüne çıkabilsin
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Konuşma bittiğinde altyazıyı otomatik temizleyen zamanlayıcı (7 saniye)
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self.clear_subtitle)

        # Tam ekran video veya oyun açıldığında Z-Order'da en üstte kalmayı garantileyen zamanlayıcı
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self.ensure_topmost)
        self._topmost_timer.start(500)

        self._build_layout()
        self.sync_geometry_from_config()
        self.show()
        exclude_window_from_capture(int(self.winId()))
        self.apply_config()
        self.ensure_topmost()

    @staticmethod
    def get_dpr() -> float:
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def ensure_topmost(self) -> None:
        """Tam ekran YouTube videosu veya oyun açıldığında overlay'i tekrar en üst katmana taşır."""
        if sys.platform != "win32":
            return
        try:
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

    def sync_geometry_from_config(self) -> None:
        """Fiziksel ekran piksellerini (`1920x1080`) Qt koordinatlarına çevirir."""
        dpr = self.get_dpr()
        reg = self.config.capture_region
        lx = int(round(reg.get("left", 260) / dpr))
        ly = int(round(reg.get("top", 790) / dpr))
        lw = max(200, int(round(reg.get("width", 1400) / dpr)))
        lh = max(60, int(round(reg.get("height", 220) / dpr)))
        self.setGeometry(lx, ly, lw, lh)

    def _save_geometry_to_config(self) -> None:
        """Kullanıcı kutuyu taşıdığında doğrudan Win32 fiziksel ekran koordinatlarını kaydeder."""
        if sys.platform == "win32":
            try:
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(wintypes.HWND(int(self.winId())), ctypes.byref(rect)):
                    new_region = {
                        "left": max(0, int(rect.left)),
                        "top": max(0, int(rect.top)),
                        "width": max(120, int(rect.right - rect.left)),
                        "height": max(50, int(rect.bottom - rect.top)),
                    }
                    self.config.capture_region = new_region
                    self.config.save()
                    if self.on_region_moved:
                        self.on_region_moved(new_region)
                    return
            except Exception:
                pass

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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Kilitli moddayken ve ekranda aktif altyazı yokken:
        # Gözle görünmez 1x1 piksel (alpha=1) çiz ki Windows DWM tam ekran videoda pencereyi kapatmasın!
        if self.config.overlay_click_through and not self._has_active_text:
            painter.fillRect(0, 0, 1, 1, QColor(0, 0, 0, 1))
            return

        rect = self.rect().adjusted(2, 2, -2, -2)

        if not self.config.overlay_click_through:
            painter.setBrush(QColor(11, 15, 25, 185))
            painter.setPen(QPen(QColor("#00F5D4"), 2.0, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect, 12, 12)
            return

        # Aktif Altyazı Varken: Sadece yazının arkasında sinematik yarı saydam bant çiz
        alpha = int(max(0.25, min(1.0, float(self.config.overlay_opacity))) * 220)
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
                self.translated_label.setText("Altyazı Alanını Oyundaki / Videodaki Altyazının Üzerine Sürükleyin")
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
                base_style = style | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOPMOST
                if enabled:
                    new_style = base_style | WS_EX_TRANSPARENT
                else:
                    new_style = base_style & ~WS_EX_TRANSPARENT
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                self.ensure_topmost()
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
            return

        self._has_active_text = True
        if self.config.show_original_text and original:
            self.source_label.setText(original)
            self.source_label.setVisible(True)
        else:
            self.source_label.setVisible(False)

        self.translated_label.setText(cleaned_tr)
        self.translated_label.setVisible(True)
        self.ensure_topmost()
        # 7 saniye boyunca yeni altyazı gelmezse ekrandan otomatik kaldır
        self._auto_hide_timer.start(7000)
        self.update()

    def clear_subtitle(self) -> None:
        """Konuşma bittiğinde ekrandaki altyazı kutusunu gizler."""
        self._has_active_text = False
        self.source_label.setText("")
        self.source_label.setVisible(False)
        if self.config.overlay_click_through:
            self.translated_label.setText("")
            self.translated_label.setVisible(False)
        else:
            self.translated_label.setText("Altyazı Alanını Oyundaki / Videodaki Altyazının Üzerine Sürükleyin")
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
