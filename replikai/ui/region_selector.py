"""Qt6 Tabanlı Profesyonel HUD Ekran Bölgesi Seçici (Crosshair + Boyut Göstergeli)."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegionSelector(QWidget):
    """Oyun ekranında altyazı alanını milimetrik seçmek için tam ekran HUD katmanı."""

    def __init__(
        self,
        on_selected: Callable[[Dict[str, int]], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.on_selected = on_selected

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.geometry()
            self.setGeometry(geom)
        else:
            self.showFullScreen()

        self._start_pos: Optional[QPoint] = None
        self._current_pos: Optional[QPoint] = None
        self.show()
        self.raise_()
        self.activateWindow()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Yarı saydam koyu sinematik maske
        painter.fillRect(self.rect(), QColor(7, 10, 16, 175))

        # Üst bilgi rozeti
        banner_rect = QRect((self.width() - 680) // 2, 36, 680, 48)
        painter.setPen(QPen(QColor(0, 245, 212, 180), 1.5))
        painter.setBrush(QColor(15, 20, 32, 230))
        painter.drawRoundedRect(banner_rect, 12, 12)

        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(
            banner_rect,
            Qt.AlignmentFlag.AlignCenter,
            "🎯 Oyundaki Altyazı Alanını Fareyle Sürükleyerek Seçin   •   İptal: ESC",
        )

        if self._start_pos and self._current_pos:
            sel_rect = QRect(self._start_pos, self._current_pos).normalized()

            # Seçilen alanı şeffaflaştır (oyun net görünsün)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(sel_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # İnce neon çerçeve
            painter.setPen(QPen(QColor(0, 245, 212, 220), 2, Qt.PenStyle.SolidLine))
            painter.setBrush(QColor(0, 245, 212, 18))
            painter.drawRect(sel_rect)

            # Köşe L-braketleri (HUD görünümü)
            corner_len = 16
            painter.setPen(QPen(QColor("#FFE66D"), 3.5))
            x1, y1, x2, y2 = sel_rect.left(), sel_rect.top(), sel_rect.right(), sel_rect.bottom()
            # Sol-Üst
            painter.drawLine(x1, y1, x1 + corner_len, y1)
            painter.drawLine(x1, y1, x1, y1 + corner_len)
            # Sağ-Üst
            painter.drawLine(x2, y1, x2 - corner_len, y1)
            painter.drawLine(x2, y1, x2, y1 + corner_len)
            # Sol-Alt
            painter.drawLine(x1, y2, x1 + corner_len, y2)
            painter.drawLine(x1, y2, x1, y2 - corner_len)
            # Sağ-Alt
            painter.drawLine(x2, y2, x2 - corner_len, y2)
            painter.drawLine(x2, y2, x2, y2 - corner_len)

            # Boyut etiketi
            dim_text = f"{sel_rect.width()} × {sel_rect.height()} px  ({sel_rect.x()}, {sel_rect.y()})"
            tag_rect = QRect(sel_rect.left(), max(8, sel_rect.top() - 32), 230, 26)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(13, 18, 28, 235))
            painter.drawRoundedRect(tag_rect, 6, 6)
            painter.setPen(QColor("#00F5D4"))
            painter.setFont(QFont("Segoe UI Semibold", 9))
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, dim_text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.globalPosition().toPoint()
            self._current_pos = self._start_pos
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start_pos is not None:
            self._current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos and self._current_pos:
            rect = QRect(self._start_pos, event.globalPosition().toPoint()).normalized()
            region = {
                "left": int(rect.x()),
                "top": int(rect.y()),
                "width": max(80, int(rect.width())),
                "height": max(36, int(rect.height())),
            }
            self.close()
            self.on_selected(region)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
