"""ReplikAI Pro — Uygulama Giriş Noktası (Qt6 / PySide6)."""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtWidgets import QApplication

from replikai.ui.dashboard import ReplikAIDashboard


def enable_windows_dpi_awareness() -> None:
    """Windows yüksek DPI ölçeklendirmesinde ekran koordinatlarının şaşmasını önler."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    enable_windows_dpi_awareness()
    app = QApplication(sys.argv)
    app.setApplicationName("ReplikAI Pro")
    window = ReplikAIDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
