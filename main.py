"""ReplikAI — Uygulama Giriş Noktası."""

from __future__ import annotations

import ctypes
import sys

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
    app = ReplikAIDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
