"""ReplikAI Pro — Uygulama Giriş Noktası (Qt6 / PySide6 + PerMonitorV2 DPI + UAC)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from replikai.ui.dashboard import ReplikAIDashboard


def ensure_admin_privileges_for_game_overlay() -> None:
    """
    GTA 5, RDR 2 gibi Rockstar / BattlEye oyunları Yönetici (Administrator) olarak çalışır.
    Windows UIPI kuralları gereği, Yönetici olmayan bir uygulamanın HWND_TOPMOST katmanı
    oyunun üstüne ÇIKAMAZ (sadece Alt+Tab atınca görünür).
    Bu fonksiyon uygulama Yönetici değilse otomatik olarak Yönetici (UAC) yetkisiyle yeniden başlatır.
    """
    if sys.platform != "win32" or "--no-elevate" in sys.argv:
        return
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
        script_path = str(Path(__file__).resolve())
        args = " ".join([f'"{script_path}"'] + [f'"{a}"' for a in sys.argv[1:]] + ['"--no-elevate"'])
        cwd = str(Path(__file__).resolve().parent)
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            args,
            cwd,
            1,
        )
        if int(ret) > 32:
            sys.exit(0)
    except Exception:
        pass


def enable_windows_dpi_awareness() -> None:
    """
    OverTranslate app.manifest ile birebir aynı PerMonitorV2 DPI farkındalığını açar;
    %125/%150 ölçekli Windows 11 ekranlarda overlay koordinatlarının kaymasını önler.
    """
    if sys.platform != "win32":
        return
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_ssize_t(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    ensure_admin_privileges_for_game_overlay()
    enable_windows_dpi_awareness()

    # Windows 11 %125/%150 DPI ölçeklemede Qt6 kesirli piksel yuvarlama kaymasını engelle
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("ReplikAI Pro")
    window = ReplikAIDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
