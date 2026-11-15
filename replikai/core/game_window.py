"""Win32 Oyun Penceresi Yöneticisi, Çerçevesiz Tam Ekran Dönüştürücü ve Yakalama İzolasyonu."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import Dict, List, Optional, Tuple

import psutil

GWL_STYLE = -16
GWL_EXSTYLE = -20

WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000

SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
SWP_NOZORDER = 0x0004
SWP_NOOWNERZORDER = 0x0200

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

KNOWN_GAME_PROCESSES = {
    "gta5.exe": "Grand Theft Auto V",
    "gtavenhanced.exe": "Grand Theft Auto V Enhanced",
    "playgtav.exe": "Grand Theft Auto V",
    "rdr2.exe": "Red Dead Redemption 2",
    "witcher3.exe": "The Witcher 3",
    "cyberpunk2077.exe": "Cyberpunk 2077",
    "bg3.exe": "Baldur's Gate 3",
    "bg3_dx11.exe": "Baldur's Gate 3",
    "eldenring.exe": "Elden Ring",
    "starfield.exe": "Starfield",
    "hogwartslegacy.exe": "Hogwarts Legacy",
    "alanwake2.exe": "Alan Wake 2",
    "re4.exe": "Resident Evil 4",
}

NON_GAME_PROCESSES = {
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "vivaldi.exe",
    "explorer.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "applicationframehost.exe",
    "python.exe",
    "pythonw.exe",
    "code.exe",
    "devenv.exe",
    "discord.exe",
    "spotify.exe",
    "steamwebhelper.exe",
    "obs64.exe",
}

VK_ESCAPE = 0x1B
VK_M = 0x4D


def is_game_active_in_foreground(target_hwnd: Optional[int] = None) -> bool:
    """
    Sadece gerçek bir oyun penceresi ön plandayken True döndürür.
    YouTube (Chrome/Edge/Firefox) veya masaüstü açıkken False döndürerek
    Tam Ekran HUD & Menü çevirisinin videoları etkilemesini %100 engeller.
    """
    if sys.platform != "win32":
        return False
    try:
        fg_hwnd = int(ctypes.windll.user32.GetForegroundWindow())
        if not fg_hwnd:
            return False
        exe_name = get_window_process_name(fg_hwnd)
        if not exe_name or exe_name in NON_GAME_PROCESSES:
            return False
        if exe_name in KNOWN_GAME_PROCESSES:
            return True
        if target_hwnd and fg_hwnd == int(target_hwnd):
            return True
    except Exception:
        pass
    return False


def is_escape_key_pressed() -> bool:
    """Oyuncunun ESC tuşuna basıp menüyü açtığını ~0.01ms içinde kontrol eder."""
    if sys.platform != "win32":
        return False
    try:
        state = ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE)
        return bool(state & 0x8000)
    except Exception:
        return False


def set_process_gaming_friendly_priority() -> None:
    """ReplikAI işlemini BELOW_NORMAL önceliğine alır; GTA5 / RDR2 gibi oyunların FPS'ini korur."""
    try:
        proc = psutil.Process()
        if sys.platform == "win32":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def exclude_window_from_capture(hwnd: int) -> bool:
    """
    Belirtilen pencereyi (örn. ReplikAI Altyazı Overlay'i) ekran yakalamadan gizler.
    Böylece Türkçe altyazı penceresi, oyunun orijinal İngilizce altyazısının tam üstüne
    konulsa bile OCR kamerası kendi penceresini görmez, doğrudan arkadaki oyunu okur!
    """
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        res = ctypes.windll.user32.SetWindowDisplayAffinity(
            wintypes.HWND(int(hwnd)),
            wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE),
        )
        return bool(res)
    except Exception:
        return False


def get_window_process_name(hwnd: int) -> str:
    if sys.platform != "win32" or not hwnd:
        return ""
    try:
        pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        if pid.value:
            return psutil.Process(pid.value).name().lower()
    except Exception:
        pass
    return ""


def list_candidate_game_windows() -> List[Tuple[int, str, str]]:
    """
    Açık olan oyun ve uygulama pencerelerini listeler.
    Dönüş: `[(hwnd, pencere_basligi, exe_adi), ...]`
    """
    if sys.platform != "win32":
        return []

    results: List[Tuple[int, str, str]] = []
    user32 = ctypes.windll.user32

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 1:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title or title in ("Program Manager", "Settings", "Windows Input Experience"):
            return True
        if "ReplikAI" in title:
            return True
        exe_name = get_window_process_name(hwnd)
        if exe_name in ("explorer.exe", "searchhost.exe", "shellexperiencehost.exe", "applicationframehost.exe"):
            return True
        results.append((int(hwnd), title, exe_name))
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)

    # Bilinen oyunları listenin en üstüne taşı
    results.sort(key=lambda item: (0 if item[2] in KNOWN_GAME_PROCESSES else 1, item[1].lower()))
    return results


def auto_detect_running_game() -> Optional[Tuple[int, str, str]]:
    """GTA 5, RDR 2 vb. bilinen bir oyun açıksa otomatik bulur."""
    for hwnd, title, exe in list_candidate_game_windows():
        if exe in KNOWN_GAME_PROCESSES:
            return hwnd, title, exe
    return None


def make_window_borderless_fullscreen(hwnd: Optional[int] = None) -> Tuple[bool, str]:
    """
    Hedef oyun penceresini (veya aktif ön plan oyununu) Çerçevesiz Tam Ekran (Borderless Windowed)
    moduna geçirir. GTA 5 ve RDR 2'de Exclusive Fullscreen kaynaklı 'masaüstünü okuma' ve
    'overlay görünmeme' sorununu tek tuşla çözer.
    """
    if sys.platform != "win32":
        return False, "Sadece Windows desteklenir."

    user32 = ctypes.windll.user32
    target_hwnd = hwnd
    if not target_hwnd:
        auto = auto_detect_running_game()
        if auto:
            target_hwnd = auto[0]
        else:
            target_hwnd = user32.GetForegroundWindow()

    if not target_hwnd:
        return False, "Hedef oyun penceresi bulunamadı."

    exe_name = get_window_process_name(target_hwnd)
    if "python" in exe_name:
        auto = auto_detect_running_game()
        if auto:
            target_hwnd = auto[0]
            exe_name = auto[2]
        else:
            return False, "Lütfen önce oyunu açın veya listeden oyun penceresini seçin."

    try:
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)

        style = user32.GetWindowLongW(target_hwnd, GWL_STYLE)
        new_style = style & ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
        new_style |= WS_POPUP | WS_VISIBLE

        user32.SetWindowLongW(target_hwnd, GWL_STYLE, new_style)
        user32.SetWindowPos(
            target_hwnd,
            0,
            0,
            0,
            screen_w,
            screen_h,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER | SWP_NOOWNERZORDER,
        )
        return True, f"{exe_name or 'Oyun'} Çerçevesiz Tam Ekran ({screen_w}x{screen_h}) yapıldı!"
    except Exception as exc:
        return False, f"Pencere dönüştürme hatası: {exc}"
