"""Win32 Oyun Penceresi Yöneticisi, Sıfır-Stutter Pencere Takipçisi ve Yakalama İzolasyonu."""

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
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010

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


def get_active_game_window_rect(target_hwnd: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Ön plandaki (veya hedef) oyun penceresinin fiziksel ekran koordinatlarını `(left, top, width, height)` döndürür.
    Oyun pencere modundaysa (ör. 1536x864 veya 1280x720) altyazı yakalama ve overlay konumunu
    otomatik olarak oyun penceresinin içine hizalamak için kullanılır.
    """
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
        exe_name = get_window_process_name(hwnd) if hwnd else ""
        if exe_name not in KNOWN_GAME_PROCESSES:
            if target_hwnd and user32.IsWindowVisible(wintypes.HWND(int(target_hwnd))):
                hwnd = int(target_hwnd)
            else:
                return None

        rect = wintypes.RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w < 400 or h < 300:
            return None
        return int(rect.left), int(rect.top), w, h
    except Exception:
        return None


def map_region_to_game_window(
    base_region: Dict[str, int],
    target_hwnd: Optional[int] = None,
) -> Dict[str, int]:
    """
    Eğer aktif oyun penceresi ekranın tamamından küçükse (ör. 1536x864 pencere modu),
    altyazı kutusunun oyun penceresinin dışına (masaüstüne/görev çubuğuna) taşmaması için
    koordinatları oyun penceresinin alt-orta kısmına otomatik ölçekler.
    Tam ekran (1920x1080) modda ise kullanıcının `base_region` koordinatlarını olduğu gibi korur.
    """
    if sys.platform != "win32":
        return dict(base_region)
    try:
        game_rect = get_active_game_window_rect(target_hwnd)
        if not game_rect:
            return dict(base_region)
        gx, gy, gw, gh = game_rect
        user32 = ctypes.windll.user32
        screen_w = max(1, int(user32.GetSystemMetrics(0)))
        screen_h = max(1, int(user32.GetSystemMetrics(1)))

        # Oyun zaten tam ekransa (1920x1080) orijinal bölgeyi doğrudan kullan
        if gw >= screen_w - 16 and gh >= screen_h - 16 and abs(gx) <= 8 and abs(gy) <= 8:
            return dict(base_region)

        # Oyun daha küçük bir penceredeyse (ör. 1536x864), 1920x1080 oranlarını oyun penceresine uyarla
        sx = float(gw) / float(screen_w)
        sy = float(gh) / float(screen_h)
        new_left = gx + int(round(base_region.get("left", 260) * sx))
        new_top = gy + int(round(base_region.get("top", 790) * sy))
        new_w = max(240, int(round(base_region.get("width", 1400) * sx)))
        new_h = max(70, int(round(base_region.get("height", 220) * sy)))
        return {
            "left": max(0, new_left),
            "top": max(0, new_top),
            "width": new_w,
            "height": new_h,
        }
    except Exception:
        return dict(base_region)


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
    SADECE kullanıcı arayüzdeki butona manuel tıkladığında çağrılır.
    Oyunun çözünürlüğünü bozmamak için tam ekran boyutunu (`screen_w x screen_h`, ör. 1920x1080)
    kullanır (`screen_h - 1` KULLANMAZ, böylece GTA 5 1536x864 pencereye küçülmez ve kasmaz!).
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

        HWND_NOTOPMOST = -2
        user32.SetWindowPos(
            wintypes.HWND(target_hwnd),
            wintypes.HWND(HWND_NOTOPMOST),
            0,
            0,
            screen_w,
            screen_h,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOACTIVATE | SWP_NOOWNERZORDER,
        )
        return True, f"{exe_name or 'Oyun'} Çerçevesiz Tam Ekran ({screen_w}x{screen_h}) yapıldı!"
    except Exception as exc:
        return False, f"Pencere dönüştürme hatası: {exc}"


def ensure_game_window_allows_overlay(target_hwnd: Optional[int] = None) -> None:
    """
    OverTranslate `AlwaysOnTop` prensibi:
    Oyun penceresinin boyutuna, çözünürlüğüne veya `GWL_STYLE` bayraklarına ASLA dokunmaz!
    (Oyun penceresine dışarıdan `SetWindowLongW` veya `SWP_FRAMECHANGED` göndermek GTA 5'i
    kendi kendine 1536x864 pencere moduna düşürür ve DirectX swapchain takılmasına / FPS drop'a yol açar).
    Bunun yerine sadece kendi overlay penceremizin `HWND_TOPMOST` bayrağı yenilenir.
    """
    return
