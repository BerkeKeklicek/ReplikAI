"""Win32 Click-Through (Tıklama Geçirgen) destekli Hayalet Altyazı Katmanı (Ghost Overlay)."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Optional

from replikai.config import AppConfig

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020


class SubtitleOverlay(tk.Toplevel):
    """Oyunun üstünde duran, fare tıklamalarını engellemeyen Türkçe altyazı penceresi."""

    def __init__(self, master: tk.Misc, config: AppConfig) -> None:
        super().__init__(master)
        self.config = config

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", float(self.config.overlay_opacity))
        self.configure(bg=self.config.overlay_bg_color)

        # Başlangıç konumu: seçilen yakalama bölgesinin hemen altı veya ekranın alt ortası
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1100, int(screen_w * 0.72))
        height = 115
        left = (screen_w - width) // 2
        top = int(screen_h * 0.80)
        self.geometry(f"{width}x{height}+{left}+{top}")

        # İnce üst durum çubuğu (Kilidi açıkken sürüklemek için)
        self.header_frame = tk.Frame(self, bg="#1E222D", height=20)
        self.header_frame.pack(fill="x", side="top")

        self.badge_label = tk.Label(
            self.header_frame,
            text="👻 ReplikAI Canlı Altyazı  |  Alt+L: Kilitle/Taşı  |  Alt+T: Başlat/Durdur",
            fg="#8E9AAF",
            bg="#1E222D",
            font=("Segoe UI", 9, "bold"),
            padx=10,
             anchor="w",
        )
        self.badge_label.pack(fill="x", side="left")

        # Orijinal metin (opsiyonel küçük satır)
        self.source_label = tk.Label(
            self,
            text="",
            fg="#A0AAB8",
            bg=self.config.overlay_bg_color,
            font=("Segoe UI", max(10, int(self.config.overlay_font_size) - 6), "italic"),
            wraplength=width - 40,
            justify="center",
        )

        # Ana Türkçe çeviri metni
        self.translated_label = tk.Label(
            self,
            text="ReplikAI Hazır — Altyazı bölgesi seçip (Alt+R) çeviriyi başlatın (Alt+T)",
            fg=self.config.overlay_text_color,
            bg=self.config.overlay_bg_color,
            font=("Segoe UI Semibold", int(self.config.overlay_font_size)),
            wraplength=width - 40,
            justify="center",
            padx=16,
            pady=8,
        )
        self.translated_label.pack(fill="both", expand=True)

        self._drag_x = 0
        self._drag_y = 0
        for widget in (self, self.header_frame, self.badge_label, self.translated_label):
            widget.bind("<ButtonPress-1>", self._start_move)
            widget.bind("<B1-Motion>", self._on_move)

        self.update_idletasks()
        self.apply_config()

    def _start_move(self, event: tk.Event) -> None:
        if self.config.overlay_click_through:
            return
        self._drag_x = event.x_root - self.winfo_x()
        self._drag_y = event.y_root - self.winfo_y()

    def _on_move(self, event: tk.Event) -> None:
        if self.config.overlay_click_through:
            return
        new_x = event.x_root - self._drag_x
        new_y = event.y_root - self._drag_y
        self.geometry(f"+{new_x}+{new_y}")

    def set_click_through(self, enabled: bool) -> None:
        """Windows API üzerinden pencereyi fare tıklamalarına geçirgen (Ghost) yapar."""
        self.config.overlay_click_through = enabled
        if enabled:
            self.badge_label.configure(
                text="🔒 Hayalet Mod Aktif (Tıklamalar Oyuna Geçer)  |  Taşımak için: Alt+L",
                fg="#00F5D4",
            )
        else:
            self.badge_label.configure(
                text="🔓 Taşıma Modu Açık (Sürükleyebilirsiniz)  |  Kilitlemek için: Alt+L",
                fg="#FFD166",
            )

        if sys.platform != "win32":
            return

        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if not hwnd:
                hwnd = self.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if enabled:
                new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            else:
                new_style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        except Exception:
            pass

    def apply_config(self) -> None:
        self.attributes("-alpha", float(self.config.overlay_opacity))
        self.configure(bg=self.config.overlay_bg_color)
        self.translated_label.configure(
            fg=self.config.overlay_text_color,
            bg=self.config.overlay_bg_color,
            font=("Segoe UI Semibold", int(self.config.overlay_font_size)),
        )
        self.source_label.configure(
            bg=self.config.overlay_bg_color,
            font=("Segoe UI", max(10, int(self.config.overlay_font_size) - 6), "italic"),
        )
        if self.config.show_original_text:
            self.source_label.pack(fill="x", side="top", padx=16, pady=(4, 0), before=self.translated_label)
        else:
            self.source_label.pack_forget()
        self.set_click_through(self.config.overlay_click_through)

    def update_subtitle(self, translated: str, original: Optional[str] = None) -> None:
        if self.config.show_original_text and original:
            self.source_label.configure(text=original)
        self.translated_label.configure(text=translated)
