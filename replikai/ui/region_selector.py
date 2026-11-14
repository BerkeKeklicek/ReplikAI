"""Tam ekran yarı saydam interaktif OCR altyazı bölgesi seçici."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, Optional


class RegionSelector(tk.Toplevel):
    """Kullanıcının ekranda fareyle sürükleyerek altyazı alanını seçmesini sağlar."""

    def __init__(
        self,
        master: tk.Misc,
        on_selected: Callable[[Dict[str, int]], None],
    ) -> None:
        super().__init__(master)
        self.on_selected = on_selected

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.35)
        self.configure(bg="#090B10", cursor="crosshair")

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{screen_w}x{screen_h}+0+0")

        self.canvas = tk.Canvas(
            self,
            bg="#090B10",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_text(
            screen_w // 2,
            60,
            text="🎯 Oyundaki altyazı bölgesini fareyle sürükleyerek seçin (İptal: ESC)",
            fill="#00F5D4",
            font=("Segoe UI", 16, "bold"),
        )

        self._start_x = 0
        self._start_y = 0
        self._rect_id: Optional[int] = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.focus_force()

    def _on_press(self, event: tk.Event) -> None:
        self._start_x = event.x_root
        self._start_y = event.y_root
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            self._start_x,
            self._start_y,
            self._start_x,
            self._start_y,
            outline="#00F5D4",
            width=3,
            fill="#14213D",
            stipple="gray25",
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self._rect_id:
            self.canvas.coords(
                self._rect_id,
                self._start_x,
                self._start_y,
                event.x_root,
                event.y_root,
            )

    def _on_release(self, event: tk.Event) -> None:
        x1, y1 = min(self._start_x, event.x_root), min(self._start_y, event.y_root)
        x2, y2 = max(self._start_x, event.x_root), max(self._start_y, event.y_root)
        width = max(80, x2 - x1)
        height = max(40, y2 - y1)
        region = {
            "left": int(x1),
            "top": int(y1),
            "width": int(width),
            "height": int(height),
        }
        self.destroy()
        self.on_selected(region)
