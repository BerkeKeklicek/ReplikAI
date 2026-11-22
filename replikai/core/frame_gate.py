"""Nino (`FrameGate.swift`) Mimarisi: Tek-Kare Akış Kapısı ve Takılma Koruması.

Okuma hattına aynı anda yalnızca tek bir kare sokar; önceki kare hala işleniyorsa yeni kareyi
sıraya yığmak yerine atlar (gecikme birikmesini %100 önler). Her girişe benzersiz bir `token`
numarası verir; durdur/başlat yapıldığında veya takılma (`stuck_after_ms`) yaşandığında eski
okumanın geç gelen sonucu yeni altyazıyı asla ezemez.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class FrameGate:
    """`nino-game-translator/core/Sources/CouchSubsCore/Pipeline/FrameGate.swift` portudur."""

    def __init__(self, stuck_after_ms: int = 12_000) -> None:
        self._stuck_after_ms = max(2_000, int(stuck_after_ms))
        self._lock = threading.Lock()
        self._current_token: Optional[int] = None
        self._current_since_ms: int = 0
        self._last_token: int = 0

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000.0)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current_token is not None

    def enter(self) -> Tuple[str, int]:
        """
        Hatta yeni bir kare sokmayı dener.
        Dönüş:
        - `("enter", token)`: Kare kabul edildi; işlem bitince `leave(token)` çağrılmalıdır.
        - `("busy", 0)`: Önceki kare hala işleniyor; bu kare kuyruk biriktirmemek için atlanır.
        - `("stuck", new_token)`: Önceki kare `stuck_after_ms` süresini aştığı için kapı sıfırlandı
          ve yeni kare `new_token` ile kabul edildi.
        """
        now = self._now_ms()
        with self._lock:
            if self._current_token is not None:
                elapsed = now - self._current_since_ms
                if elapsed < self._stuck_after_ms:
                    return "busy", 0
                # Önceki işlem takılmış; kapıyı sıfırla ve yeni token üret
                self._last_token += 1
                self._current_token = self._last_token
                self._current_since_ms = now
                return "stuck", self._last_token

            self._last_token += 1
            self._current_token = self._last_token
            self._current_since_ms = now
            return "enter", self._last_token

    def is_current(self, token: int) -> bool:
        """Geç gelen bir çeviri sonucunun hala geçerli/güncel olup olmadığını kontrol eder."""
        with self._lock:
            return self._current_token == token

    def leave(self, token: int) -> None:
        """Yalnızca aktif `token` eşleşiyorsa kapıyı serbest bırakır."""
        with self._lock:
            if self._current_token == token:
                self._current_token = None
                self._current_since_ms = 0

    def reset(self) -> None:
        """Durdur / Başlat anında kapıyı anında sıfırlar; eski token'lar geçersiz kalır."""
        with self._lock:
            self._current_token = None
            self._current_since_ms = 0
            self._last_token += 1
