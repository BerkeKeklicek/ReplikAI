"""SQLite + Bellek içi (LRU) sıfır gecikmeli çeviri önbelleği ve Akıllı Altyazı Süzgeci."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cache.db"

DESKTOP_NOISE_PATTERNS = (
    "obs studio",
    "visual studio",
    "antigravity",
    "recycle bin",
    "geri dönüşüm",
    "rapor yaz",
    "arastirma ve",
    "araştırma ve",
    "replikai",
    "hayalet mod",
    "taşıma modu",
    "tasima modu",
    "markdown kaynak",
    "larkdown kaynak",
    "yeniden yaptim",
    "yeniden yaptım",
    "yemekteyiz",
    "ask anything",
    "@ to mentio",
    "alt + c",
    "alt+c",
    "alt+t",
)

# Sadece Türkçede bulunan harfler + ASCII'ye dönüşmüş yaygın Türkçe kelimeler
# (Türkçeyi Türkçeye çevirmeyi %100 engeller; Türkçe hariç her dili Türkçeye çevirir)
TURKISH_SPECIFIC_CHARS = set("ğĞşŞıİ")
TURKISH_COMMON_WORDS = {
    "bir", "için", "icin", "olan", "olarak", "değil", "degil",
    "yeni", "kaynak", "bayt", "oyun", "oyunu", "bolge", "bölge", "çeviri",
    "ceviri", "lütfen", "lutfen", "evet", "hayır", "hayir", "tamam",
    "veya", "ile", "dene", "bakalim", "bakalım", "reisim", "nasil", "nasıl",
    "simdi", "şimdi", "sadece", "ama", "fakat", "ancak", "gibi", "daha",
    "cok", "çok", "var", "yok", "bunu", "sunu", "oyle", "boyle", "kanka",
    "hocam", "merhaba", "selam", "altyazi", "altyazı", "ekran", "pencere",
    "uygulama", "ayarlar", "baslat", "durdur", "cunku", "çünkü", "belki",
    "zaten", "hemen", "sonra", "once", "önce", "senin", "benim", "bizim",
}


def normalize_ocr_text(text: str) -> str:
    """
    OCR gürültüsünü temizler, yapışık kelimeleri ('AGood' -> 'A Good') onarır,
    masaüstü ikonlarını ve zaten Türkçe olan yazıları filtreler.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    # Küçük harften hemen sonra gelen büyük harfi ayır ("returnTo" / "AGood" -> "A Good")
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    # Arka plan/kontur kaynaklı anlamsız OCR ses/harf gürültülerini temizle ('h NGH!', 'u GHgh!', 'UR')
    cleaned = re.sub(r"\b(?:[uUhH]\s+)?(?:NGH|GHgh|HH|UR)\b[!?.]*", "", cleaned)
    cleaned = re.sub(r"\bWhoeuer\b", "Whoever", cleaned, flags=re.IGNORECASE)
    # Cümle başındaki tek harfli tuş/ikon gürültülerini temizle ('D You are through...' -> 'You are through...')
    cleaned = re.sub(r"^[BCDFGJKLMPQRSTVWXYZ]\s+(?=[A-Z][a-z])", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,_|~`^•·")

    # GTA 5 sağ üst para/saat sayacı gibi salt sayısal HUD satırlarını ('MONDAY 13:58 $22,331,609') ele
    if re.search(r"\$\d[\d,]*", cleaned) and re.search(r"\d{1,2}:\d{2}", cleaned):
        return ""

    # Uzun altyazıların en sonuna çerçeve/kontur yüzünden eklenen tek harfli ' A' veya ' |' gürültüsünü temizle
    raw_words = cleaned.split()
    if len(raw_words) >= 5 and raw_words[-1] in {"A", "I", "l", "1", "|"}:
        cleaned = " ".join(raw_words[:-1])

    alpha_count = sum(1 for ch in cleaned if ch.isalpha())
    if alpha_count < 4:
        return ""

    # Zaten Türkçe özel harfler içeriyorsa çevirme
    if any(ch in TURKISH_SPECIFIC_CHARS for ch in cleaned):
        return ""

    low = cleaned.lower()
    for pat in DESKTOP_NOISE_PATTERNS:
        if pat in low:
            return ""

    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+", low)
    if not words:
        return ""

    # Tek kelimelik çok kısa gürültüleri ele (en az 2 kelime veya >=5 harfli anlamlı kelime olmalı)
    if len(words) == 1 and len(words[0]) < 5:
        return ""

    if any(w in TURKISH_COMMON_WORDS for w in words):
        return ""

    return cleaned


def is_similar_text(a: str, b: str, threshold: float = 0.90) -> bool:
    """İki OCR okuması arasındaki benzerliği ölçer."""
    na = re.sub(r"\s+", " ", (a or "")).strip().lower()
    nb = re.sub(r"\s+", " ", (b or "")).strip().lower()
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


class TranslationCache:
    """Hem RAM hem SQLite katmanlı çeviri önbelleği."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory: Dict[str, str] = {}
        self.hits: int = 0
        self.misses: int = 0
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    key_hash TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 1
                )
                """
            )
            conn.commit()

    @staticmethod
    def _make_key(source_text: str, target_lang: str, genre: str) -> str:
        norm = source_text.strip().lower()
        raw = f"{norm}||{target_lang.lower()}||{genre.lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, source_text: str, target_lang: str = "Türkçe", genre: str = "Genel") -> Optional[str]:
        norm = normalize_ocr_text(source_text)
        if not norm:
            return None
        key = self._make_key(norm, target_lang, genre)
        with self._lock:
            if key in self._memory:
                self.hits += 1
                return self._memory[key]
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT translated_text FROM translations WHERE key_hash = ?",
                    (key,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE translations SET hit_count = hit_count + 1 WHERE key_hash = ?",
                        (key,),
                    )
                    conn.commit()
                    self._memory[key] = row[0]
                    self.hits += 1
                    return row[0]
            self.misses += 1
            return None

    def set(
        self,
        source_text: str,
        translated_text: str,
        provider: str = "unknown",
        target_lang: str = "Türkçe",
        genre: str = "Genel",
    ) -> None:
        norm = normalize_ocr_text(source_text)
        if not norm or not translated_text.strip():
            return
        key = self._make_key(norm, target_lang, genre)
        with self._lock:
            self._memory[key] = translated_text.strip()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO translations (key_hash, source_text, translated_text, provider, genre, created_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key_hash) DO UPDATE SET
                        translated_text = excluded.translated_text,
                        provider = excluded.provider,
                        created_at = excluded.created_at
                    """,
                    (key, norm, translated_text.strip(), provider, genre, time.time()),
                )
                conn.commit()

    def count(self) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM translations").fetchone()
                return int(row[0]) if row else 0

    def recent_entries(self, limit: int = 20) -> List[Tuple[str, str, str]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT source_text, translated_text, provider FROM translations ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [(r[0], r[1], r[2]) for r in rows]

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM translations")
                conn.commit()
