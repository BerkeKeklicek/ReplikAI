"""SQLite + Bellek içi (LRU) sıfır gecikmeli çeviri önbelleği."""

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


def normalize_ocr_text(text: str) -> str:
    """OCR gürültüsünü temizler ve anlamlı oyun metnini normalize eder."""
    if not text:
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Sadece noktalama veya tek karakterli gürültüleri ele
    alnum_count = sum(1 for ch in cleaned if ch.isalnum())
    if alnum_count < 2:
        return ""
    return cleaned


def is_similar_text(a: str, b: str, threshold: float = 0.90) -> bool:
    """İki OCR okuması arasındaki benzerliği ölçer (küçük OCR titremelerini yoksaymak için)."""
    na = normalize_ocr_text(a).lower()
    nb = normalize_ocr_text(b).lower()
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
        norm = normalize_ocr_text(source_text).lower()
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
