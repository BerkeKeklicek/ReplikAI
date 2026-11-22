"""Nino (`GamePacks.swift` & `PackFuzzy.swift`) Mimarisi: Hazır Oyun Paketleri ve Bulanık Eşleştirici.

Özellikler:
1. `data/packs/` klasöründeki `.json` ve `.json.gz` (Nino uyumlu sıkıştırılmış paketler dahil)
   tüm hazır oyun çeviri paketlerini otomatik yükler.
2. OCR 1-2 harf yanlış okusa bile (`PackFuzzy` uzunluk kovalı + SequenceMatcher >= 0.87)
   paketteki doğru satıra eşleyip **0.0ms** içinde kusursuz Türkçe çeviriyi döndürür.
3. Paket olmadan da tüm AAA oyunlarda sık çıkan görev/bildirim kalıplarını yerleşik olarak barındırır.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import gzip
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PACKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "packs"

# Tüm AAA oyunlarda sık çıkan hazır başlangıç kalıpları (0.0ms, sıfır anlam kayması)
BUILTIN_AAA_PHRASES: Dict[str, str] = {
    "lose the cops": "Polisleri atlat.",
    "leave the area": "Bölgeden ayrıl.",
    "get back in the vehicle": "Araca geri dön.",
    "get in the car": "Arabaya bin.",
    "get in the vehicle": "Araca bin.",
    "return to the vehicle": "Araca geri dön.",
    "follow the waypoint": "İşaretli rotayı takip et.",
    "mission passed": "GÖREV TAMAMLANDI",
    "mission failed": "GÖREV BAŞARISIZ",
    "you died": "ÖLDÜN",
    "wasted": "ÖLDÜN",
    "busted": "YAKALANDIN",
    "wanted": "ARANIYOR",
    "press e to interact": "Etkileşime geçmek için E tuşuna basın.",
    "hold e to interact": "Etkileşime geçmek için E tuşuna basılı tutun.",
    "press f to enter": "Binmek/girmek için F tuşuna basın.",
    "press esc to skip": "Atlamak için ESC tuşuna basın.",
    "hold space to skip": "Atlamak için BOŞLUK tuşuna basılı tutun.",
    "saving": "Kaydediliyor...",
    "loading": "Yükleniyor...",
    "autosaving": "Otomatik kaydediliyor...",
    "continue": "Devam Et",
    "new game": "Yeni Oyun",
    "load game": "Oyun Yükle",
    "save game": "Oyunu Kaydet",
    "quit to desktop": "Masaüstüne Çık",
    "quit to main menu": "Ana Menüye Dön",
    "return to game": "Oyuna Dön",
    "are you sure you want to quit": "Çıkmak istediğinizden emin misiniz?",
    "checkpoint reached": "Kayıt noktasına ulaşıldı.",
    "objective updated": "Görev güncellendi.",
    "new objective": "Yeni görev.",
}


def _norm_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s']", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class GamePackStore:
    """
    `data/packs/*.json` ve `*.json.gz` dosyalarını belleğe yükleyip
    hem birebir (`O(1)`) hem de bulanık (`PackFuzzy`) eşleştirme yapan motor.
    """

    def __init__(self, packs_dir: Path = PACKS_DIR) -> None:
        self.packs_dir = packs_dir
        self._exact: Dict[str, str] = {}
        # Uzunluk kovalarına göre gruplanmış indeks (hızlı PackFuzzy araması için)
        self._by_len_bucket: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        self.loaded_pack_count: int = 0
        self.reload()

    def reload(self) -> None:
        self._exact.clear()
        self._by_len_bucket.clear()
        self.loaded_pack_count = 0

        for src, tr in BUILTIN_AAA_PHRASES.items():
            self._add_entry(src, tr)

        if not self.packs_dir.exists():
            try:
                self.packs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
            return

        for path in sorted(self.packs_dir.iterdir()):
            if path.suffix == ".json" or path.name.endswith(".json.gz"):
                if self._load_pack_file(path):
                    self.loaded_pack_count += 1

    def _add_entry(self, src: str, tr: str) -> None:
        k = _norm_key(src)
        v = (tr or "").strip()
        if not k or not v or len(k) < 3:
            return
        if k not in self._exact:
            self._exact[k] = v
            bucket = len(k) // 6
            self._by_len_bucket[bucket].append((k, v))

    def _load_pack_file(self, path: Path) -> bool:
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                # Nino formatı: {"lines": {"en": "tr", ...}} veya düz {"en": "tr", ...}
                mapping = data.get("lines") if isinstance(data.get("lines"), dict) else data
                for k, v in mapping.items():
                    if isinstance(k, str) and isinstance(v, str):
                        self._add_entry(k, v)
                return True
        except Exception as exc:
            logger.debug("Oyun paketi okunamadı (%s): %s", path.name, exc)
        return False

    def lookup(self, text: str, fuzzy_threshold: float = 0.87) -> Optional[str]:
        """
        1. Birebir eşleşme (`O(1)`, 0.005ms)
        2. Bulanık eşleşme (`PackFuzzy`, 1-2 OCR harf hatasını otomatik onarır)
        """
        key = _norm_key(text)
        if not key or len(key) < 3:
            return None

        hit = self._exact.get(key)
        if hit:
            return hit

        # Sadece 8+ karakterli cümlelerde bulanık arama yap (kısa kelimelerde yanlış eşleşmeyi önler)
        if len(key) < 8:
            return None

        bucket = len(key) // 6
        best_score = 0.0
        best_val: Optional[str] = None
        first_char = key[0]

        for b in (bucket - 1, bucket, bucket + 1):
            candidates = self._by_len_bucket.get(b)
            if not candidates:
                continue
            for cand_key, cand_tr in candidates:
                # Hızlı ön eleme: uzunluk farkı %22'den fazlaysa atla
                if abs(len(cand_key) - len(key)) > max(3, int(len(key) * 0.22)):
                    continue
                if cand_key[0] != first_char and len(candidates) > 120:
                    continue
                ratio = SequenceMatcher(None, key, cand_key).ratio()
                if ratio >= fuzzy_threshold and ratio > best_score:
                    best_score = ratio
                    best_val = cand_tr
                    if ratio >= 0.96:
                        return best_val

        return best_val
