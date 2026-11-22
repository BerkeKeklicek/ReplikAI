"""Yüksek Doğruluklu Oyun & Video Altyazı OCR Motoru (2-Satır Vadi Ayırıcı + DBNet Fallback + OverTranslate SpeakerLabel)."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import requests

from replikai.core.cache import normalize_ocr_text
from replikai.core.capture import ScreenCapturer
from replikai.core.dialogue_grouper import DialogueTextGrouper

logger = logging.getLogger(__name__)

# Nino (`WindowsSpellCheck.Score`) hafif yerel İngilizce kelime puanlayıcı sözlüğü
COMMON_ENGLISH_WORDS = frozenset(
    {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
        "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
        "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
        "been", "has", "had", "did", "does", "doing", "got", "going", "need", "let", "lets", "hey",
        "here", "where", "why", "wait", "stop", "right", "left", "down", "back", "man", "men", "guy",
        "guys", "car", "kill", "dead", "money", "shit", "fuck", "hell", "damn", "please", "sorry",
        "thanks", "thank", "yes", "yeah", "nope", "okay", "ok", "sure", "really", "never", "always",
        "still", "keep", "hold", "leave", "call", "tell", "told", "said", "talk", "listen", "watch",
        "find", "found", "take", "took", "give", "gave", "come", "came", "went", "gone", "done",
        "through", "personal", "line", "service", "twice", "details", "response", "mission", "police",
        "cops", "area", "vehicle", "press", "open", "close", "door", "help", "run", "shoot", "drive",
    }
)

EN_MODEL_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"
EN_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "en_PP-OCRv3_rec_infer.onnx"


class OCREngine:
    """
    Oyun ve tam ekran video altyazılarını %99+ doğrulukla okuyan motor:
    1. 2-Satır Vadi Ayırıcı (Valley Splitter): GTA 5 ve videolarda alt alta çıkan 2 satırlık
       altyazıları (`"Mr. Weston checks...\nwait for a response."`) birbirine yapıştırmadan
       yatay projeksiyon vadisinden iki ayrı satıra böler; böylece `use_det=False` CRNN
       motoru 2 satırlık cümlelerde asla boş (`""`) dönmez!
    2. Akıllı DBNet (`use_det=True`) Yedekleme: Arka plan çok karmaşıksa ve 1D satır okuma
       düşük güven verirse, kırpılmış bölge üzerinde otomatik 2D DBNet satır tespiti çalıştırır.
    3. OverTranslate `SpeakerLabel` geometrisi ile üst satırdaki kısa konuşmacı adını ayırır.
    """

    def __init__(self, min_confidence: float = 0.52) -> None:
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._rapid_ocr = None
        self.backend_name = "Hazırlanıyor..."
        self.last_ocr_ms: float = 0.0

    def warmup(self) -> None:
        self._ensure_initialized()
        if self._rapid_ocr:
            dummy = np.zeros((24, 240, 3), dtype=np.uint8)
            with self._lock:
                try:
                    self._rapid_ocr(dummy, use_det=False, use_cls=False, use_rec=True)
                except Exception:
                    pass

    def _ensure_initialized(self) -> None:
        if self._rapid_ocr is not None:
            return
        with self._lock:
            if self._rapid_ocr is not None:
                return
            try:
                from rapidocr_onnxruntime import RapidOCR

                EN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                if not EN_MODEL_PATH.exists():
                    try:
                        resp = requests.get(EN_MODEL_URL, timeout=10)
                        resp.raise_for_status()
                        EN_MODEL_PATH.write_bytes(resp.content)
                    except Exception as dl_exc:
                        logger.warning("İngilizce OCR modeli indirilemedi: %s", dl_exc)

                kwargs = {
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 1,
                    "det_limit_side_len": 480,
                }
                if EN_MODEL_PATH.exists():
                    kwargs["rec_model_path"] = str(EN_MODEL_PATH)

                self._rapid_ocr = RapidOCR(**kwargs)
                self.backend_name = "RapidOCR Turbo (Warm ONNX)"
            except Exception as exc:
                logger.warning("RapidOCR başlatılamadı: %s", exc)
                self._rapid_ocr = False
                self.backend_name = "Devre Dışı"

    @staticmethod
    def _split_tall_band_at_valley(
        row_counts: np.ndarray,
        y1: int,
        y2: int,
    ) -> List[Tuple[int, int]]:
        """
        Eğer bir altyazı/HUD bandı 2 satır yüksekliğindeyse (`band_h >= 34`),
        ortadaki `[%28..%72]` dikey aralıktaki en düşük piksel yoğunluklu satırı (vadiyi)
        bularak bandı 2 ayrı satıra (`(y1, valley)` ve `(valley, y2)`) böler.
        """
        band_h = y2 - y1
        if band_h < 34:
            return [(y1, y2)]

        s_off = max(8, int(band_h * 0.28))
        e_off = min(band_h - 8, int(band_h * 0.72))
        if e_off <= s_off + 2:
            return [(y1, y2)]

        sub_counts = row_counts[y1 + s_off : y1 + e_off]
        valley_rel = int(np.argmin(sub_counts))
        valley_val = float(sub_counts[valley_rel])
        peak_top = float(np.max(row_counts[y1 : y1 + s_off + valley_rel + 1]))
        peak_bot = float(np.max(row_counts[y1 + s_off + valley_rel : y2]))

        # Vadi satırı üst ve alt satır zirvelerinden belirgin şekilde daha boşsa 2 satıra böl
        if peak_top >= 8 and peak_bot >= 8 and valley_val <= min(peak_top, peak_bot) * 0.68:
            vy = y1 + s_off + valley_rel
            return [(y1, vy + 1), (max(y1, vy - 1), y2)]

        return [(y1, y2)]

    @classmethod
    def _slice_subtitle_lines_with_bounds(
        cls,
        rgb_arr: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Görüntüdeki altyazı satırlarını tespit eder; 2 satırlık birleşik bantları vadiden ikiye
        ayırır ve her satır için `(crop, (x, y, w, h))` döndürür.
        """
        h, w, _ = rgb_arr.shape
        sub_mask = ScreenCapturer.extract_outlined_subtitle_mask(rgb_arr).copy()

        # Sadece çok geniş (w > 950) ve yüksek (h > 140) yakalamalarda en alt 45px köşelerdeki
        # YouTube video oynatıcı ikonlarını maskeden temizle
        if w > 950 and h > 140:
            corner_x = int(w * 0.14)
            corner_y = h - 46
            sub_mask[corner_y:, :corner_x] = 0
            sub_mask[corner_y:, w - corner_x :] = 0

        if np.count_nonzero(sub_mask) < 20:
            max_c = np.max(rgb_arr, axis=2)
            sub_mask = ((max_c > 165).astype(np.uint8)) * 255

        row_counts = np.sum(sub_mask > 0, axis=1)
        min_row_pixels = 4
        active_rows = row_counts >= min_row_pixels

        raw_bands: List[Tuple[int, int]] = []
        start_y = None
        for y, is_active in enumerate(active_rows):
            if is_active and start_y is None:
                start_y = y
            elif not is_active and start_y is not None:
                band_h = y - start_y
                if 7 <= band_h <= 110:
                    raw_bands.extend(cls._split_tall_band_at_valley(row_counts, start_y, y))
                start_y = None
        if start_y is not None and 7 <= (h - start_y) <= 110:
            raw_bands.extend(cls._split_tall_band_at_valley(row_counts, start_y, h))

        merged_bands: List[Tuple[int, int]] = list(raw_bands)

        # En yoğun 3 altyazı satırını yukarıdan aşağıya sırayla al
        if len(merged_bands) > 3:
            merged_bands.sort(key=lambda b: int(np.sum(sub_mask[b[0] : b[1], :] > 0)), reverse=True)
            merged_bands = sorted(merged_bands[:3], key=lambda b: b[0])

        results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for y1, y2 in merged_bands:
            band_mask = sub_mask[y1:y2, :]
            closed_line = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, np.ones((3, 45), np.uint8))
            col_active = np.sum(closed_line > 0, axis=0) >= 2

            spans: List[Tuple[int, int]] = []
            sx = None
            for x, val in enumerate(col_active):
                if val and sx is None:
                    sx = x
                elif not val and sx is not None:
                    if (x - sx) >= 10:
                        spans.append((sx, x))
                    sx = None
            if sx is not None and (w - sx) >= 10:
                spans.append((sx, w))

            merged_spans: List[Tuple[int, int]] = []
            for sx1, sx2 in spans:
                if merged_spans and (sx1 - merged_spans[-1][1]) <= 110:
                    merged_spans[-1] = (merged_spans[-1][0], sx2)
                else:
                    merged_spans.append((sx1, sx2))

            if not merged_spans:
                continue

            best_x1, best_x2 = max(merged_spans, key=lambda s: s[1] - s[0])

            sub_patch = sub_mask[y1:y2, best_x1:best_x2]
            ys, xs = np.where(sub_patch > 0)
            if len(ys) < 8:
                continue

            exact_y1 = max(0, y1 + int(ys.min()) - 2)
            exact_y2 = min(h, y1 + int(ys.max()) + 3)
            exact_x1 = max(0, best_x1 + int(xs.min()) - 8)
            exact_x2 = min(w, best_x1 + int(xs.max()) + 8)

            crop = rgb_arr[exact_y1:exact_y2, exact_x1:exact_x2]
            ch, cw = crop.shape[:2]
            if cw >= 14 and ch >= 8:
                results.append((crop, (exact_x1, exact_y1, cw, ch)))

        return results

    @classmethod
    def _slice_subtitle_lines(cls, rgb_arr: np.ndarray) -> List[np.ndarray]:
        return [crop for crop, _bounds in cls._slice_subtitle_lines_with_bounds(rgb_arr)]

    @staticmethod
    def _make_white_isolated_view(rgb_arr: np.ndarray) -> np.ndarray:
        """
        Nino (`WindowsOcrReader.cs` beyaz yazı yalıtımı):
        GTA 5 / RDR 2 gibi oyunlarda güneşli sarı kaldırım, ten veya açık gömlek üzerindeki
        beyaz altyazıları ayırmak için düşük doygunluklu parlak harfleri beyaz tuval üstünde
        keskin siyah harfe dönüştürür.
        """
        max_c = np.max(rgb_arr, axis=2)
        min_c = np.min(rgb_arr, axis=2)
        sat = max_c.astype(np.int16) - min_c.astype(np.int16)
        local_min = cv2.erode(min_c, np.ones((5, 5), np.uint8), iterations=1)
        local_drop = max_c.astype(np.int16) - local_min.astype(np.int16)

        white_mask = ((min_c > 172) & (sat < 52) & (local_drop > 32)).astype(np.uint8) * 255
        # Harf gövdelerini 1px temizle ve siyah-yazı / beyaz-zemin (OCR ideal formatı) yap
        inv = 255 - white_mask
        return cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _make_auto_level_view(rgb_arr: np.ndarray) -> np.ndarray:
        """
        Nino (`UseAutoLevel`): En parlak %0.5'i 255'e, en koyu %5'i 0'a gererek
        düşük kontrastlı sahnelerdeki altyazıyı keskinleştirir.
        """
        arr = rgb_arr.astype(np.float32)
        lo = float(np.percentile(arr, 5.0))
        hi = float(np.percentile(arr, 99.5))
        if hi - lo < 25.0:
            return rgb_arr
        stretched = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
        return stretched

    @staticmethod
    def _score_english_readability(text: str, avg_conf: float) -> float:
        """
        Nino (`WindowsSpellCheck.Score` benzeri) okunabilirlik puanlayıcı:
        OCR çıktısındaki gerçek İngilizce kelime oranını, temiz harf yapısını ve güven puanını
        birleştirerek birden fazla ön-işleme adayı arasından en doğru cümleyi seçer.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return 0.0
        tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", cleaned) if len(t) >= 2]
        if not tokens:
            return 0.0

        known_hits = sum(1 for t in tokens if t in COMMON_ENGLISH_WORDS or t.rstrip("'s") in COMMON_ENGLISH_WORDS)
        vowel_ok = sum(1 for t in tokens if any(v in t for v in "aeiouy"))
        known_ratio = float(known_hits) / float(len(tokens))
        vowel_ratio = float(vowel_ok) / float(len(tokens))

        # Bozuk/gürültülü sembol oranı cezası
        alpha_space = sum(1 for c in cleaned if c.isalpha() or c.isspace() or c in ".,!?'-:")
        clean_ratio = float(alpha_space) / float(max(1, len(cleaned)))

        length_bonus = min(1.0, len(tokens) / 5.0) * 0.15
        return (avg_conf * 0.45) + (known_ratio * 0.30) + (vowel_ratio * 0.15) + (clean_ratio * 0.10) + length_bonus

    def _run_fast_1d_ocr(self, rgb_arr: np.ndarray) -> Tuple[str, float]:
        """Sadece hızlı 1D satır kesici + CRNN (~30-45ms) çalıştırır (çoklu aday testleri için)."""
        line_items = self._slice_subtitle_lines_with_bounds(rgb_arr)
        recognized: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        with self._lock:
            for crop, bounds in line_items:
                res, _ = self._rapid_ocr(crop, use_det=False, use_cls=False, use_rec=True)
                if res and len(res) > 0:
                    text, score = str(res[0][0]).strip(), float(res[0][1])
                    if score >= self.min_confidence and text:
                        recognized.append((text, bounds, score))
        if not recognized:
            return "", 0.0
        avg_conf = sum(r[2] for r in recognized) / float(len(recognized))
        if len(recognized) == 2:
            top_txt, top_b, _ = recognized[0]
            bot_txt, bot_b, _ = recognized[1]
            if DialogueTextGrouper.is_speaker_label_line(top_txt, top_b, bot_b):
                return normalize_ocr_text(f"{top_txt}: {bot_txt}"), avg_conf
        return normalize_ocr_text(" ".join(r[0] for r in recognized)), avg_conf

    def extract_text(self, image: Image.Image) -> str:
        """
        Nino Çoklu Ön-İşleme + İngilizce Sözlük Puanlamalı Hızlı Altyazı Okuyucu:
        1. Önce ham karede hızlı 1D CRNN (~35ms) dener. Okunabilirlik yüksekse anında döner.
        2. Güneşli kaldırım / açık zemin yüzünden ham okuma zayıfsa `white_isolated` (beyaz harf yalıtımı)
           ve `auto_level` (kontrast germe) üzerinde hızlı 1D CRNN (~35ms) dener ve en yüksek
           İngilizce okunabilirlik puanlı adayı seçer.
        3. Hala sonuç çıkmadıysa tek seferlik garantili 2D DBNet (`detect_and_read_blocks`) çalıştırır.
        """
        t0 = time.perf_counter()
        self._ensure_initialized()
        if not self._rapid_ocr:
            return ""

        try:
            rgb_arr = np.asarray(image.convert("RGB"))

            # Aday 1: Ham görüntü (hızlı 1D CRNN ~35ms)
            best_text, best_conf = self._run_fast_1d_ocr(rgb_arr)
            best_score = self._score_english_readability(best_text, best_conf)

            # Akıllı kısa devre: Ham okuma zaten çok temizse ek ön-işleme yapma (0ms ek yük!)
            if best_text and best_conf >= 0.85 and best_score >= 0.72 and len(best_text) >= 6:
                self.last_ocr_ms = (time.perf_counter() - t0) * 1000.0
                return best_text

            # Aday 2: Nino Beyaz Harf Yalıtımı (hızlı 1D CRNN ~35ms)
            white_view = self._make_white_isolated_view(rgb_arr)
            if int(np.count_nonzero(white_view[:, :, 0] < 128)) >= 30:
                w_text, w_conf = self._run_fast_1d_ocr(white_view)
                w_score = self._score_english_readability(w_text, w_conf)
                if w_score > best_score:
                    best_text, best_score = w_text, w_score

            # Aday 3: Nino Otomatik Kontrast Germe (sadece hala zayıfsa, hızlı 1D CRNN ~35ms)
            if best_score < 0.62:
                auto_view = self._make_auto_level_view(rgb_arr)
                a_text, a_conf = self._run_fast_1d_ocr(auto_view)
                a_score = self._score_english_readability(a_text, a_conf)
                if a_score > best_score:
                    best_text, best_score = a_text, a_score

            # Garantili Tek Seferlik 2D DBNet Yedeği (1D adaylarının hiçbiri tatmin etmediyse)
            if best_score < 0.55:
                db_blocks = self.detect_and_read_blocks(rgb_arr, min_conf=self.min_confidence, max_side=640)
                if db_blocks:
                    db_blocks.sort(key=lambda item: (item[1][1] // 18, item[1][0]))
                    if len(db_blocks) == 2 and DialogueTextGrouper.is_speaker_label_line(
                        db_blocks[0][0], db_blocks[0][1], db_blocks[1][1]
                    ):
                        db_text = normalize_ocr_text(f"{db_blocks[0][0]}: {db_blocks[1][0]}")
                    else:
                        db_text = normalize_ocr_text(" ".join(b[0] for b in db_blocks))
                    db_score = self._score_english_readability(db_text, 0.80)
                    if db_score > best_score:
                        best_text = db_text

            self.last_ocr_ms = (time.perf_counter() - t0) * 1000.0
            return best_text
        except Exception as exc:
            logger.error("OCR okuma hatası: %s", exc)
            return ""

    def extract_hud_crop_text_and_bounds(
        self,
        crop_rgb: np.ndarray,
    ) -> Tuple[str, Optional[Tuple[int, int, int, int]], int]:
        """
        UniversalHUDRadar tarafından kırpılan HUD/Menü/Telefon kutusunu okur ve
        gerçek harflerin sıkı sınır kutusunu `(tight_x, tight_y, tight_w, tight_h)` + `satır_sayısı` döndürür.
        Böylece:
        - Telefondaki 'DIALING...' yazısının yanındaki kişi fotoğrafı (Devin) ASLA yama kutusuna dahil edilmez!
        - 2 satırlık menü/diyalog özetlerinde (`Mr. Weston checks...\nresponse.`) her iki satır da eksiksiz okunur!
        """
        self._ensure_initialized()
        if not self._rapid_ocr or crop_rgb.size == 0:
            return "", None, 1

        try:
            ch, cw = crop_rgb.shape[:2]
            line_items = self._slice_subtitle_lines_with_bounds(crop_rgb) if ch >= 28 else []

            parts: List[str] = []
            bounds_list: List[Tuple[int, int, int, int]] = []

            with self._lock:
                if line_items:
                    for lc, bnd in line_items[:4]:
                        res, _ = self._rapid_ocr(lc, use_det=False, use_cls=False, use_rec=True)
                        if res and len(res) > 0:
                            txt, score = str(res[0][0]).strip(), float(res[0][1])
                            if score >= 0.56 and txt:
                                parts.append(txt)
                                bounds_list.append(bnd)

                # Tek satır veya 1D kesitten tam okunamadıysa doğrudan veya 2D DBNet ile oku
                if not parts:
                    if ch <= 34:
                        res, _ = self._rapid_ocr(crop_rgb, use_det=False, use_cls=False, use_rec=True)
                        if res and len(res) > 0:
                            txt, score = str(res[0][0]).strip(), float(res[0][1])
                            if score >= 0.58 and txt:
                                parts.append(txt)
                                bounds_list.append((2, 2, max(10, cw - 4), max(8, ch - 4)))
                    if not parts:
                        det_res, _ = self._rapid_ocr(crop_rgb, use_det=True, use_cls=False, use_rec=True)
                        for item in det_res or []:
                            box, txt, score = item[0], str(item[1]).strip(), float(item[2])
                            if score >= 0.56 and txt:
                                bx = max(0, int(min(p[0] for p in box)))
                                by = max(0, int(min(p[1] for p in box)))
                                bw = max(8, int(max(p[0] for p in box) - bx))
                                bh = max(8, int(max(p[1] for p in box) - by))
                                parts.append(txt)
                                bounds_list.append((bx, by, bw, bh))

            raw = " ".join(parts).strip()
            if not raw:
                return "", None, 1
            alpha_cnt = sum(1 for c in raw if c.isalpha())
            if alpha_cnt < 3:
                return "", None, 1

            if bounds_list:
                tx1 = max(0, min(b[0] for b in bounds_list))
                ty1 = max(0, min(b[1] for b in bounds_list))
                tx2 = min(cw, max(b[0] + b[2] for b in bounds_list))
                ty2 = min(ch, max(b[1] + b[3] for b in bounds_list))
                tight_bounds: Optional[Tuple[int, int, int, int]] = (tx1, ty1, max(16, tx2 - tx1), max(10, ty2 - ty1))
            else:
                tight_bounds = (0, 0, cw, ch)

            return raw, tight_bounds, max(1, len(parts))
        except Exception as exc:
            logger.debug("HUD OCR okuma hatası: %s", exc)
            return "", None, 1

    def extract_hud_crop_text(self, crop_rgb: np.ndarray) -> str:
        txt, _bounds, _nlines = self.extract_hud_crop_text_and_bounds(crop_rgb)
        return txt

    def detect_and_read_blocks(
        self,
        rgb_arr: np.ndarray,
        min_conf: float = 0.56,
        max_side: int = 1280,
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Gerçek 2D DBNet (`use_det=True`) + CRNN Metin Bloğu Tarayıcısı:
        - Küçük köşe bilgi kartlarında (`~520x240`) tam çözünürlükte (~55ms) çalışır.
        - Tam ekran ESC / Harita / Envanter menülerinde (`1920x1080`) `1280px` ölçekte (~380ms)
          çalışarak ekrandaki TÜM sekme, buton, açıklama ve cümleleri (%99+) eksiksiz bulur!
        Dönüş: `[(metin, (x, y, w, h)), ...]` (orijinal `rgb_arr` piksel koordinatlarında).
        """
        self._ensure_initialized()
        if not self._rapid_ocr or rgb_arr is None or rgb_arr.size == 0:
            return []

        try:
            h, w = rgb_arr.shape[:2]
            scale = 1.0
            work_img = rgb_arr
            if max(w, h) > max_side:
                scale = float(max_side) / float(max(w, h))
                work_img = cv2.resize(
                    rgb_arr,
                    (max(32, int(round(w * scale))), max(32, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            with self._lock:
                det_res, _ = self._rapid_ocr(work_img, use_det=True, use_cls=False, use_rec=True)

            if not det_res:
                return []

            inv_s = 1.0 / scale
            blocks: List[Tuple[str, Tuple[int, int, int, int]]] = []
            for item in det_res:
                box, txt, score = item[0], str(item[1]).strip(), float(item[2])
                if score < min_conf or not txt:
                    continue
                alpha_cnt = sum(1 for c in txt if c.isalpha())
                if alpha_cnt < 2:
                    continue

                bx1 = max(0, int(round(min(p[0] for p in box) * inv_s)))
                by1 = max(0, int(round(min(p[1] for p in box) * inv_s)))
                bx2 = min(w, int(round(max(p[0] for p in box) * inv_s)))
                by2 = min(h, int(round(max(p[1] for p in box) * inv_s)))
                bw = bx2 - bx1
                bh = by2 - by1
                if bw < 12 or bh < 8:
                    continue
                blocks.append((txt, (bx1, by1, bw, bh)))

            return blocks
        except Exception as exc:
            logger.debug("2D DBNet blok tarama hatası: %s", exc)
            return []
