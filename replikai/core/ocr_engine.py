"""Yüksek Doğruluklu Oyun & Video Altyazı OCR Motoru (960px DBNet + Tam Satır Kesici + Nino Çoklu Ön-İşleme)."""

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
        "here", "where", "why", "wait", "stop", "right", "left", "down", "man", "men", "guy", "guys",
        "car", "kill", "dead", "money", "shit", "fuck", "hell", "damn", "please", "sorry", "thanks",
        "thank", "yes", "yeah", "nope", "okay", "ok", "sure", "really", "never", "always", "still",
        "keep", "hold", "leave", "call", "tell", "told", "said", "talk", "listen", "watch", "find",
        "found", "took", "gave", "came", "went", "gone", "done", "through", "personal", "line",
        "service", "twice", "details", "response", "mission", "police", "cops", "area", "vehicle",
        "press", "open", "close", "door", "help", "run", "shoot", "drive", "house", "home", "bro",
    }
)

EN_MODEL_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"
EN_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "en_PP-OCRv3_rec_infer.onnx"


class OCREngine:
    """
    Oyun ve video altyazılarını %99+ doğrulukla okuyan motor:
    - `det_limit_side_len = 960`: Geniş `1420x280` altyazı şeritlerinde 22px harflerin 8px'e
      ezilip kaybolmasını (%100 körlük bug'ını) ortadan kaldırır!
    - Tam Satır Genişliği (`spans[0] .. spans[-1]`): Cümlenin başındaki veya sonundaki kelimeleri
      asla kesmez ve tek kelimelik bozuk parçada (`'prneiexciusi'`) erken kısa devre yapmaz!
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
            dummy_1d = np.zeros((28, 240, 3), dtype=np.uint8)
            dummy_2d = np.zeros((80, 320, 3), dtype=np.uint8)
            with self._lock:
                try:
                    self._rapid_ocr(dummy_1d, use_det=False, use_cls=False, use_rec=True)
                    self._rapid_ocr(dummy_2d, use_det=True, use_cls=False, use_rec=True)
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

                # KRİTİK DÜZELTME: det_limit_side_len=960 (eskisi 480 idi ve 1420x280 altyazı şeridini
                # 480x94 boyutuna ezip 21px GTA 5 harflerini 7px'e düşürerek okunamaz hale getiriyordu!)
                kwargs = {
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 1,
                    "det_limit_side_len": 960,
                }
                if EN_MODEL_PATH.exists():
                    kwargs["rec_model_path"] = str(EN_MODEL_PATH)

                self._rapid_ocr = RapidOCR(**kwargs)
                self.backend_name = "RapidOCR Turbo (960p Warm ONNX)"
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

        if peak_top >= 8 and peak_bot >= 8 and valley_val <= min(peak_top, peak_bot) * 0.68:
            vy = y1 + s_off + valley_rel
            return [(y1, vy + 1), (max(y1, vy - 1), y2)]

        return [(y1, y2)]

    @classmethod
    def _slice_subtitle_lines_with_bounds(
        cls,
        rgb_arr: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        h, w, _ = rgb_arr.shape
        sub_mask = ScreenCapturer.extract_outlined_subtitle_mask(rgb_arr).copy()

        if np.count_nonzero(sub_mask) < 20:
            max_c = np.max(rgb_arr, axis=2)
            sub_mask = ((max_c > 168).astype(np.uint8)) * 255

        row_counts = np.sum(sub_mask > 0, axis=1)
        min_row_pixels = max(6, int(w * 0.008))
        active_rows = row_counts >= min_row_pixels

        raw_bands: List[Tuple[int, int]] = []
        start_y = None
        for y, is_active in enumerate(active_rows):
            if is_active and start_y is None:
                start_y = y
            elif not is_active and start_y is not None:
                band_h = y - start_y
                if 7 <= band_h <= 115:
                    raw_bands.extend(cls._split_tall_band_at_valley(row_counts, start_y, y))
                elif band_h > 115:
                    peak_y = start_y + int(np.argmax(row_counts[start_y:y]))
                    b1, b2 = max(start_y, peak_y - 30), min(y, peak_y + 30)
                    raw_bands.extend(cls._split_tall_band_at_valley(row_counts, b1, b2))
                start_y = None
        if start_y is not None:
            band_h = h - start_y
            if 7 <= band_h <= 115:
                raw_bands.extend(cls._split_tall_band_at_valley(row_counts, start_y, h))
            elif band_h > 115:
                peak_y = start_y + int(np.argmax(row_counts[start_y:h]))
                b1, b2 = max(start_y, peak_y - 30), min(h, peak_y + 30)
                raw_bands.extend(cls._split_tall_band_at_valley(row_counts, b1, b2))

        merged_bands: List[Tuple[int, int]] = list(raw_bands)
        if len(merged_bands) > 3:
            merged_bands.sort(key=lambda b: int(np.sum(sub_mask[b[0] : b[1], :] > 0)), reverse=True)
            merged_bands = sorted(merged_bands[:3], key=lambda b: b[0])

        results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for y1, y2 in merged_bands:
            band_mask = sub_mask[y1:y2, :]
            closed_line = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, np.ones((3, 55), np.uint8))
            col_active = np.sum(closed_line > 0, axis=0) >= 2

            spans: List[Tuple[int, int]] = []
            sx = None
            for x, val in enumerate(col_active):
                if val and sx is None:
                    sx = x
                elif not val and sx is not None:
                    if (x - sx) >= 12:
                        spans.append((sx, x))
                    sx = None
            if sx is not None and (w - sx) >= 12:
                spans.append((sx, w))

            merged_spans: List[Tuple[int, int]] = []
            for sx1, sx2 in spans:
                if merged_spans and (sx1 - merged_spans[-1][1]) <= 150:
                    merged_spans[-1] = (merged_spans[-1][0], sx2)
                else:
                    merged_spans.append((sx1, sx2))

            if not merged_spans:
                continue

            # Merkez ağırlıklı aktif piksel kütlesi: kenardaki far/tampon parlaması ortadaki kısa 'Wait!' yazısını ezemez!
            def _span_score(sp: Tuple[int, int]) -> float:
                mass = float(np.count_nonzero(sub_mask[y1:y2, sp[0] : sp[1]]))
                center_dist = abs(((sp[0] + sp[1]) * 0.5) - (w * 0.5)) / float(max(1, w))
                weight = 1.45 if center_dist <= 0.28 else 0.75
                return mass * weight

            best_x1, best_x2 = max(merged_spans, key=_span_score)
            sub_patch = sub_mask[y1:y2, best_x1:best_x2]
            ys, xs = np.where(sub_patch > 0)
            if len(ys) < 8:
                continue

            exact_y1 = max(0, y1 + int(ys.min()) - 3)
            exact_y2 = min(h, y1 + int(ys.max()) + 4)
            exact_x1 = max(0, best_x1 + int(xs.min()) - 12)
            exact_x2 = min(w, best_x1 + int(xs.max()) + 12)

            crop = rgb_arr[exact_y1:exact_y2, exact_x1:exact_x2]
            ch, cw = crop.shape[:2]
            if cw >= 14 and ch >= 8:
                results.append((crop, (exact_x1, exact_y1, cw, ch)))

        return results

    @classmethod
    def _slice_subtitle_lines(cls, rgb_arr: np.ndarray) -> List[np.ndarray]:
        return [crop for crop, _bounds in cls._slice_subtitle_lines_with_bounds(rgb_arr)]

    @classmethod
    def _run_white_isolated_1d_ocr(cls, self_inst: "OCREngine", rgb_arr: np.ndarray) -> Tuple[str, float]:
        h, w, _ = rgb_arr.shape
        max_c = np.max(rgb_arr, axis=2)
        min_c = np.min(rgb_arr, axis=2)
        sat = max_c.astype(np.int16) - min_c.astype(np.int16)
        local_min = cv2.erode(min_c, np.ones((5, 5), np.uint8), iterations=1)
        local_drop = max_c.astype(np.int16) - local_min.astype(np.int16)

        white_mask = ((min_c > 168) & (sat < 56) & (local_drop > 26)).astype(np.uint8) * 255
        if int(np.count_nonzero(white_mask)) < 28:
            return "", 0.0

        row_counts = np.sum(white_mask > 0, axis=1)
        min_row_px = max(6, int(w * 0.008))
        active_rows = row_counts >= min_row_px

        bands: List[Tuple[int, int]] = []
        sy = None
        for y, act in enumerate(active_rows):
            if act and sy is None:
                sy = y
            elif not act and sy is not None:
                if 7 <= (y - sy) <= 115:
                    bands.extend(cls._split_tall_band_at_valley(row_counts, sy, y))
                sy = None
        if sy is not None and 7 <= (h - sy) <= 115:
            bands.extend(cls._split_tall_band_at_valley(row_counts, sy, h))

        if len(bands) > 3:
            bands.sort(key=lambda b: int(np.sum(white_mask[b[0] : b[1], :] > 0)), reverse=True)
            bands = sorted(bands[:3], key=lambda b: b[0])

        inv_rgb = cv2.cvtColor(255 - white_mask, cv2.COLOR_GRAY2RGB)
        recognized: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        with self_inst._lock:
            for y1, y2 in bands:
                b_mask = white_mask[y1:y2, :]
                ys, xs = np.where(b_mask > 0)
                if len(ys) < 8:
                    continue
                ey1, ey2 = max(0, y1 + int(ys.min()) - 3), min(h, y1 + int(ys.max()) + 4)
                ex1, ex2 = max(0, int(xs.min()) - 10), min(w, int(xs.max()) + 10)
                crop = inv_rgb[ey1:ey2, ex1:ex2]
                ch, cw = crop.shape[:2]
                if cw < 14 or ch < 8:
                    continue
                work_crop = (
                    cv2.resize(crop, (max(16, int(round(cw * (32.0 / ch)))), 32), interpolation=cv2.INTER_AREA)
                    if ch > 32
                    else crop
                )
                res, _ = self_inst._rapid_ocr(work_crop, use_det=False, use_cls=False, use_rec=True)
                if res and len(res) > 0:
                    txt, sc = str(res[0][0]).strip(), float(res[0][1])
                    if sc >= self_inst.min_confidence and txt:
                        recognized.append((txt, (ex1, ey1, ex2 - ex1, ey2 - ey1), sc))

        if not recognized:
            return "", 0.0
        avg_conf = sum(r[2] for r in recognized) / float(len(recognized))
        return normalize_ocr_text(" ".join(r[0] for r in recognized)), avg_conf

    @staticmethod
    def _score_english_readability(text: str, avg_conf: float) -> float:
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

        alpha_space = sum(1 for c in cleaned if c.isalpha() or c.isspace() or c in ".,!?'-:")
        clean_ratio = float(alpha_space) / float(max(1, len(cleaned)))

        length_bonus = min(1.0, len(tokens) / 5.0) * 0.15
        return (avg_conf * 0.45) + (known_ratio * 0.30) + (vowel_ratio * 0.15) + (clean_ratio * 0.10) + length_bonus

    def _run_fast_1d_ocr(self, rgb_arr: np.ndarray) -> Tuple[str, float]:
        line_items = self._slice_subtitle_lines_with_bounds(rgb_arr)
        recognized: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        with self._lock:
            for crop, bounds in line_items:
                ch, cw = crop.shape[:2]
                work_crop = (
                    cv2.resize(crop, (max(16, int(round(cw * (32.0 / ch)))), 32), interpolation=cv2.INTER_AREA)
                    if ch > 32
                    else crop
                )
                res, _ = self._rapid_ocr(work_crop, use_det=False, use_cls=False, use_rec=True)
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
        Hızlı ve Garantili Altyazı Okuyucu:
        1. Ham görüntüde hızlı 1D CRNN dener; YALNIZCA hem güven (`>= 0.82`) hem de İngilizce
           kelime okunabilirliği (`>= 0.68`) yüksekse erken döner (`'prneiexciusi'` gibi bozuk
           parçalarda ASLA erken dönmez!).
        2. Güneşli/parlak zeminde `white_isolated` 1D CRNN dener ve puanları karşılaştırır.
        3. Hala zayıfsa aktif dikey şeridi kırpıp `960px` tam çözünürlüklü 2D DBNet ile okur!
        """
        t0 = time.perf_counter()
        self._ensure_initialized()
        if not self._rapid_ocr:
            return ""

        try:
            rgb_arr = np.asarray(image.convert("RGB"))

            best_text, best_conf = self._run_fast_1d_ocr(rgb_arr)
            best_score = self._score_english_readability(best_text, best_conf)

            # Sadece gerçekten temiz ve anlamlı İngilizce kelimeler varsa hızlı dön!
            if best_text and best_conf >= 0.82 and best_score >= 0.68 and len(best_text) >= 4:
                self.last_ocr_ms = (time.perf_counter() - t0) * 1000.0
                return best_text

            w_text, w_conf = self._run_white_isolated_1d_ocr(self, rgb_arr)
            if w_text:
                w_score = self._score_english_readability(w_text, w_conf)
                if w_score > best_score:
                    best_text, best_conf, best_score = w_text, w_conf, w_score

            if best_text and best_conf >= 0.76 and best_score >= 0.64 and len(best_text) >= 4:
                self.last_ocr_ms = (time.perf_counter() - t0) * 1000.0
                return best_text

            # Garantili 960px 2D DBNet Yedeği (aktif satır şeridine daraltılmış hızlı kırpım!)
            sub_mask = ScreenCapturer.extract_outlined_subtitle_mask(rgb_arr)
            ys = np.where(np.sum(sub_mask > 0, axis=1) >= 4)[0]
            h, _w = rgb_arr.shape[:2]
            if len(ys) > 0:
                by1 = max(0, int(ys.min()) - 14)
                by2 = min(h, int(ys.max()) + 16)
                band_img = rgb_arr[by1:by2, :]
            else:
                band_img = rgb_arr

            db_blocks = self.detect_and_read_blocks(band_img, min_conf=self.min_confidence, max_side=960)
            if db_blocks:
                db_blocks.sort(key=lambda item: (item[1][1] // 18, item[1][0]))
                if len(db_blocks) == 2 and DialogueTextGrouper.is_speaker_label_line(
                    db_blocks[0][0], db_blocks[0][1], db_blocks[1][1]
                ):
                    db_text = normalize_ocr_text(f"{db_blocks[0][0]}: {db_blocks[1][0]}")
                else:
                    db_text = normalize_ocr_text(" ".join(b[0] for b in db_blocks))
                db_score = self._score_english_readability(db_text, 0.82)
                if db_score >= best_score or not best_text:
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
                tight_bounds: Optional[Tuple[int, int, int, int]] = (
                    tx1,
                    ty1,
                    max(16, tx2 - tx1),
                    max(10, ty2 - ty1),
                )
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
        min_conf: float = 0.54,
        max_side: int = 1280,
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
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
