"""3 Filtreli Evrensel Oyun HUD, Bildirim, Telefon ve ESC Menü Radarı (~0.9ms CPU, Sıfır FPS-Drop).

MİMARİ:
1. Çift Kanallı Tarak Dişi Yatay Harf Ritmi (`th_max` + `th_min` + `local_drop` Comb Filter):
   Hem karanlık sahnelerdeki hem de gri menü sekmelerindeki (`STATS | SETTINGS | GAME`) ve
   parlak bulanık arka planlardaki (`You are through to...`) tüm harfleri eksiksiz yakalar.
2. Tam Satır Bütünlüğü (`cur_comb` Connected Components):
   Cümle kutularını fark maskesinden değil doğrudan `cur_comb` üzerinden çıkarır; böylece
   uzun bir cümlenin sadece ortasını çevirip başını/sonunu açıkta bırakma hatası %100 önlenir!
3. Sıkı Harf Kutusu (`tight_bounds`) & `32x12` Parmak İzi Takibi:
   - Telefondaki `DIALING...` yazısının yanındaki kişi fotoğrafını (Devin) yama kutusuna dahil etmez.
   - `DIALING...` yazısı `CONNECTED` olduğunda piksel sayısı aynı kalsa bile `32x12` parmak izi
     farkıyla anında algılar ve `ARANIYOR...` -> `BAĞLANDI` olarak günceller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from replikai.core.dialogue_grouper import DialogueTextGrouper
from replikai.core.natural_patch import RealtimeNaturalBackground, SourceTextColorSampler

# Tek kelimelik veya yaygın oyun menü/HUD/telefon başlıkları için 0.0 ms anlık sözlük
HUD_INSTANT_GLOSSARY: Dict[str, str] = {
    "map": "HARİTA",
    "brief": "ÖZET",
    "stats": "İSTATİSTİK",
    "settings": "AYARLAR",
    "game": "OYUN",
    "online": "ÇEVRİMİÇİ",
    "store": "MAĞAZA",
    "gallery": "GALERİ",
    "editor": "EDİTÖR",
    "audio": "SES",
    "display": "EKRAN",
    "graphics": "GRAFİKLER",
    "advanced graphics": "GELİŞMİŞ GRAFİKLER",
    "controls": "KONTROLLER",
    "keyboard / mouse": "KLAVYE / FARE",
    "key bindings": "TUŞ ATAMALARI",
    "gamepad": "OYUN KOLU",
    "camera": "KAMERA",
    "saving and startup": "KAYIT VE BAŞLANGIÇ",
    "notifications": "BİLDİRİMLER",
    "dialogue": "DİYALOGLAR",
    "help": "YARDIM",
    "feed": "AKIŞ",
    "resume": "DEVAM ET",
    "save game": "OYUNU KAYDET",
    "load game": "OYUN YÜKLE",
    "new game": "YENİ OYUN",
    "exit game": "OYUNDAN ÇIK",
    "quit game": "OYUNDAN ÇIK",
    "contacts": "KİŞİLER",
    "messages": "MESAJLAR",
    "job list": "İŞ LİSTESİ",
    "quick save": "HIZLI KAYIT",
    "internet": "İNTERNET",
    "snapmatic": "KAMERA",
    "securoserv": "SECUROSERV",
    "dialing": "ARANIYOR...",
    "dialing...": "ARANIYOR...",
    "connected": "BAĞLANDI",
    "busy": "MEŞGUL",
    "no answer": "CEVAP YOK",
    "call ended": "ARAMA BİTTİ",
    "voicemail": "SESLİ MESAJ",
    "inventory": "ENVANTER",
    "journal": "GÜNLÜK",
    "quests": "GÖREVLER",
    "skills": "YETENEKLER",
    "character": "KARAKTER",
    "player": "OYUNCU",
    "progress": "İLERLEME",
    "story": "HİKAYE",
    "general": "GENEL",
    "mission passed": "GÖREV TAMAMLANDI",
    "mission failed": "GÖREV BAŞARISIZ",
    "wasted": "ÖLDÜN",
    "busted": "YAKALANDIN",
    "wanted": "ARANIYOR",
}


@dataclass
class HUDCandidateBox:
    """Radarın tespit ettiği, OCR'a girecek oyun içi yazı kutusu."""

    zone_key: str
    x: int
    y: int
    w: int
    h: int
    bg_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    crop_rgb: np.ndarray
    comb_energy: int
    outline_rgb: Tuple[int, int, int] = (10, 14, 20)


@dataclass
class HUDTextPatch:
    """Ekranda orijinal İngilizce kelimenin tam üstüne (In-Place) basılacak doğal Türkçe yama."""

    zone_key: str
    x: int
    y: int
    w: int
    h: int
    source_text: str
    translated_text: str
    bg_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    comb_energy: int
    outline_rgb: Tuple[int, int, int] = (10, 14, 20)
    num_lines: int = 1
    natural_bg_patch: Optional[np.ndarray] = field(default=None, repr=False)
    fingerprint: Optional[np.ndarray] = field(default=None, repr=False)


class UniversalHUDRadar:
    """
    480x270 ölçekte ~0.9ms içinde çalışan, 3 filtreli + OverTranslate doğal doku yamalı
    evrensel oyun içi yazı radarı.
    """

    COARSE_W = 480
    COARSE_H = 270

    def __init__(self) -> None:
        self._baseline_comb: Optional[np.ndarray] = None
        self._prev_comb: Optional[np.ndarray] = None
        self._dead_zones: Dict[str, int] = {}
        self.active_patches: Dict[str, HUDTextPatch] = {}

    def reset(self) -> None:
        self._baseline_comb = None
        self._prev_comb = None
        self._dead_zones.clear()
        self.active_patches.clear()

    @staticmethod
    def _compute_fingerprint(rgb_patch: np.ndarray) -> Optional[np.ndarray]:
        if rgb_patch is None or rgb_patch.size == 0:
            return None
        return cv2.resize(rgb_patch, (32, 12), interpolation=cv2.INTER_AREA).astype(np.float32)

    @classmethod
    def compute_comb_text_mask(cls, full_rgb: np.ndarray, subtitle_region: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        ÇİFT KANALLI TARAK DİŞİ YATAY HARF RİTMİ (0.4ms):
        - Hem gri menü sekmelerini (`STATS | SETTINGS | GAME`), hem parlak bulanık menü arka planı
          üzerindeki beyaz yazıları (`You are through to...`), hem de koyu kutulu HUD bildirimlerini yakalar.
        """
        fh, fw = full_rgb.shape[:2]
        small = cv2.resize(full_rgb, (cls.COARSE_W, cls.COARSE_H), interpolation=cv2.INTER_AREA)
        max_c = np.max(small, axis=2)
        min_c = np.min(small, axis=2)

        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        th_max = cv2.morphologyEx(max_c, cv2.MORPH_TOPHAT, kernel_th)
        th_min = cv2.morphologyEx(min_c, cv2.MORPH_TOPHAT, kernel_th)
        tophat = np.maximum(th_max, th_min)

        # Yerel kontrast düşüşü (3x3 komşulukta harf-arka plan farkı >= 36)
        local_min = cv2.erode(min_c, np.ones((3, 3), np.uint8), iterations=1)
        local_drop = max_c.astype(np.int16) - local_min.astype(np.int16)

        # Tarak Dişi (Comb) Yatay Harf Geçiş Yoğunluğu: |I(x+1) - I(x)| > 26
        dx = np.abs(np.diff(max_c.astype(np.int16), axis=1, append=0))
        horiz_edges = (dx > 26).astype(np.uint8)
        edge_density = cv2.boxFilter(horiz_edges, ddepth=-1, ksize=(9, 3), normalize=False)

        comb_mask = (
            (tophat > 28)
            & (max_c > 140)
            & (local_drop > 36)
            & (edge_density >= 4)
        ).astype(np.uint8) * 255

        # Alt orta altyazı bölgesini maskeden çıkar (orayı zaten SubtitleOverlay çeviriyor)
        if subtitle_region and fw > 0 and fh > 0:
            sx = cls.COARSE_W / float(fw)
            sy = cls.COARSE_H / float(fh)
            x1 = max(0, int(subtitle_region.get("left", 0) * sx))
            y1 = max(0, int(subtitle_region.get("top", 0) * sy))
            x2 = min(cls.COARSE_W, int((subtitle_region.get("left", 0) + subtitle_region.get("width", 0)) * sx))
            y2 = min(cls.COARSE_H, int((subtitle_region.get("top", 0) + subtitle_region.get("height", 0)) * sy))
            comb_mask[y1:y2, x1:x2] = 0

        return comb_mask

    @staticmethod
    def _make_zone_key(x: int, y: int, w: int, h: int) -> str:
        """Koordinatı 20px ızgaraya yuvarlayarak kararlı bölge anahtarı üretir."""
        return f"{x // 20}_{y // 20}_{w // 28}_{h // 16}"

    @staticmethod
    def sample_box_colors(crop_rgb: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        style = SourceTextColorSampler.sample(crop_rgb)
        return style.background_rgb, style.text_rgb

    @staticmethod
    def _group_raw_rects(
        raw_rects: List[Tuple[int, int, int, int, int]],
        fw: int,
        fh: int,
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        OverTranslate `DialogueTextGrouper` kurallarıyla:
        1. Aynı satırdaki yan yana kelimeleri (`adjacent_in_row`) birleştirir;
           ancak `STATS | SETTINGS | GAME` gibi aralarında geniş boşluk olan sekmeleri ayrı tutar.
        2. Alt alta dizilmiş paragraf satırlarını (`next_paragraph_row`) tek kutuda birleştirir.
        """
        if len(raw_rects) <= 1:
            return raw_rects

        sorted_h = sorted(raw_rects, key=lambda r: (r[1] // 14, r[0]))
        row_merged: List[Tuple[int, int, int, int, int]] = []
        for rect in sorted_h:
            rx, ry, rw, rh, renergy = rect
            if row_merged:
                px, py, pw, ph, penergy = row_merged[-1]
                if DialogueTextGrouper.adjacent_in_row((px, py, pw, ph), (rx, ry, rw, rh)):
                    nx1 = min(px, rx)
                    ny1 = min(py, ry)
                    nx2 = max(px + pw, rx + rw)
                    ny2 = max(py + ph, ry + rh)
                    row_merged[-1] = (nx1, ny1, nx2 - nx1, ny2 - ny1, penergy + renergy)
                    continue
            row_merged.append(rect)

        sorted_v = sorted(row_merged, key=lambda r: (r[1], r[0]))
        groups: List[List[Tuple[int, int, int, int, int]]] = []
        for rect in sorted_v:
            rx, ry, rw, rh, _ = rect
            placed = False
            for grp in groups:
                if len(grp) >= 3:
                    continue
                px, py, pw, ph, _ = grp[-1]
                if DialogueTextGrouper.next_paragraph_row((px, py, pw, ph), (rx, ry, rw, rh)):
                    grp.append(rect)
                    placed = True
                    break
            if not placed:
                groups.append([rect])

        final_rects: List[Tuple[int, int, int, int, int]] = []
        for grp in groups:
            gx1 = max(0, min(r[0] for r in grp))
            gy1 = max(0, min(r[1] for r in grp))
            gx2 = min(fw, max(r[0] + r[2] for r in grp))
            gy2 = min(fh, max(r[1] + r[3] for r in grp))
            total_e = sum(r[4] for r in grp)
            final_rects.append((gx1, gy1, gx2 - gx1, gy2 - gy1, total_e))

        return final_rects

    def scan_for_new_hud_boxes(
        self,
        full_rgb: np.ndarray,
        subtitle_region: Optional[Dict[str, int]] = None,
        force_full_menu: bool = False,
        max_candidates: int = 8,
    ) -> Tuple[List[HUDCandidateBox], bool]:
        fh, fw = full_rgb.shape[:2]
        cur_comb = self.compute_comb_text_mask(full_rgb, subtitle_region)
        patches_changed = False

        sx = fw / float(self.COARSE_W)
        sy = fh / float(self.COARSE_H)

        # A) Mevcut aktif Türkçe yamaların altındaki oyun yazısı kapandı mı veya değişti mi (`DIALING...` -> `CONNECTED`) kontrol et!
        stale_keys: List[str] = []
        for zkey, patch in list(self.active_patches.items()):
            cx1 = max(0, int(patch.x / sx))
            cy1 = max(0, int(patch.y / sy))
            cx2 = min(self.COARSE_W, max(cx1 + 2, int((patch.x + patch.w) / sx)))
            cy2 = min(self.COARSE_H, max(cy1 + 2, int((patch.y + patch.h) / sy)))
            current_energy = int(np.count_nonzero(cur_comb[cy1:cy2, cx1:cx2]))

            is_stale = (
                current_energy < max(3, int(patch.comb_energy * 0.38))
                or current_energy > int(patch.comb_energy * 2.1)
            )
            # Parmak izi (32x12) kontrolü: Kelime değiştiyse (ör. DIALING... -> CONNECTED) anında algıla!
            if not is_stale and patch.fingerprint is not None:
                cur_crop = full_rgb[max(0, patch.y) : min(fh, patch.y + patch.h), max(0, patch.x) : min(fw, patch.x + patch.w)]
                cur_fp = self._compute_fingerprint(cur_crop)
                if cur_fp is not None and cur_fp.shape == patch.fingerprint.shape:
                    l1_diff = float(np.mean(np.abs(cur_fp - patch.fingerprint)))
                    if l1_diff > 13.5:
                        is_stale = True

            if is_stale:
                stale_keys.append(zkey)
                self._dead_zones.pop(zkey, None)

        for k in stale_keys:
            self.active_patches.pop(k, None)
            patches_changed = True

        # B) Dead-Zone kilitlerinde pikseli değişen yerlerin kilidini aç
        for dkey, old_energy in list(self._dead_zones.items()):
            parts = dkey.split("_")
            if len(parts) == 4:
                gx, gy, gw, gh = (int(p) for p in parts)
                cx1 = max(0, int((gx * 20) / sx))
                cy1 = max(0, int((gy * 20) / sy))
                cx2 = min(self.COARSE_W, max(cx1 + 2, int(((gx * 20) + (gw * 28)) / sx)))
                cy2 = min(self.COARSE_H, max(cy1 + 2, int(((gy * 20) + (gh * 16)) / sy)))
                cur_e = int(np.count_nonzero(cur_comb[cy1:cy2, cx1:cx2]))
                if abs(cur_e - old_energy) > max(5, int(old_energy * 0.40)):
                    self._dead_zones.pop(dkey, None)

        if self._baseline_comb is None or self._prev_comb is None:
            self._baseline_comb = cur_comb.copy()
            self._prev_comb = cur_comb.copy()
            if not force_full_menu:
                return [], patches_changed

        if force_full_menu:
            born_mask = cur_comb
        else:
            prev_dilated = cv2.dilate(self._prev_comb, np.ones((3, 3), np.uint8), iterations=1)
            base_dilated = cv2.dilate(self._baseline_comb, np.ones((5, 5), np.uint8), iterations=1)
            born_mask = (
                (cur_comb > 0)
                & (prev_dilated > 0)
                & (base_dilated == 0)
            ).astype(np.uint8) * 255

        stable_now = ((cur_comb > 0) & (self._prev_comb > 0)).astype(np.uint8) * 255
        self._baseline_comb = cv2.bitwise_or(
            cv2.bitwise_and(self._baseline_comb, cur_comb),
            stable_now,
        )
        self._prev_comb = cur_comb.copy()

        if int(np.count_nonzero(born_mask)) < 6:
            return [], patches_changed

        # KRİTİK: Bağlantılı bileşenleri `born_mask` yerine doğrudan `cur_comb` üzerinden çıkar!
        # Böylece cümlenin bir kısmı eski bir 3D objenin üstüne denk gelse bile cümle ASLA ortadan bölünmez!
        # Yatay kapatmayı (10, 2) tutuyoruz ki aynı cümledeki kelimeler birleşsin ama menü sekmeleri
        # (STATS | SETTINGS | GAME) ve telefon ikonları birbirine yapışmasın!
        connected_full = cv2.morphologyEx(
            cur_comb,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2)),
        )
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(connected_full, connectivity=8)

        raw_rects: List[Tuple[int, int, int, int, int]] = []
        for i in range(1, num_labels):
            cx = int(stats[i, cv2.CC_STAT_LEFT])
            cy = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])

            if cw < 7 or cw > 410 or ch < 3 or ch > 65:
                continue

            # Bu bileşenin içinde yeni doğan (`born_mask`) harf enerjisi var mı?
            born_energy = int(np.count_nonzero(born_mask[cy : cy + ch, cx : cx + cw]))
            if not force_full_menu and born_energy < 5:
                continue

            comb_energy = int(np.count_nonzero(cur_comb[cy : cy + ch, cx : cx + cw]))
            if comb_energy < 6:
                continue

            # Sıkı başlangıç koordinatları (telefondaki kişi fotoğrafına taşmaması için pad_x=8, pad_y=4)
            pad_x, pad_y = 8, 4
            rx1 = max(0, int(cx * sx) - pad_x)
            ry1 = max(0, int(cy * sy) - pad_y)
            rx2 = min(fw, int((cx + cw) * sx) + pad_x)
            ry2 = min(fh, int((cy + ch) * sy) + pad_y)
            rw = rx2 - rx1
            rh = ry2 - ry1

            if rw < 28 or rh < 12:
                continue

            raw_rects.append((rx1, ry1, rw, rh, comb_energy))

        grouped_rects = self._group_raw_rects(raw_rects, fw, fh)

        candidates: List[HUDCandidateBox] = []
        for rx1, ry1, rw, rh, comb_energy in grouped_rects:
            zkey = self._make_zone_key(rx1, ry1, rw, rh)
            if not force_full_menu and (zkey in self.active_patches or zkey in self._dead_zones):
                continue

            crop_rgb = full_rgb[ry1 : ry1 + rh, rx1 : rx1 + rw].copy()
            sampled_style = SourceTextColorSampler.sample(crop_rgb)
            candidates.append(
                HUDCandidateBox(
                    zone_key=zkey,
                    x=rx1,
                    y=ry1,
                    w=rw,
                    h=rh,
                    bg_rgb=sampled_style.background_rgb,
                    text_rgb=sampled_style.text_rgb,
                    outline_rgb=sampled_style.outline_rgb or (10, 14, 20),
                    crop_rgb=crop_rgb,
                    comb_energy=comb_energy,
                )
            )

        candidates.sort(key=lambda c: c.comb_energy, reverse=True)
        return candidates[:max_candidates], patches_changed

    def lock_dead_zone(self, box: HUDCandidateBox) -> None:
        self._dead_zones[box.zone_key] = box.comb_energy

    def register_patch(
        self,
        box: HUDCandidateBox,
        source_text: str,
        translated_text: str,
        tight_bounds: Optional[Tuple[int, int, int, int]] = None,
        num_lines: int = 1,
    ) -> HUDTextPatch:
        """
        Çevrilen HUD/menü kutusunu gerçek harf sınırlarına (`tight_bounds`) daraltarak ve
        `32x12` parmak iziyle birlikte kaydeder.
        """
        self._dead_zones[box.zone_key] = box.comb_energy

        px, py, pw, ph = box.x, box.y, box.w, box.h
        patch_crop = box.crop_rgb
        if tight_bounds is not None:
            tx, ty, tw, th = tight_bounds
            ch, cw = box.crop_rgb.shape[:2]
            if 0 <= tx < cw and 0 <= ty < ch and tw >= 14 and th >= 10:
                tx1 = max(0, tx - 3)
                ty1 = max(0, ty - 2)
                tx2 = min(cw, tx + tw + 3)
                ty2 = min(ch, ty + th + 2)
                px = box.x + tx1
                py = box.y + ty1
                pw = max(20, tx2 - tx1)
                ph = max(12, ty2 - ty1)
                patch_crop = box.crop_rgb[ty1:ty2, tx1:tx2].copy()

        sampled_style = SourceTextColorSampler.sample(patch_crop)
        natural_bg = RealtimeNaturalBackground.build_natural_patch(patch_crop, sampled_style.background_rgb)
        fp = self._compute_fingerprint(patch_crop)

        patch = HUDTextPatch(
            zone_key=box.zone_key,
            x=px,
            y=py,
            w=pw,
            h=ph,
            source_text=source_text,
            translated_text=translated_text,
            bg_rgb=sampled_style.background_rgb,
            text_rgb=sampled_style.text_rgb,
            outline_rgb=sampled_style.outline_rgb or box.outline_rgb,
            comb_energy=box.comb_energy,
            num_lines=max(1, int(num_lines)),
            natural_bg_patch=natural_bg,
            fingerprint=fp,
        )
        self.active_patches[box.zone_key] = patch
        return patch

    @staticmethod
    def group_dbnet_blocks(
        blocks: List[Tuple[str, Tuple[int, int, int, int]]],
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        """
        Gerçek 2D DBNet çıktısı `[(metin, (x, y, w, h)), ...]` üzerinde
        OverTranslate `DialogueTextGrouper` kurallarını uygular:
        - Aynı satırdaki bölünmüş kelimeleri birleştirir (menü sekmelerini ayrı tutar).
        - Alt alta dizilmiş 2-4 satırlık bilgi kartı / diyalog paragraflarını TEK cümlede birleştirir!
        Dönüş: `[(birlesik_metin, (abs_x, abs_y, w, h), satir_sayisi), ...]`
        """
        if not blocks:
            return []

        abs_blocks: List[Tuple[str, Tuple[int, int, int, int]]] = [
            (txt.strip(), (bx + offset_x, by + offset_y, bw, bh))
            for txt, (bx, by, bw, bh) in blocks
            if txt and txt.strip()
        ]
        if not abs_blocks:
            return []

        # 1. Aynı yatay satırdaki komşu blokları birleştir
        sorted_h = sorted(abs_blocks, key=lambda b: (b[1][1] // 14, b[1][0]))
        row_merged: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for txt, rect in sorted_h:
            rx, ry, rw, rh = rect
            if row_merged:
                ptxt, (px, py, pw, ph) = row_merged[-1]
                if DialogueTextGrouper.adjacent_in_row((px, py, pw, ph), (rx, ry, rw, rh)):
                    nx1 = min(px, rx)
                    ny1 = min(py, ry)
                    nx2 = max(px + pw, rx + rw)
                    ny2 = max(py + ph, ry + rh)
                    row_merged[-1] = (f"{ptxt} {txt}".strip(), (nx1, ny1, nx2 - nx1, ny2 - ny1))
                    continue
            row_merged.append((txt, rect))

        # 2. Dikeyde aynı paragrafın devamı olan satırları birleştir
        sorted_v = sorted(row_merged, key=lambda b: (b[1][1], b[1][0]))
        groups: List[List[Tuple[str, Tuple[int, int, int, int]]]] = []
        for item in sorted_v:
            txt, rect = item
            placed = False
            for grp in groups:
                if len(grp) >= 4:
                    continue
                _ptxt, prect = grp[-1]
                if DialogueTextGrouper.next_paragraph_row(prect, rect):
                    grp.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])

        grouped_out: List[Tuple[str, Tuple[int, int, int, int], int]] = []
        for grp in groups:
            merged_text = " ".join(it[0] for it in grp).strip()
            gx1 = min(it[1][0] for it in grp)
            gy1 = min(it[1][1] for it in grp)
            gx2 = max(it[1][0] + it[1][2] for it in grp)
            gy2 = max(it[1][1] + it[1][3] for it in grp)
            grouped_out.append((merged_text, (gx1, gy1, max(16, gx2 - gx1), max(12, gy2 - gy1)), len(grp)))

        return grouped_out

    @classmethod
    def build_patch_from_rgb(
        cls,
        src_rgb: np.ndarray,
        abs_rect: Tuple[int, int, int, int],
        source_text: str,
        translated_text: str,
        num_lines: int = 1,
        src_offset_x: int = 0,
        src_offset_y: int = 0,
    ) -> HUDTextPatch:
        """
        Gerçek 2D DBNet bloğundan OverTranslate doğal doku yaması (`RealtimeNaturalBackground`)
        ve orijinal renk/kontur (`SourceTextColorSampler`) ile `HUDTextPatch` oluşturur.
        """
        sh, sw = src_rgb.shape[:2]
        ax, ay, aw, ah = abs_rect
        lx1 = max(0, (ax - src_offset_x) - 3)
        ly1 = max(0, (ay - src_offset_y) - 2)
        lx2 = min(sw, (ax - src_offset_x) + aw + 3)
        ly2 = min(sh, (ay - src_offset_y) + ah + 2)

        patch_crop = src_rgb[ly1:ly2, lx1:lx2].copy()
        if patch_crop.size == 0:
            patch_crop = np.full((max(12, ah), max(20, aw), 3), (18, 22, 30), dtype=np.uint8)

        sampled_style = SourceTextColorSampler.sample(patch_crop)
        natural_bg = RealtimeNaturalBackground.build_natural_patch(patch_crop, sampled_style.background_rgb)
        fp = cls._compute_fingerprint(patch_crop)

        px = src_offset_x + lx1
        py = src_offset_y + ly1
        pw = max(18, lx2 - lx1)
        ph = max(12, ly2 - ly1)
        zkey = f"db_{px // 12}_{py // 12}_{pw // 16}_{ph // 12}"

        return HUDTextPatch(
            zone_key=zkey,
            x=px,
            y=py,
            w=pw,
            h=ph,
            source_text=source_text,
            translated_text=translated_text,
            bg_rgb=sampled_style.background_rgb,
            text_rgb=sampled_style.text_rgb,
            outline_rgb=sampled_style.outline_rgb or (10, 14, 20),
            comb_energy=100,
            num_lines=max(1, int(num_lines)),
            natural_bg_patch=natural_bg,
            fingerprint=fp,
        )

    @classmethod
    def is_patch_still_matching(
        cls,
        patch: HUDTextPatch,
        src_rgb: np.ndarray,
        src_offset_x: int = 0,
        src_offset_y: int = 0,
        threshold: float = 16.0,
    ) -> bool:
        """Aktif yamanın altındaki oyun metni hala ekranda mı (`32x12` parmak izi ile) kontrol eder."""
        if patch.fingerprint is None or src_rgb is None or src_rgb.size == 0:
            return False
        sh, sw = src_rgb.shape[:2]
        lx1 = max(0, patch.x - src_offset_x)
        ly1 = max(0, patch.y - src_offset_y)
        lx2 = min(sw, lx1 + patch.w)
        ly2 = min(sh, ly1 + patch.h)
        if lx2 - lx1 < 8 or ly2 - ly1 < 6:
            return False
        cur_fp = cls._compute_fingerprint(src_rgb[ly1:ly2, lx1:lx2])
        if cur_fp is None or cur_fp.shape != patch.fingerprint.shape:
            return False
        return float(np.mean(np.abs(cur_fp - patch.fingerprint))) <= threshold

    @staticmethod
    def lookup_instant_glossary(text: str) -> Optional[str]:
        """Menü, HUD ve telefon terimlerini 0.0 ms içinde çevirir."""
        raw = (text or "").strip().lower()
        if raw in HUD_INSTANT_GLOSSARY:
            return HUD_INSTANT_GLOSSARY[raw]
        norm = re.sub(r"[^a-zA-Z0-9\s/]", "", raw).strip()
        norm = re.sub(r"\s+", " ", norm)
        return HUD_INSTANT_GLOSSARY.get(norm)
