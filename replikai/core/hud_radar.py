"""3 Filtreli Evrensel Oyun HUD, Bildirim, Telefon ve ESC Menü Radarı (~0.9ms CPU, Sıfır FPS-Drop).

MİMARİ (3 Matematiksel Filtre + OverTranslate Entegrasyonu):
1. Tarak Dişi Yatay Harf Ritmi (Sobel-X + Top-Hat Comb Filter):
   3D dünyadaki araba farı, direk, bina ve çit gibi tek parça objeleri eler;
   sadece yan yana dizilmiş 'Harf-Boşluk-Harf' yatay geçiş ritmine sahip gerçek yazıları geçirir.
2. Aniden Doğma ve Mıh Gibi Sabit Kalma (Pop-In Birth & Temporal Anchor):
   Oyuncu sabit durduğunda zaten ekranda olan 3D objeleri ASLA taramaz. Sadece bir önceki
   karede orada yokken aniden sıfırdan beliren (0 -> 1) ve kaymadan aynı pikselde sabit kalan
   (1 -> 1) yeni oyun içi bildirim/ipucu/telefon/menü yazılarını yakalar.
3. Bir Kez Bak ve Unut (Dead-Zone Kilidi):
   Taranan bir koordinat (yazı olsun veya olmasın) pikselleri değişmediği sürece kilitlenir;
   oyuncu aynı yerde saatlerce dursa bile tekrar taranmaz (%0 CPU). Oyun bildirimi kapattığı an
   üstündeki Türkçe yama da otomatik silinir.
4. OverTranslate `DialogueTextGrouper` + `RealtimeNaturalBackground` + `SourceTextColorSampler`:
   - Çok satırlı bildirim/ipucu paragraflarını geometrik hizayla tek kutuda birleştirir (menü maddelerini ayrı tutar).
   - Orijinal İngilizce harfleri düz kutu yerine doğal arka plan doku yaması (`Inpainting`) ile siler
     ve oyunun orijinal harf rengini + konturunu WCAG 2.0 kontrastıyla korur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from replikai.core.dialogue_grouper import DialogueTextGrouper
from replikai.core.natural_patch import RealtimeNaturalBackground, SourceTextColorSampler

# Tek kelimelik veya yaygın oyun menü/HUD başlıkları için 0.0 ms anlık sözlük
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
    "inventory": "ENVANTER",
    "journal": "GÜNLÜK",
    "quests": "GÖREVLER",
    "skills": "YETENEKLER",
    "character": "KARAKTER",
    "player": "OYUNCU",
    "progress": "İLERLEME",
    "story": "HİKAYE",
    "help": "YARDIM",
    "general": "GENEL",
    "mission passed": "GÖREV TAMAMLANDI",
    "mission failed": "GÖREV BAŞARISIZ",
    "wasted": "ÖLDÜN",
    "busted": "YAKALANDIN",
    "wanted": "ARANIYOR",
}


@dataclass
class HUDCandidateBox:
    """Radarın tespit ettiği, OCR'a girecek küçük oyun içi yazı kutusu."""

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
    natural_bg_patch: Optional[np.ndarray] = field(default=None, repr=False)


class UniversalHUDRadar:
    """
    480x270 ölçekte ~0.9ms içinde çalışan, 3 filtreli + OverTranslate doğal doku yamalı
    evrensel oyun içi yazı radarı.
    """

    COARSE_W = 480
    COARSE_H = 270

    def __init__(self) -> None:
        # 1. Filtre için zaman durumları: baseline (t-2), prev (t-1)
        self._baseline_comb: Optional[np.ndarray] = None
        self._prev_comb: Optional[np.ndarray] = None

        # 3. Filtre (Dead-Zone Kilidi): zone_key -> son bilinen comb enerjisi
        self._dead_zones: Dict[str, int] = {}

        # Aktif olarak ekranda gösterilen yerinde (In-Place) Türkçe yamalar
        self.active_patches: Dict[str, HUDTextPatch] = {}

    def reset(self) -> None:
        self._baseline_comb = None
        self._prev_comb = None
        self._dead_zones.clear()
        self.active_patches.clear()

    @classmethod
    def compute_comb_text_mask(cls, full_rgb: np.ndarray, subtitle_region: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        2. FİLTRE — TARAK DİŞİ YATAY HARF RİTMİ (0.4ms):
        Tam ekranı 480x270'e küçültür; hem ince konturlu (Top-Hat) hem de yatay eksende
        yüksek frekanslı 'Harf-Boşluk-Harf' (Sobel-X tarak dişi) geçişine sahip pikselleri bulur.
        Tek parça 3D objeler (araba farı, bina duvarı, direk, gökyüzü) bu filtreden geçemez!
        """
        fh, fw = full_rgb.shape[:2]
        small = cv2.resize(full_rgb, (cls.COARSE_W, cls.COARSE_H), interpolation=cv2.INTER_AREA)
        max_c = np.max(small, axis=2)

        # İnce harf darbelerini yakalayan 7x7 Top-Hat
        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        tophat = cv2.morphologyEx(max_c, cv2.MORPH_TOPHAT, kernel_th)

        # Koyu kontur veya koyu HUD kutusu komşuluğu
        dark_bg = (max_c < 88).astype(np.uint8) * 255
        near_dark = cv2.dilate(dark_bg, np.ones((3, 3), np.uint8), iterations=1)

        # Tarak Dişi (Comb) Yatay Harf Geçiş Yoğunluğu: |I(x+1) - I(x)| > 36
        dx = np.abs(np.diff(max_c.astype(np.int16), axis=1, append=0))
        horiz_edges = (dx > 36).astype(np.uint8)
        # 11x3 yatay pencerede en az 5 harf kenarı geçişi olmalı (yan yana harf dizisi şartı!)
        edge_density = cv2.boxFilter(horiz_edges, ddepth=-1, ksize=(11, 3), normalize=False)

        comb_mask = (
            (tophat > 48)
            & (max_c > 145)
            & (near_dark > 0)
            & (edge_density >= 5)
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
        """Koordinatı 24px ızgaraya yuvarlayarak kararlı bölge anahtarı üretir."""
        return f"{x // 24}_{y // 24}_{w // 32}_{h // 20}"

    @staticmethod
    def sample_box_colors(crop_rgb: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        OverTranslate `SourceTextColorSampler` + `OverlayTextColor` kullanarak
        arka plan rengini (`bg_rgb`) ve WCAG/HSL ayarlı harf rengini (`text_rgb`) döndürür.
        """
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
        1. Aynı satırdaki yan yana kelimeleri (`adjacent_in_row`) birleştirir.
        2. Alt alta dizilmiş paragraf satırlarını (`next_paragraph_row` — ör. GTA 5 sol üst 3 satırlık
           bilgi kutusu) tek bir paragraf kutusunda birleştirir; menü maddelerini ise ayrı tutar!
        Girdi/Çıktı elemanları: `(x, y, w, h, comb_energy)`
        """
        if len(raw_rects) <= 1:
            return raw_rects

        # 1. Yatayda aynı satırdaki yakın kelime parçalarını birleştir
        sorted_h = sorted(raw_rects, key=lambda r: (r[1] // 16, r[0]))
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

        # 2. Dikeyde aynı paragrafın devamı olan satırları (`next_paragraph_row`) birleştir
        sorted_v = sorted(row_merged, key=lambda r: (r[1], r[0]))
        groups: List[List[Tuple[int, int, int, int, int]]] = []
        for rect in sorted_v:
            rx, ry, rw, rh, _ = rect
            placed = False
            for grp in groups:
                # Paragraf en fazla 4 satır yüksekliğinde olabilir (dev menü listelerini yutmaması için)
                if len(grp) >= 4:
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
        """
        3 Filtreli + OverTranslate Gruplamalı tarama yapar:
        1. Ekrandan kaybolan eski HUD/menü bildirimlerini `active_patches` listesinden siler.
        2. Sadece yeni doğup sabitlenen (`Pop-In + Stationary`) ve `Dead-Zone` kilidinde olmayan
           gerçek yazı kutularını döndürür.
        Dönüş: `(yeni_aday_kutular, mevcut_yamalar_degisti_mi)`
        """
        fh, fw = full_rgb.shape[:2]
        cur_comb = self.compute_comb_text_mask(full_rgb, subtitle_region)
        patches_changed = False

        sx = fw / float(self.COARSE_W)
        sy = fh / float(self.COARSE_H)

        # A) Mevcut aktif Türkçe yamaların altındaki oyun yazısı kapandı mı kontrol et
        stale_keys: List[str] = []
        for zkey, patch in list(self.active_patches.items()):
            cx1 = max(0, int(patch.x / sx))
            cy1 = max(0, int(patch.y / sy))
            cx2 = min(self.COARSE_W, max(cx1 + 2, int((patch.x + patch.w) / sx)))
            cy2 = min(self.COARSE_H, max(cy1 + 2, int((patch.y + patch.h) / sy)))
            current_energy = int(np.count_nonzero(cur_comb[cy1:cy2, cx1:cx2]))
            # Alttaki oyun bildirimi/menüsü kapandıysa veya tamamen değiştiyse yamayı kaldır
            if current_energy < max(3, int(patch.comb_energy * 0.38)) or current_energy > int(patch.comb_energy * 2.2):
                stale_keys.append(zkey)
                self._dead_zones.pop(zkey, None)

        for k in stale_keys:
            self.active_patches.pop(k, None)
            patches_changed = True

        # B) Dead-Zone kilitlerinde pikseli boşalan yerlerin kilidini aç
        for dkey, old_energy in list(self._dead_zones.items()):
            parts = dkey.split("_")
            if len(parts) == 4:
                gx, gy, gw, gh = (int(p) for p in parts)
                cx1 = max(0, int((gx * 24) / sx))
                cy1 = max(0, int((gy * 24) / sy))
                cx2 = min(self.COARSE_W, max(cx1 + 2, int(((gx * 24) + (gw * 32)) / sx)))
                cy2 = min(self.COARSE_H, max(cy1 + 2, int(((gy * 24) + (gh * 20)) / sy)))
                cur_e = int(np.count_nonzero(cur_comb[cy1:cy2, cx1:cx2]))
                if abs(cur_e - old_energy) > max(6, int(old_energy * 0.45)):
                    self._dead_zones.pop(dkey, None)

        # İlk karelerde referans maskeleri doldur
        if self._baseline_comb is None or self._prev_comb is None:
            self._baseline_comb = cur_comb.copy()
            self._prev_comb = cur_comb.copy()
            if not force_full_menu:
                return [], patches_changed

        # 1. FİLTRE: Aniden Doğma ve Sabit Kalma (Pop-In Birth + Stationary)
        if force_full_menu:
            # ESC / Menü tuşuna basıldığında ekrandaki tüm sabit tarak-dişi yazılarını tek seferde al
            target_mask = cur_comb
        else:
            prev_dilated = cv2.dilate(self._prev_comb, np.ones((3, 3), np.uint8), iterations=1)
            base_dilated = cv2.dilate(self._baseline_comb, np.ones((5, 5), np.uint8), iterations=1)
            born_and_stationary = (
                (cur_comb > 0)
                & (prev_dilated > 0)
                & (base_dilated == 0)
            ).astype(np.uint8) * 255
            target_mask = born_and_stationary

        # Bir sonraki tur için sabitlenen pikselleri baseline'a aktar (Dead-Zone & 3D sahne hafızası)
        stable_now = ((cur_comb > 0) & (self._prev_comb > 0)).astype(np.uint8) * 255
        self._baseline_comb = cv2.bitwise_or(
            cv2.bitwise_and(self._baseline_comb, cur_comb),
            stable_now,
        )
        self._prev_comb = cur_comb.copy()

        if int(np.count_nonzero(target_mask)) < 8:
            return [], patches_changed

        # Kelime/satır kutularını birleştir ve baş/son harflerin kesilmemesi için yatayda 6px genişlet
        connected = cv2.morphologyEx(
            target_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3)),
        )
        connected = cv2.dilate(
            connected,
            cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2)),
            iterations=1,
        )
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)

        raw_rects: List[Tuple[int, int, int, int, int]] = []
        for i in range(1, num_labels):
            cx = int(stats[i, cv2.CC_STAT_LEFT])
            cy = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Gerçek bir kelime/bildirim kutusu boyutu kontrolü (coarse 480x270 ölçekte)
            if cw < 9 or cw > 320 or ch < 3 or ch > 65:
                continue

            comb_energy = int(np.count_nonzero(cur_comb[cy : cy + ch, cx : cx + cw]))
            if comb_energy < 7:
                continue

            # Gerçek ekran koordinatlarına (1920x1080) çevir (sol/sağ ilk-son harf payı pad_x=18)
            pad_x, pad_y = 18, 6
            rx1 = max(0, int(cx * sx) - pad_x)
            ry1 = max(0, int(cy * sy) - pad_y)
            rx2 = min(fw, int((cx + cw) * sx) + pad_x)
            ry2 = min(fh, int((cy + ch) * sy) + pad_y)
            rw = rx2 - rx1
            rh = ry2 - ry1

            if rw < 36 or rh < 14:
                continue

            raw_rects.append((rx1, ry1, rw, rh, comb_energy))

        # OverTranslate DialogueTextGrouper ile çok satırlı bildirim paragraflarını tek kutuda birleştir
        grouped_rects = self._group_raw_rects(raw_rects, fw, fh)

        candidates: List[HUDCandidateBox] = []
        for rx1, ry1, rw, rh, comb_energy in grouped_rects:
            zkey = self._make_zone_key(rx1, ry1, rw, rh)
            # 3. FİLTRE: Zaten aktif yaması varsa veya Dead-Zone kilidindeyse ASLA tekrar tarama!
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

        # En belirgin kutuları önceliklendir
        candidates.sort(key=lambda c: c.comb_energy, reverse=True)
        return candidates[:max_candidates], patches_changed

    def lock_dead_zone(self, box: HUDCandidateBox) -> None:
        """Bir kez bakılan kutuyu (içinde anlamlı yazı olmasa bile) kilitler; tekrar OCR çalışmaz."""
        self._dead_zones[box.zone_key] = box.comb_energy

    def register_patch(self, box: HUDCandidateBox, source_text: str, translated_text: str) -> HUDTextPatch:
        """
        Çevrilen HUD/menü kutusunu OverTranslate `RealtimeNaturalBackground` doğal doku yamasıyla
        birlikte aktif yerinde (In-Place) yama olarak kaydeder.
        """
        self._dead_zones[box.zone_key] = box.comb_energy
        natural_bg = RealtimeNaturalBackground.build_natural_patch(box.crop_rgb, box.bg_rgb)
        patch = HUDTextPatch(
            zone_key=box.zone_key,
            x=box.x,
            y=box.y,
            w=box.w,
            h=box.h,
            source_text=source_text,
            translated_text=translated_text,
            bg_rgb=box.bg_rgb,
            text_rgb=box.text_rgb,
            outline_rgb=box.outline_rgb,
            comb_energy=box.comb_energy,
            natural_bg_patch=natural_bg,
        )
        self.active_patches[box.zone_key] = patch
        return patch

    @staticmethod
    def lookup_instant_glossary(text: str) -> Optional[str]:
        """Menü ve HUD terimlerini 0.0 ms içinde çevirir."""
        norm = re.sub(r"[^a-zA-Z0-9\s/]", "", (text or "")).strip().lower()
        norm = re.sub(r"\s+", " ", norm)
        return HUD_INSTANT_GLOSSARY.get(norm)
