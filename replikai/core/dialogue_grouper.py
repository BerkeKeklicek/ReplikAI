"""OverTranslate Mimarisi: Diyalog Konuşmacı Ayırıcı (`SpeakerLabel`) ve Geometrik Satır/Paragraf Gruplayıcı (`DialogueTextGrouper`).

Hon-Lu/OverTranslate projesindeki `DialogueTextGrouper.cs` ve `SpeakerLineStart.cs` modüllerinin
Python karşılığıdır:

1. `SpeakerLineStart` & `SpeakerLabel`:
   - Altyazı veya diyalog kutularının başındaki konuşmacı isimlerini (`"Michael: Hey!"`, `"[Phone] Lester: ..."`)
     veya üst satırdaki kısa konuşmacı etiketini cümleden ayırır.
   - Konuşmacı adı çeviri motoruna gönderilmez (özel isimlerin yanlış çevrilmesini önler) ve
     çeviri tamamlandığında `"Michael: ..."` formatında geri eklenir.
2. `DialogueTextGrouper`:
   - Oyun içi HUD bildirim kutularında (ör. GTA 5 sol üst 3 satırlık görev/ipucu kutusu veya telefon mesajı)
     alt alta dizilmiş paragraf satırlarını geometrik hiza (`LeftDelta`, `CenterDelta`, `VerticalGap`, `HeightRatio`)
     kurallarıyla TEK BİR PARAGRAF KUTUSUNDA birleştirir.
   - Buna karşılık menü sekmelerini (`MAP`, `BRIEF`, `STATS` gibi `SpacedItems`) veya ayrı menü maddelerini
     birbirine karıştırmadan ayrı tutar!
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

SPEAKER_COLON_RE = re.compile(
    r"^\s*(?:\[[A-Za-z0-9_\-\s]{1,14}\]\s*)?([A-Z][A-Za-z0-9_\-\.]{1,16}(?:\s+[A-Z][A-Za-z0-9_\-\.]{1,14})?)\s*:\s+(.+)$"
)

# Konuşmacı adı olamayacak yaygın İngilizce kelimeler ("Note:", "Warning:", "Objective:" vb. çevrilmelidir)
NON_SPEAKER_LABELS = {
    "note",
    "warning",
    "tip",
    "hint",
    "objective",
    "mission",
    "error",
    "alert",
    "help",
    "info",
    "press",
    "hold",
    "use",
    "go",
    "find",
    "kill",
    "steal",
    "deliver",
    "location",
    "target",
    "reward",
    "bonus",
}

SPEAKER_CLOSING_PUNCT = set(".!?…\"”')~,:;")
SPEAKER_OPENING_PUNCT = set("\"'([")


class DialogueTextGrouper:
    """
    `OverTranslate/Services/Realtime/DialogueTextGrouper.cs` portudur.
    """

    @staticmethod
    def split_speaker_prefix(text: str) -> Tuple[Optional[str], str]:
        """
        Metnin başında konuşmacı etiketi varsa (`"Michael: What are you doing?"`)
        `("Michael", "What are you doing?")` döndürür; yoksa `(None, text)` döndürür.
        """
        raw = (text or "").strip()
        if not raw:
            return None, ""

        m = SPEAKER_COLON_RE.match(raw)
        if not m:
            return None, raw

        speaker = m.group(1).strip()
        body = m.group(2).strip()
        if not speaker or not body or len(body) < 2:
            return None, raw

        if speaker.lower() in NON_SPEAKER_LABELS:
            return None, raw

        # "this: never give up" gibi küçük harfli veya normal kelimeleri konuşmacı sanma
        if not speaker[0].isupper():
            return None, raw

        return speaker, body

    @staticmethod
    def is_speaker_label_line(top_line: str, top_box: Tuple[int, int, int, int], bottom_box: Tuple[int, int, int, int]) -> bool:
        """
        `OverTranslate` `SpeakerLabel(previous, current)` kuralı:
        Üstteki satır, alttaki diyalog satırının yarısından kısaysa (`w_a < w_b * 0.52`),
        sonunda cümle noktalaması yoksa ve sola/hafif dışa yaslıysa bir konuşmacı adıdır.
        """
        txt = (top_line or "").strip()
        if not txt or len(txt) > 18 or " " in txt.strip():
            return False
        if txt[-1] in SPEAKER_CLOSING_PUNCT or txt[0] in SPEAKER_OPENING_PUNCT:
            return False
        if not txt[0].isupper() or txt.lower() in NON_SPEAKER_LABELS:
            return False

        ax, ay, aw, ah = top_box
        bx, by, bw, bh = bottom_box
        height = float(max(ah, bh, 1))
        if aw >= bw * 0.55:
            return False

        # Sol kenar hizası kontrolü
        left_delta = abs(ax - bx)
        return left_delta <= height * 1.4

    @staticmethod
    def same_row(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        """İki kutunun aynı yatay satırda olup olmadığını kontrol eder (`SameRow`)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a_bot, b_bot = ay + ah, by + bh
        overlap_y = min(a_bot, b_bot) - max(ay, by)
        min_h = max(1.0, float(min(ah, bh)))
        max_h = max(1.0, float(max(ah, bh)))
        center_dy = abs((ay + a_bot - by - b_bot) / 2.0)
        return overlap_y >= min_h * 0.55 and center_dy <= max_h * 0.38

    @staticmethod
    def adjacent_in_row(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        """
        Aynı satırda yan yana duran iki kelime kutusunun tek bir cümlenin parçası olup olmadığını
        kontrol eder (`Adjacent`). Menü sekmeleri gibi aralarında geniş boşluk olanları birleştirmez.
        """
        if not DialogueTextGrouper.same_row(a, b):
            return False
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        max_h = max(1.0, float(max(ah, bh)))
        min_h = max(1.0, float(min(ah, bh)))
        if (min_h / max_h) < 0.68:
            return False
        horiz_gap = bx - (ax + aw)
        # Kısa menü sekmeleri (ör. MAP   BRIEF   STATS) birleştirilmemeli (`ShortItem` & `SpacedItems`)
        if aw <= ah * 2.6 and bw <= bh * 2.6 and horiz_gap >= max_h * 0.75:
            return False
        return -max_h * 0.3 <= horiz_gap <= max_h * 1.35

    @staticmethod
    def next_paragraph_row(prev_box: Tuple[int, int, int, int], curr_box: Tuple[int, int, int, int]) -> bool:
        """
        `OverTranslate` `NextRow` kuralı:
        Alt alta duran iki satırın dikey boşluğu (`VerticalGap`), yüksekliği (`HeightRatio`)
        ve sol/merkez hizası (`AlignmentDelta`) aynı paragrafın devamı olduğunu gösteriyor mu?
        - GTA 5 sol üst bilgi kutularındaki 2-4 satırlık paragrafları tek kutuda birleştirir.
        - Dikey menü listelerindeki kısa bağımsız satırları (`RESUME`, `SETTINGS`, `QUIT`) birleştirmez!
        """
        px, py, pw, ph = prev_box
        cx, cy, cw, ch = curr_box

        max_h = max(1.0, float(max(ph, ch)))
        min_h = max(1.0, float(min(ph, ch)))

        # Harf yükseklikleri birbirine yakın olmalı
        if (min_h / max_h) < 0.72:
            return False

        vertical_gap = cy - (py + ph)
        if vertical_gap < -max_h * 0.25 or vertical_gap > max_h * 0.72:
            return False

        # Bir paragraf satırı yeterince geniş olmalıdır (dikey menüdeki tek kelimelik 'MAP'/'AUDIO' maddelerini birleştirme!)
        # En az üst satır geniş bir cümle satırı (`pw >= ph * 4.2`) olmalı
        if pw < ph * 4.2:
            return False

        left_delta = abs(px - cx) / max_h
        center_delta = abs((px + pw / 2.0) - (cx + cw / 2.0)) / max_h

        # Sola yaslı paragraf (GTA 5 bildirim kutusu) veya ortalanmış diyalog
        return left_delta <= 0.65 or center_delta <= 0.55

    @staticmethod
    def is_valid_corner_tip_block(
        text: str,
        rect: Tuple[int, int, int, int],
        num_lines: int = 1,
        is_known_glossary: bool = False,
    ) -> bool:
        """
        Nino (`CornerBlock.swift`) Akıllı Köşe Bilgi Kartı Filtresi:
        Sol-üst köşedeki gerçek ipucu/bildirim kutularını (`"Press E to..."`, `"Discount Store is closed..."`)
        harita yer adları ve 3D gürültüden (`"CA"`, `"NH"`, tek kelimelik sapmalar) ayırır:
        - Tek satırsa yatay şekilli olmalı (`w > h * 1.15`).
        - Bilinen sözlük terimi değilse en az 3 gerçek kelime içermeli VEYA görev işaretiyle (`◆`, `•`, `[`) başlamalı!
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        if is_known_glossary:
            return True

        _x, _y, w, h = rect
        if num_lines == 1 and w <= max(1, h) * 1.15:
            return False

        # Gerçek kelimeleri say (en az 2 harfli kelimeler)
        words = [tok for tok in re.split(r"[^A-Za-z']+", cleaned) if len(tok) >= 2]
        if len(words) >= 3:
            return True

        # Görev işaretiyle ("◆ Escape the mine", "[E] Interact") başlayan kısa satırları da geçir
        first_ch = cleaned[0]
        starts_with_marker = (not first_ch.isalnum()) and (first_ch not in ("'", '"', "-", "."))
        if starts_with_marker and len(words) >= 2:
            return True

        return False
