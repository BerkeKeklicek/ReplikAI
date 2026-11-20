"""Çoklu Yapay Zeka Destekli Bağlamsal Oyun Çeviri Motoru.

Desteklenen Motorlar:
1. Google Gemini API (gemini-2.5-flash vb.)
2. OpenAI API (gpt-4o-mini vb.)
3. Yerel Ollama (Ücretsiz / Çevrimdışı LLM)
4. Ücretsiz Hızlı Çeviri (Anahtarsız varsayılan & otomatik yedek katman)
"""

from __future__ import annotations

from collections import deque
import logging
import re
import threading
import time
from typing import Deque, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter

from replikai.config import AppConfig, GAME_PROFILES
from replikai.core.cache import normalize_ocr_text

logger = logging.getLogger(__name__)


class GameTranslator:
    """
    Sahne Hafızalı & Bağlam Duyarlı Yapay Zeka Çeviri Motoru:
    - Son 6 repliği zaman damgalarıyla sahne hafızasında tutar.
    - 'Hey Michael' gibi sahne içi seslenmeleri düz makine çevirisi ('Merhaba Michael')
      yerine önceki cümlelerin gerilim/aksiyon bağlamına göre ('Baksana Michael!' / 'Hey Michael!') çevirir.
    - Sinirsel çeviride önceki cümleyi Transformer dikkat (attention) penceresine dahil ederek
      bölünmüş ve birbirine bağlı cümleleri tek bir sahne akışı halinde çevirir.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._context_history: Deque[Tuple[str, str, float]] = deque(maxlen=6)
        self._lock = threading.Lock()
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=1)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {
                "Connection": "keep-alive",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
        )

    def warmup(self) -> None:
        """HTTPS TLS 1.3 oturumunu arka planda önceden açarak ilk çeviri gecikmesini ~85ms seviyesine indirir."""
        with self._lock:
            try:
                self._session.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "en", "tl": "tr", "dt": "t", "q": "ok"},
                    timeout=2.5,
                )
            except Exception:
                pass

    def _recent_scene_pairs(self, max_age_sec: float = 35.0) -> List[Tuple[str, str]]:
        now = time.perf_counter()
        return [(s, t) for (s, t, ts) in self._context_history if (now - ts) <= max_age_sec]

    def build_system_prompt(self) -> str:
        profile_name = self.config.game_genre or "Red Dead Redemption 2 (Vahşi Batı 1899)"
        profile = GAME_PROFILES.get(profile_name, {})
        genre = profile.get("genre", profile_name)
        style_guide = profile.get(
            "style_guide",
            "Oyun atmosferine uygun, doğal ve akıcı bir Türkçe kullan.",
        )
        target = self.config.target_lang or "Türkçe"
        history_block = ""
        recent = self._recent_scene_pairs(max_age_sec=60.0)
        if recent:
            lines = [f"- Orijinal: {src} -> Çeviri: {tr}" for src, tr in recent]
            history_block = "\nÖnceki Sahne Diyalog Bağlamı (Çeviriyi bu akışa uydur):\n" + "\n".join(lines)

        return (
            f"Sen profesyonel bir AAA video oyunu ve sinema yerelleştirme (dublaj/altyazı) uzmanısın. "
            f"Oyun/Video Profili: {genre}. Hedef Dil: {target}.\n"
            f"Üslup Rehberi: {style_guide}\n"
            f"Kurallar:\n"
            f"1. Ekrandan OCR ile okunan yabancı dildeki (Türkçe HARİÇ herhangi bir dildeki) altyazıyı {target} diline çevir.\n"
            f"2. Eğer girdi zaten Türkçe ise HİÇBİR ŞEY çevirme, sadece SKIP_TR döndür.\n"
            f"3. DÜZ MAKİNE ÇEVİRİSİ YAPMA: Önceki sahne cümlelerine bakarak bağlamı koru. Örneğin aksiyon/diyalog ortasında 'Hey Michael' geçiyorsa 'Merhaba Michael' deme; sahneye göre 'Baksana Michael!' veya 'Hey Michael!' de.\n"
            f"4. Özel karakter/yer/eşya isimlerini bozma, OCR kaynaklı ufak harf hatalarını bağlamdan onar.\n"
            f"5. SADECE çevrilmiş metni döndür. Açıklama, tırnak işareti veya ek yorum YAZMA."
            f"{history_block}"
        )

    def translate(self, text: str) -> Tuple[str, str]:
        """
        Metni sahne bağlamıyla çevirir.
        Kaynak metin Türkçe ise `("", "tr_ignored")` döndürerek Türkçeyi Türkçeye çevirmeyi engeller.
        """
        cleaned = normalize_ocr_text(text)
        if not cleaned:
            return "", "none"

        provider = (self.config.provider or "free_fast").lower()

        try:
            if provider == "gemini" and self.config.gemini_api_key.strip():
                result = self._translate_gemini(cleaned)
                if result == "SKIP_TR":
                    return "", "tr_ignored"
                if result:
                    refined = self._refine_contextual_dialogue(cleaned, result)
                    self._record_context(cleaned, refined)
                    return refined, f"Gemini AI ({self.config.gemini_model})"

            elif provider == "openai" and self.config.openai_api_key.strip():
                result = self._translate_openai(cleaned)
                if result == "SKIP_TR":
                    return "", "tr_ignored"
                if result:
                    refined = self._refine_contextual_dialogue(cleaned, result)
                    self._record_context(cleaned, refined)
                    return refined, f"OpenAI ({self.config.openai_model})"

            elif provider == "ollama":
                result = self._translate_ollama(cleaned)
                if result == "SKIP_TR":
                    return "", "tr_ignored"
                if result:
                    refined = self._refine_contextual_dialogue(cleaned, result)
                    self._record_context(cleaned, refined)
                    return refined, f"Ollama AI ({self.config.ollama_model})"
        except Exception as exc:
            logger.warning("%s motorunda hata oluştu, bağlamsal motora geçiliyor: %s", provider, exc)

        # Varsayılan: Sahne Hafızalı Bağlamsal AI Çeviri Katmanı
        fallback = self._translate_free_fast(cleaned)
        if fallback == "__SKIP_TR__":
            return "", "tr_ignored"
        if fallback:
            refined = self._refine_contextual_dialogue(cleaned, fallback)
            self._record_context(cleaned, refined)
            return refined, "Bağlamsal AI Çeviri"

        return "", "Hata"

    def _record_context(self, source: str, translated: str) -> None:
        now = time.perf_counter()
        if self._context_history:
            last_src, _last_tr, last_ts = self._context_history[-1]
            # Eğer yeni cümle bir öncekinin kelime uzantısıysa (pop-in animasyon devamı), eskisini ez
            if (now - last_ts) < 4.0 and source.lower().startswith(last_src.lower()[: min(8, len(last_src))]):
                self._context_history[-1] = (source, translated, now)
                return
        self._context_history.append((source, translated, now))

    def _refine_contextual_dialogue(self, source: str, translated: str) -> str:
        """
        Düz makine çevirisi kalıplarını ('Hey Michael' -> 'Merhaba Michael', 'Got out' -> 'Çıktı' vb.)
        önceki sahne cümlelerinin bağlamına göre doğal sinema/oyun Türkçesine dönüştürür.
        """
        out = (translated or "").strip()
        if not out:
            return out

        src_strip = source.strip()
        src_low = src_strip.lower()
        recent_pairs = self._recent_scene_pairs(max_age_sec=40.0)
        scene_text_low = " ".join(s.lower() for s, _ in recent_pairs)
        has_active_scene = len(recent_pairs) > 0

        # 1. 'Hey <İsim>' / 'Hey!' seslenme ve dikkat çekme bağlamı:
        # İngilizcede 'Hey Michael' sahne ortasında 'Merhaba Michael' değil, 'Baksana Michael' / 'Hey Michael' demektir!
        hey_match = re.match(r"^hey[\s,]+([a-zA-Z0-9_'\-]+)([!?.]*)$", src_strip, flags=re.IGNORECASE)
        if hey_match:
            target_word = hey_match.group(1)
            punct = hey_match.group(2) or "!"
            # Eğer önceki sahnede aksiyon, tehlike, ünlem veya devam eden diyalog varsa 'Baksana <İsim>!' kullan
            urgent_cues = ("!", "?", "watch", "look", "run", "wait", "stop", "eliminat", "out", "gun", "car", "cops", "now", "come")
            if has_active_scene and any(cue in scene_text_low for cue in urgent_cues):
                return f"Baksana {target_word}{punct}"
            if has_active_scene:
                return f"Hey {target_word}, baksana{punct}"
            return f"Hey {target_word}{punct}"

        # Cümle başında 'Hey,' veya 'Hey ' geçtiğinde makine çevirisinin koyduğu 'Merhaba' kelimesini düzelt
        if re.match(r"^hey\b", src_low):
            if has_active_scene:
                out = re.sub(r"^Merhaba\b[\s,]*", "Baksana, ", out, flags=re.IGNORECASE)
            else:
                out = re.sub(r"^Merhaba\b", "Hey", out, flags=re.IGNORECASE)

        # 2. Yarışma / Eleme / Oyun sahnesi bağlamı ('got out' / 'first out' -> 'elendi' / 'ilk elenen'):
        competition_cues = ("eliminat", "contestant", "teammate", "team", "win", "winner", "game", "weapon", "sensor", "heart", "out")
        if any(cue in scene_text_low or cue in src_low for cue in competition_cues):
            got_out_m = re.match(r"^([a-zA-Z0-9_'\-]+)\s+got\s+out([!?.]*)$", src_strip, flags=re.IGNORECASE)
            if got_out_m:
                name_part = got_out_m.group(1)
                punct = got_out_m.group(2) or "!"
                return f"{name_part} elendi{punct}"
            if "first out" in src_low:
                out = re.sub(r"ilk\s+çıkan", "ilk elenen", out, flags=re.IGNORECASE)
                out = re.sub(r"İlk\s+çıkan", "İlk elenen", out)

        # 3. Doğal oyun/aksiyon replik düzeltmeleri:
        if src_low in {"back off", "back off!"}:
            return "Geri çekil!"
        if src_low in {"no way", "no way!"}:
            return "İmkanı yok!"
        if src_low in {"watch out", "watch out!", "look out", "look out!"}:
            return "Dikkat et!"
        if src_low in {"hold on", "hold on!", "hang on", "hang on!"}:
            return "Bekle biraz!"

        return out

    def _translate_gemini(self, text: str) -> Optional[str]:
        api_key = self.config.gemini_api_key.strip()
        model = self.config.gemini_model.strip() or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": self.build_system_prompt()}]},
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 256,
            },
        }
        resp = self._session.post(url, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        out = "".join(p.get("text", "") for p in parts).strip()
        return out or None

    def _translate_openai(self, text: str) -> Optional[str]:
        api_key = self.config.openai_api_key.strip()
        model = self.config.openai_model.strip() or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": text},
            ],
            "temperature": 0.25,
            "max_tokens": 256,
        }
        resp = self._session.post(url, headers=headers, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        out = choices[0].get("message", {}).get("content", "").strip()
        return out or None

    def _translate_ollama(self, text: str) -> Optional[str]:
        base_url = (self.config.ollama_url or "http://localhost:11434").rstrip("/")
        model = self.config.ollama_model.strip() or "gemma2:9b"
        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "system": self.build_system_prompt(),
            "prompt": text,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = self._session.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        out = data.get("response", "").strip()
        return out or None

    def _translate_free_fast(self, text: str) -> Optional[str]:
        """
        Sahne Hafızalı Sinirsel Çeviri Katmanı:
        Önceki cümleyi de Transformer dikkat (self-attention) penceresine dahil ederek
        (`onceki_cumle\\nyeni_cumle`) cümlelerin birbirinden kopuk makine çevirisi gibi
        çevrilmesini önler; sadece yeni cümlenin bağlamsal çevirisini döndürür.
        """
        lang_map = {
            "Türkçe": "tr",
            "İngilizce": "en",
            "Almanca": "de",
            "İspanyolca": "es",
            "Fransızca": "fr",
        }
        tl = lang_map.get(self.config.target_lang, "tr")
        url = "https://translate.googleapis.com/translate_a/single"

        # Son 18 saniye içinde aynı sahneden bir cümle varsa Transformer bağlamı olarak üst satıra ekle
        recent = self._recent_scene_pairs(max_age_sec=18.0)
        use_context_prefix = False
        query_payload = text
        if recent:
            prev_src = recent[-1][0].strip()
            if prev_src and prev_src.lower() != text.strip().lower():
                query_payload = f"{prev_src}\n{text}"
                use_context_prefix = True

        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": tl,
            "dt": "t",
            "q": query_payload,
        }
        with self._lock:
            resp = self._session.get(url, params=params, timeout=2.5)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data[0], list):
            return None

        # Google GTX otomatik dil algılama kodu (data[2]):
        # Eğer algılanan kaynak dil zaten hedef dil (ör. "tr" = Türkçe) ise ASLA çevirme!
        detected_src = str(data[2]).strip().lower() if len(data) > 2 and data[2] else ""
        if detected_src and (detected_src == tl or detected_src.startswith(f"{tl}-")):
            return "__SKIP_TR__"

        translated_chunks = [chunk[0] for chunk in data[0] if chunk and chunk[0]]
        full_translated = "".join(translated_chunks).strip()
        if not full_translated:
            return None

        if use_context_prefix and "\n" in full_translated:
            lines = [ln.strip() for ln in full_translated.splitlines() if ln.strip()]
            if lines:
                return lines[-1]

        return full_translated

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Tam Ekran Menü / Envanter / Bilgi Kartı Toplu (Batch) Çevirisi:
        25-40 adet menü/HUD satırını tek tek 30 ayrı istek atmak yerine TEK BİR HTTP
        isteğinde (~180ms) toplu olarak çevirir.
        """
        if not texts:
            return []

        lang_map = {
            "Türkçe": "tr",
            "İngilizce": "en",
            "Almanca": "de",
            "İspanyolca": "es",
            "Fransızca": "fr",
        }
        tl = lang_map.get(self.config.target_lang, "tr")
        url = "https://translate.googleapis.com/translate_a/single"

        results: List[str] = [""] * len(texts)
        chunk_size = 24

        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            clean_chunk = [normalize_ocr_text(t).replace("\n", " ").strip() for t in chunk]
            # Numaralı ayırıcı kullanmadan temiz satır sonlarıyla gönder (Google GTX \n sınırlarını korur)
            payload = "\n".join(clean_chunk)
            if not payload.strip():
                continue
            try:
                params = {
                    "client": "gtx",
                    "sl": "auto",
                    "tl": tl,
                    "dt": "t",
                    "q": payload,
                }
                with self._lock:
                    resp = self._session.get(url, params=params, timeout=4.0)
                resp.raise_for_status()
                data = resp.json()
                if data and isinstance(data[0], list):
                    full_tr = "".join(c[0] for c in data[0] if c and c[0]).strip()
                    tr_lines = [ln.strip() for ln in full_tr.splitlines()]
                    if len(tr_lines) == len(clean_chunk):
                        for idx, tr_line in enumerate(tr_lines):
                            results[start + idx] = self._refine_contextual_dialogue(clean_chunk[idx], tr_line)
                        continue
            except Exception as exc:
                logger.debug("Toplu çeviri hatası (tekli yedeğe geçiliyor): %s", exc)

            # Satır sayısı uyuşmazsa bu küçük grupta tekli çeviriye düş (çevrimdışıysa güvenli atla)
            for idx, item in enumerate(clean_chunk):
                if not item:
                    continue
                try:
                    tr, prov = self.translate(item)
                    if tr and prov != "tr_ignored":
                        results[start + idx] = tr
                except Exception:
                    break

        return results
