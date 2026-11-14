"""ReplikAI Modern Kontrol Paneli (CustomTkinter)."""

from __future__ import annotations

import threading
from typing import Dict
import customtkinter as ctk

from replikai.config import AppConfig
from replikai.core.pipeline import TranslationEvent, TranslationPipeline
from replikai.ui.overlay import SubtitleOverlay
from replikai.ui.region_selector import RegionSelector

PROVIDER_DISPLAY_TO_KEY = {
    "Ücretsiz Hızlı Çeviri (Anahtarsız)": "free_fast",
    "Google Gemini AI (Bağlamsal)": "gemini",
    "Yerel Ollama (Ücretsiz & Çevrimdışı LLM)": "ollama",
    "OpenAI GPT (Bağlamsal)": "openai",
}
PROVIDER_KEY_TO_DISPLAY = {v: k for k, v in PROVIDER_DISPLAY_TO_KEY.items()}


class ReplikAIDashboard(ctk.CTk):
    """ReplikAI Ana Masaüstü Arayüzü."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("ReplikAI — Yapay Zeka Destekli Gerçek Zamanlı Oyun Çeviri Motoru")
        self.geometry("980x700")
        self.minsize(900, 640)

        self.config_data = AppConfig.load()
        self.pipeline = TranslationPipeline(
            config=self.config_data,
            on_translation=self._on_translation_threadsafe,
            on_status=self._on_status_threadsafe,
        )

        self._build_ui()
        self.overlay = SubtitleOverlay(self, self.config_data)
        self._register_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # SOL KENAR ÇUBUĞU (Kontroller & Durum)
        sidebar = ctk.CTkFrame(self, width=290, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        logo = ctk.CTkLabel(
            sidebar,
            text="🎮 ReplikAI",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color="#00F5D4",
        )
        logo.pack(padx=20, pady=(22, 4), anchor="w")

        sub = ctk.CTkLabel(
            sidebar,
            text="Güvenli & Anlık Oyun Çeviri Motoru",
            font=ctk.CTkFont(size=12),
            text_color="#8E9AAF",
        )
        sub.pack(padx=20, pady=(0, 18), anchor="w")

        # Durum Rozeti
        self.status_badge = ctk.CTkLabel(
            sidebar,
            text="● Durum: Hazır",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFD166",
            fg_color="#1B202B",
            corner_radius=8,
            padx=12,
            pady=8,
        )
        self.status_badge.pack(fill="x", padx=20, pady=(0, 14))

        # Ana Aksiyon Butonları
        self.btn_toggle_scan = ctk.CTkButton(
            sidebar,
            text="▶ Canlı Çeviriyi Başlat (Alt+T)",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            fg_color="#00BB9E",
            hover_color="#009B82",
            text_color="#081014",
            command=self.toggle_scanning,
        )
        self.btn_toggle_scan.pack(fill="x", padx=20, pady=6)

        self.btn_select_region = ctk.CTkButton(
            sidebar,
            text="🎯 Altyazı Bölgesi Seç (Alt+R)",
            height=38,
            fg_color="#263147",
            hover_color="#32415E",
            command=self.open_region_selector,
        )
        self.btn_select_region.pack(fill="x", padx=20, pady=6)

        self.btn_single_shot = ctk.CTkButton(
            sidebar,
            text="⚡ Tek Kare Yakala & Çevir (Alt+C)",
            height=38,
            fg_color="#263147",
            hover_color="#32415E",
            command=self.trigger_single_capture,
        )
        self.btn_single_shot.pack(fill="x", padx=20, pady=6)

        self.btn_toggle_lock = ctk.CTkButton(
            sidebar,
            text="🔒 Hayalet Mod: Aktif (Alt+L)",
            height=38,
            fg_color="#1F2937",
            hover_color="#374151",
            command=self.toggle_overlay_lock,
        )
        self.btn_toggle_lock.pack(fill="x", padx=20, pady=6)

        # Seçili Bölge Bilgisi
        reg = self.config_data.capture_region
        self.region_info_label = ctk.CTkLabel(
            sidebar,
            text=f"📐 Seçili Alan: {reg['width']}x{reg['height']} ({reg['left']}, {reg['top']})",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
        )
        self.region_info_label.pack(padx=20, pady=(14, 4), anchor="w")

        self.cache_info_label = ctk.CTkLabel(
            sidebar,
            text=f"💾 Önbellekteki Replik: {self.pipeline.cache.count()}",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
        )
        self.cache_info_label.pack(padx=20, pady=4, anchor="w")

        # Hızlı Çeviri Test Kutusu
        test_frame = ctk.CTkFrame(sidebar, fg_color="#151922", corner_radius=10)
        test_frame.pack(fill="x", padx=18, pady=(18, 12), side="bottom")

        ctk.CTkLabel(
            test_frame,
            text="🧪 Hızlı Motor Testi",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(padx=12, pady=(10, 4), anchor="w")

        self.test_entry = ctk.CTkEntry(
            test_frame,
            placeholder_text="İngilizce bir oyun repliği yazın...",
        )
        self.test_entry.pack(fill="x", padx=12, pady=4)
        self.test_entry.insert(0, "Stay awhile and listen, stranger! The dark forest is full of traps.")

        ctk.CTkButton(
            test_frame,
            text="Test Çevirisi Yap",
            height=30,
            command=self.run_test_translation,
        ).pack(fill="x", padx=12, pady=(4, 10))

        # SAĞ ANA BÖLÜM (Sekmeli Ayarlar ve Canlı Günlük)
        main_tabs = ctk.CTkTabview(self, corner_radius=12)
        main_tabs.grid(row=0, column=1, padx=18, pady=14, sticky="nsew")

        tab_live = main_tabs.add("📡 Canlı Diyalog Akışı")
        tab_ai = main_tabs.add("🤖 Yapay Zeka & Oyun Bağlamı")
        tab_overlay = main_tabs.add("🎨 Altyazı Katmanı (Overlay)")

        self._build_live_tab(tab_live)
        self._build_ai_tab(tab_ai)
        self._build_overlay_tab(tab_overlay)

    def _build_live_tab(self, parent: ctk.CTkFrame) -> None:
        top_bar = ctk.CTkFrame(parent, fg_color="transparent")
        top_bar.pack(fill="x", padx=8, pady=(4, 8))

        self.latency_label = ctk.CTkLabel(
            top_bar,
            text="⏱️ Son Çeviri Gecikmesi: - ms",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00F5D4",
        )
        self.latency_label.pack(side="left")

        ctk.CTkButton(
            top_bar,
            text="🗑️ Önbelleği Temizle",
            width=140,
            height=28,
            fg_color="#3B1D2E",
            hover_color="#592844",
            command=self._clear_cache,
        ).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_box.insert(
            "end",
            "✨ ReplikAI başlatıldı.\n"
            "• Oyun dosyalarına dokunulmaz, Anti-Cheat korumalı oyunlarda %100 güvenlidir.\n"
            "• [Alt+R] ile oyundaki altyazı bölgesini seçin ve [Alt+T] ile canlı çeviriyi başlatın.\n"
            "• Sol alttaki 'Test Çevirisi Yap' butonuyla çeviri motorunu ve Overlay'i hemen deneyebilirsiniz!\n"
            + ("-" * 72)
            + "\n",
        )

    def _build_ai_tab(self, parent: ctk.CTkFrame) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(
            scroll,
            text="Çeviri Yapay Zeka Motoru:",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        current_display = PROVIDER_KEY_TO_DISPLAY.get(
            self.config_data.provider,
            "Ücretsiz Hızlı Çeviri (Anahtarsız)",
        )
        self.provider_menu = ctk.CTkOptionMenu(
            scroll,
            values=list(PROVIDER_DISPLAY_TO_KEY.keys()),
            width=360,
            command=self._on_provider_change,
        )
        self.provider_menu.set(current_display)
        self.provider_menu.pack(anchor="w", padx=10, pady=(0, 12))

        ctk.CTkLabel(
            scroll,
            text="Oyun Türü ve Atmosfer Bağlamı:",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 4))

        self.genre_menu = ctk.CTkOptionMenu(
            scroll,
            values=[
                "RPG / Hikaye Odaklı",
                "Karanlık Fantezi / Orta Çağ",
                "Bilim Kurgu / Cyberpunk",
                "Aksiyon / Askeri",
                "Korku / Gerilim",
                "Genel Oyun / Menü",
            ],
            width=360,
        )
        self.genre_menu.set(self.config_data.game_genre)
        self.genre_menu.pack(anchor="w", padx=10, pady=(0, 14))

        # Gemini Ayarları
        ctk.CTkLabel(
            scroll,
            text="🔑 Google Gemini API Anahtarı & Model:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.gemini_key_entry = ctk.CTkEntry(
            scroll,
            placeholder_text="AIzaSy...",
            show="*",
            width=420,
        )
        self.gemini_key_entry.pack(anchor="w", padx=10, pady=4)
        self.gemini_key_entry.insert(0, self.config_data.gemini_api_key)

        self.gemini_model_entry = ctk.CTkEntry(scroll, width=240)
        self.gemini_model_entry.pack(anchor="w", padx=10, pady=(2, 12))
        self.gemini_model_entry.insert(0, self.config_data.gemini_model)

        # Yerel Ollama Ayarları
        ctk.CTkLabel(
            scroll,
            text="🦙 Yerel Ollama Sunucu Adresi & Model (Ücretsiz / Çevrimdışı):",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.ollama_url_entry = ctk.CTkEntry(scroll, width=320)
        self.ollama_url_entry.pack(anchor="w", padx=10, pady=4)
        self.ollama_url_entry.insert(0, self.config_data.ollama_url)

        self.ollama_model_entry = ctk.CTkEntry(scroll, width=240)
        self.ollama_model_entry.pack(anchor="w", padx=10, pady=(2, 12))
        self.ollama_model_entry.insert(0, self.config_data.ollama_model)

        # OpenAI Ayarları
        ctk.CTkLabel(
            scroll,
            text="🧠 OpenAI API Anahtarı & Model:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.openai_key_entry = ctk.CTkEntry(
            scroll,
            placeholder_text="sk-...",
            show="*",
            width=420,
        )
        self.openai_key_entry.pack(anchor="w", padx=10, pady=4)
        self.openai_key_entry.insert(0, self.config_data.openai_api_key)

        self.openai_model_entry = ctk.CTkEntry(scroll, width=240)
        self.openai_model_entry.pack(anchor="w", padx=10, pady=(2, 16))
        self.openai_model_entry.insert(0, self.config_data.openai_model)

        ctk.CTkButton(
            scroll,
            text="💾 Yapay Zeka Ayarlarını Kaydet",
            fg_color="#00BB9E",
            hover_color="#009B82",
            text_color="#081014",
            font=ctk.CTkFont(weight="bold"),
            command=self.save_settings,
        ).pack(anchor="w", padx=10, pady=8)

    def _build_overlay_tab(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=16, pady=12)

        ctk.CTkLabel(
            frame,
            text="Altyazı Yazı Boyutu (px):",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(6, 2))

        self.font_slider = ctk.CTkSlider(
            frame,
            from_=14,
            to=36,
            number_of_steps=22,
            width=320,
            command=lambda _v: self._apply_overlay_ui_changes(),
        )
        self.font_slider.set(self.config_data.overlay_font_size)
        self.font_slider.pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Katman Saydamlığı (Opacity):",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(6, 2))

        self.opacity_slider = ctk.CTkSlider(
            frame,
            from_=0.35,
            to=1.0,
            number_of_steps=13,
            width=320,
            command=lambda _v: self._apply_overlay_ui_changes(),
        )
        self.opacity_slider.set(self.config_data.overlay_opacity)
        self.opacity_slider.pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="Altyazı Metin Rengi:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(6, 2))

        self.color_menu = ctk.CTkOptionMenu(
            frame,
            values=["#FFE66D (Sinematik Sarı)", "#FFFFFF (Saf Beyaz)", "#00F5D4 (Siber Camgöbeği)", "#90EE90 (Açık Yeşil)"],
            width=260,
            command=lambda _v: self._apply_overlay_ui_changes(),
        )
        self.color_menu.set("#FFE66D (Sinematik Sarı)")
        self.color_menu.pack(anchor="w", pady=(0, 12))

        self.show_orig_switch = ctk.CTkSwitch(
            frame,
            text="Çevirinin üstünde orijinal yabancı metni de küçük göster",
            command=self._apply_overlay_ui_changes,
        )
        if self.config_data.show_original_text:
            self.show_orig_switch.select()
        self.show_orig_switch.pack(anchor="w", pady=10)

        ctk.CTkLabel(
            frame,
            text="Tarama Aralığı (ms):",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(10, 2))

        self.interval_slider = ctk.CTkSlider(
            frame,
            from_=200,
            to=1500,
            number_of_steps=26,
            width=320,
            command=lambda _v: self._apply_overlay_ui_changes(),
        )
        self.interval_slider.set(self.config_data.scan_interval_ms)
        self.interval_slider.pack(anchor="w", pady=(0, 12))

    def _on_provider_change(self, selected_label: str) -> None:
        self.config_data.provider = PROVIDER_DISPLAY_TO_KEY.get(selected_label, "free_fast")
        self.config_data.save()

    def _apply_overlay_ui_changes(self) -> None:
        self.config_data.overlay_font_size = int(self.font_slider.get())
        self.config_data.overlay_opacity = round(float(self.opacity_slider.get()), 2)
        color_hex = self.color_menu.get().split(" ")[0]
        self.config_data.overlay_text_color = color_hex
        self.config_data.show_original_text = bool(self.show_orig_switch.get())
        self.config_data.scan_interval_ms = int(self.interval_slider.get())
        self.config_data.save()
        self.overlay.apply_config()

    def save_settings(self) -> None:
        self.config_data.provider = PROVIDER_DISPLAY_TO_KEY.get(
            self.provider_menu.get(), "free_fast"
        )
        self.config_data.game_genre = self.genre_menu.get()
        self.config_data.gemini_api_key = self.gemini_key_entry.get().strip()
        self.config_data.gemini_model = self.gemini_model_entry.get().strip() or "gemini-2.5-flash"
        self.config_data.ollama_url = self.ollama_url_entry.get().strip() or "http://localhost:11434"
        self.config_data.ollama_model = self.ollama_model_entry.get().strip() or "gemma2:9b"
        self.config_data.openai_api_key = self.openai_key_entry.get().strip()
        self.config_data.openai_model = self.openai_model_entry.get().strip() or "gpt-4o-mini"
        self.config_data.save()
        self._append_log("💾 Ayarlar başarıyla kaydedildi.\n")

    def toggle_scanning(self) -> None:
        self.save_settings()
        active = self.pipeline.toggle()
        if active:
            self.btn_toggle_scan.configure(
                text="⏸ Canlı Çeviriyi Durdur (Alt+T)",
                fg_color="#EF476F",
                hover_color="#D6335C",
                text_color="#FFFFFF",
            )
            self.status_badge.configure(text="● Durum: Canlı Tarama Aktif", text_color="#00F5D4")
        else:
            self.btn_toggle_scan.configure(
                text="▶ Canlı Çeviriyi Başlat (Alt+T)",
                fg_color="#00BB9E",
                hover_color="#009B82",
                text_color="#081014",
            )
            self.status_badge.configure(text="● Durum: Duraklatıldı", text_color="#FFD166")

    def open_region_selector(self) -> None:
        was_running = self.pipeline.is_running
        if was_running:
            self.pipeline.stop()

        def _selected(region: Dict[str, int]) -> None:
            self.config_data.capture_region = region
            self.config_data.save()
            self.pipeline.capturer.reset_state()
            self.region_info_label.configure(
                text=f"📐 Seçili Alan: {region['width']}x{region['height']} ({region['left']}, {region['top']})"
            )
            self._append_log(
                f"🎯 Yeni altyazı bölgesi seçildi: {region['width']}x{region['height']} @ ({region['left']}, {region['top']})\n"
            )
            if was_running:
                self.pipeline.start()

        RegionSelector(self, on_selected=_selected)

    def trigger_single_capture(self) -> None:
        self.save_settings()
        threading.Thread(
            target=lambda: self.pipeline.process_single_frame(force=True),
            daemon=True,
        ).start()

    def toggle_overlay_lock(self) -> None:
        new_state = not self.config_data.overlay_click_through
        self.overlay.set_click_through(new_state)
        self.config_data.save()
        if new_state:
            self.btn_toggle_lock.configure(text="🔒 Hayalet Mod: Aktif (Alt+L)")
        else:
            self.btn_toggle_lock.configure(text="🔓 Taşıma Modu: Açık (Alt+L)")

    def run_test_translation(self) -> None:
        self.save_settings()
        sample_text = self.test_entry.get().strip()
        if not sample_text:
            return

        def _worker() -> None:
            import time

            t0 = time.perf_counter()
            translated, provider_label = self.pipeline.translator.translate(sample_text)
            dt = (time.perf_counter() - t0) * 1000.0
            if translated:
                self.pipeline.cache.set(
                    source_text=sample_text,
                    translated_text=translated,
                    provider=provider_label,
                    target_lang=self.config_data.target_lang,
                    genre=self.config_data.game_genre,
                )
                ev = TranslationEvent(
                    source_text=sample_text,
                    translated_text=translated,
                    provider=provider_label,
                    from_cache=False,
                    latency_ms=dt,
                )
                self._on_translation_threadsafe(ev)

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_cache(self) -> None:
        self.pipeline.cache.clear()
        self.cache_info_label.configure(text="💾 Önbellekteki Replik: 0")
        self._append_log("🗑️ SQLite çeviri önbelleği temizlendi.\n")

    def _on_translation_threadsafe(self, event: TranslationEvent) -> None:
        self.after(0, lambda: self._handle_translation_ui(event))

    def _on_status_threadsafe(self, status_text: str) -> None:
        self.after(
            0,
            lambda: self.status_badge.configure(text=f"● Durum: {status_text}"),
        )

    def _handle_translation_ui(self, event: TranslationEvent) -> None:
        self.overlay.update_subtitle(event.translated_text, event.source_text)
        self.latency_label.configure(
            text=f"⏱️ Son Gecikme: {event.latency_ms:.0f} ms  |  Motor: {event.provider}"
        )
        self.cache_info_label.configure(
            text=f"💾 Önbellekteki Replik: {self.pipeline.cache.count()}"
        )
        entry = (
            f"[{event.provider} • {event.latency_ms:.0f}ms]\n"
            f"  🇬🇧 {event.source_text}\n"
            f"  🇹🇷 {event.translated_text}\n\n"
        )
        self._append_log(entry)

    def _append_log(self, text: str) -> None:
        self.log_box.insert("end", text)
        self.log_box.see("end")

    def _register_hotkeys(self) -> None:
        try:
            import keyboard

            keyboard.add_hotkey(
                self.config_data.hotkey_toggle_scan,
                lambda: self.after(0, self.toggle_scanning),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_select_region,
                lambda: self.after(0, self.open_region_selector),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_toggle_lock,
                lambda: self.after(0, self.toggle_overlay_lock),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_single_capture,
                lambda: self.after(0, self.trigger_single_capture),
            )
        except Exception:
            pass

    def _on_close(self) -> None:
        self.pipeline.stop()
        try:
            import keyboard

            keyboard.unhook_all()
        except Exception:
            pass
        self.destroy()
