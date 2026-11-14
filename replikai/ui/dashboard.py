"""ReplikAI Pro — Qt6 (PySide6) Tabanlı Profesyonel Oyuncu Komuta Merkezi."""

from __future__ import annotations

from datetime import datetime
import threading
import time
from typing import Dict, Optional

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from replikai.config import AppConfig, GAME_PROFILES
from replikai.core.pipeline import TranslationEvent, TranslationPipeline
from replikai.ui.overlay import SubtitleOverlay
from replikai.ui.region_selector import RegionSelector

PROVIDER_DISPLAY_TO_KEY = {
    "⚡ Ücretsiz Hızlı Motor (Anahtarsız - Anında Kullanım)": "free_fast",
    "✨ Google Gemini AI (gemini-2.5-flash - Bağlamsal)": "gemini",
    "🦙 Yerel Ollama LLM (Ücretsiz & Çevrimdışı AI)": "ollama",
    "🧠 OpenAI GPT (gpt-4o-mini - Bağlamsal)": "openai",
}
PROVIDER_KEY_TO_DISPLAY = {v: k for k, v in PROVIDER_DISPLAY_TO_KEY.items()}

RDR2_SAMPLE_QUOTES = [
    "We're thieves in a world that don't want us no more, Arthur.",
    "I have a plan, Arthur! Have some goddamn faith! One more big score and we disappear.",
    "Watch out partner, Pinkerton lawmen were spotted riding near Valentine.",
    "Take a gamble that love exists, and do a loving act.",
]

PRO_STYLESHEET = """
QMainWindow {
    background-color: #080B11;
}
QWidget {
    color: #E2E8F0;
    font-family: 'Segoe UI', sans-serif;
}
QFrame#Sidebar {
    background-color: #0D121C;
    border-right: 1px solid #1E293B;
}
QFrame#TopBar {
    background-color: #0D121C;
    border-bottom: 1px solid #1E293B;
}
QFrame#Card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
}
QFrame#MetricCard {
    background-color: #111827;
    border: 1px solid #243044;
    border-radius: 12px;
}
QPushButton#NavBtn {
    background-color: transparent;
    color: #94A3B8;
    text-align: left;
    padding: 12px 16px;
    border-radius: 9px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
}
QPushButton#NavBtn:hover {
    background-color: #162032;
    color: #F8FAFC;
}
QPushButton#NavBtn[active="true"] {
    background-color: #17253B;
    color: #00F5D4;
    border: 1px solid rgba(0, 245, 212, 0.35);
}
QPushButton#PrimaryActionBtn {
    background-color: #00D2B4;
    color: #061014;
    font-size: 13px;
    font-weight: 700;
    padding: 12px 18px;
    border-radius: 9px;
    border: none;
}
QPushButton#PrimaryActionBtn:hover {
    background-color: #00F5D4;
}
QPushButton#DangerActionBtn {
    background-color: #E63946;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 700;
    padding: 12px 18px;
    border-radius: 9px;
    border: none;
}
QPushButton#DangerActionBtn:hover {
    background-color: #F43F5E;
}
QPushButton#SecondaryBtn {
    background-color: #1E293B;
    color: #E2E8F0;
    font-size: 12px;
    font-weight: 600;
    padding: 9px 14px;
    border-radius: 8px;
    border: 1px solid #334155;
}
QPushButton#SecondaryBtn:hover {
    background-color: #283548;
    border-color: #475569;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #0B0F19;
    border: 1px solid #263248;
    border-radius: 8px;
    padding: 8px 12px;
    color: #F8FAFC;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #00F5D4;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QTextBrowser#FeedBrowser {
    background-color: #0B0F17;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 10px;
    font-size: 13px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #00F5D4;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #F8FAFC;
    border: 2px solid #00F5D4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
"""


class BridgeSignals(QObject):
    """Arka plan iş parçacıkları ve kısayollar için Qt ana thread sinyal köprüsü."""

    translation_ready = Signal(object)
    status_changed = Signal(str)
    hotkey_toggle_scan = Signal()
    hotkey_select_region = Signal()
    hotkey_toggle_lock = Signal()
    hotkey_single_capture = Signal()


class ReplikAIDashboard(QMainWindow):
    """ReplikAI Pro — Profesyonel Masaüstü Komuta Merkezi."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ReplikAI Pro — Gerçek Zamanlı Yapay Zeka Oyun Çeviri Stüdyosu")
        self.resize(1220, 790)
        self.setMinimumSize(1060, 700)
        self.setStyleSheet(PRO_STYLESHEET)

        self.config_data = AppConfig.load()
        self.signals = BridgeSignals()
        self.signals.translation_ready.connect(self._on_translation_ui)
        self.signals.status_changed.connect(self._on_status_ui)
        self.signals.hotkey_toggle_scan.connect(self.toggle_scanning)
        self.signals.hotkey_select_region.connect(self.open_region_selector)
        self.signals.hotkey_toggle_lock.connect(self.toggle_overlay_lock)
        self.signals.hotkey_single_capture.connect(self.trigger_single_capture)

        self.pipeline = TranslationPipeline(
            config=self.config_data,
            on_translation=lambda ev: self.signals.translation_ready.emit(ev),
            on_status=lambda st: self.signals.status_changed.emit(st),
        )

        self._quote_idx = 0
        self._nav_buttons: list[QPushButton] = []
        self._selector_window: Optional[RegionSelector] = None

        self.overlay = SubtitleOverlay(self.config_data)
        self._build_ui()
        self._register_hotkeys()
        self.refresh_region_preview()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(0, 0, 0, 0)
        main_h.setSpacing(0)

        # SOL NAVİGASYON BAR (Sidebar)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(285)
        side_v = QVBoxLayout(sidebar)
        side_v.setContentsMargins(18, 22, 18, 18)
        side_v.setSpacing(8)

        # Marka Başlığı
        brand_row = QHBoxLayout()
        title_lbl = QLabel("REPLIK AI")
        title_lbl.setFont(QFont("Segoe UI", 20, QFont.Weight.Black))
        title_lbl.setStyleSheet("color: #F8FAFC; letter-spacing: 1px;")
        brand_row.addWidget(title_lbl)

        pro_badge = QLabel("PRO v0.2")
        pro_badge.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        pro_badge.setStyleSheet(
            "background-color: rgba(0, 245, 212, 0.16); color: #00F5D4; "
            "border: 1px solid rgba(0, 245, 212, 0.4); border-radius: 5px; padding: 3px 7px;"
        )
        brand_row.addWidget(pro_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        brand_row.addStretch()
        side_v.addLayout(brand_row)

        sub_lbl = QLabel("Anti-Cheat Güvenli • Canlı OCR & AI Motoru")
        sub_lbl.setStyleSheet("color: #64748B; font-size: 11px; margin-bottom: 10px;")
        side_v.addWidget(sub_lbl)

        # Canlı Durum Göstergesi
        self.status_pill = QLabel("●  DURUM: HAZIR BEKLİYOR")
        self.status_pill.setStyleSheet(
            "background-color: #131C2E; color: #FBBF24; border: 1px solid #263552; "
            "border-radius: 8px; padding: 9px 12px; font-size: 11px; font-weight: 700;"
        )
        side_v.addWidget(self.status_pill)
        side_v.addSpacing(10)

        # Menü Sekme Butonları
        nav_items = [
            ("⚡  Komuta Merkezi & Canlı Akış", 0),
            ("🤠  Oyun Profili & AI Motorları", 1),
            ("🎬  Sinematik Overlay Stüdyosu", 2),
            ("💾  Önbellek & Kısayol Rehberi", 3),
        ]
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setObjectName("NavBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, i=idx: self._switch_page(i))
            side_v.addWidget(btn)
            self._nav_buttons.append(btn)

        side_v.addStretch()

        # Alt Hızlı Aksiyon İstasyonu
        dock_card = QFrame()
        dock_card.setObjectName("Card")
        dock_v = QVBoxLayout(dock_card)
        dock_v.setContentsMargins(12, 12, 12, 12)
        dock_v.setSpacing(8)

        dock_title = QLabel("HIZLI OYUN KONTROLLERİ")
        dock_title.setStyleSheet("color: #64748B; font-size: 10px; font-weight: 800; letter-spacing: 0.8px;")
        dock_v.addWidget(dock_title)

        self.btn_toggle_scan = QPushButton("▶  CANLI ÇEVİRİYİ BAŞLAT (Alt+T)")
        self.btn_toggle_scan.setObjectName("PrimaryActionBtn")
        self.btn_toggle_scan.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_scan.clicked.connect(self.toggle_scanning)
        dock_v.addWidget(self.btn_toggle_scan)

        self.btn_select_region = QPushButton("🎯  Altyazı Bölgesi Seç (Alt+R)")
        self.btn_select_region.setObjectName("SecondaryBtn")
        self.btn_select_region.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_region.clicked.connect(self.open_region_selector)
        dock_v.addWidget(self.btn_select_region)

        self.btn_single_shot = QPushButton("📸  Anlık Tek Kare Çevir (Alt+C)")
        self.btn_single_shot.setObjectName("SecondaryBtn")
        self.btn_single_shot.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_single_shot.clicked.connect(self.trigger_single_capture)
        dock_v.addWidget(self.btn_single_shot)

        self.btn_toggle_lock = QPushButton("🔒  Hayalet Kilit: Aktif (Alt+L)")
        self.btn_toggle_lock.setObjectName("SecondaryBtn")
        self.btn_toggle_lock.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_lock.clicked.connect(self.toggle_overlay_lock)
        dock_v.addWidget(self.btn_toggle_lock)

        side_v.addWidget(dock_card)
        main_h.addWidget(sidebar)

        # SAĞ İÇERİK ALANI
        right_container = QWidget()
        right_v = QVBoxLayout(right_container)
        right_v.setContentsMargins(24, 20, 24, 20)
        right_v.setSpacing(16)

        # Üst Telemetri Kartları (4'lü Grid)
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(14)

        self.metric_profile = self._create_metric_card(
            "AKTİF OYUN PROFİLİ",
            self.config_data.game_genre.split(" (")[0],
            "#E63946",
        )
        self.metric_latency = self._create_metric_card(
            "SON ÇEVİRİ GECİKMESİ",
            "Hazır • 0 ms",
            "#00F5D4",
        )
        self.metric_cache = self._create_metric_card(
            "SQLITE ÖNBELLEK HAVUZU",
            f"{self.pipeline.cache.count()} Kayıtlı Replik",
            "#FBBF24",
        )
        reg = self.config_data.capture_region
        self.metric_region = self._create_metric_card(
            "YAKALAMA KOORDİNATI",
            f"{reg['width']}×{reg['height']} px",
            "#60A5FA",
        )

        metrics_grid.addWidget(self.metric_profile[0], 0, 0)
        metrics_grid.addWidget(self.metric_latency[0], 0, 1)
        metrics_grid.addWidget(self.metric_cache[0], 0, 2)
        metrics_grid.addWidget(self.metric_region[0], 0, 3)
        right_v.addLayout(metrics_grid)

        # Çoklu Sayfa Yığını (StackedWidget)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_command_center_page())
        self.pages.addWidget(self._build_ai_profile_page())
        self.pages.addWidget(self._build_overlay_studio_page())
        self.pages.addWidget(self._build_cache_and_hotkeys_page())

        right_v.addWidget(self.pages, stretch=1)
        main_h.addWidget(right_container, stretch=1)

        self._switch_page(0)

    def _create_metric_card(self, title: str, value: str, accent_hex: str) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("MetricCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(4)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #64748B; font-size: 10px; font-weight: 800; letter-spacing: 0.7px;")
        v.addWidget(lbl_title)

        lbl_val = QLabel(value)
        lbl_val.setFont(QFont("Segoe UI Semibold", 14, QFont.Weight.Bold))
        lbl_val.setStyleSheet(f"color: {accent_hex};")
        v.addWidget(lbl_val)
        return card, lbl_val

    def _build_command_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # Üst Satır: Canlı Ekran Bölgesi Önizleme + RDR2 Simülatörü
        top_split = QHBoxLayout()
        top_split.setSpacing(14)

        # 1. Canlı Bölge Monitörü
        preview_card = QFrame()
        preview_card.setObjectName("Card")
        pv_layout = QVBoxLayout(preview_card)
        pv_layout.setContentsMargins(16, 14, 16, 14)

        pv_header = QHBoxLayout()
        pv_title = QLabel("👁️  SEÇİLİ ALTYAZI BÖLGESİ CANLI ÖNİZLEME")
        pv_title.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 800;")
        pv_header.addWidget(pv_title)

        btn_refresh_pv = QPushButton("🔄 Kareyi Yenile")
        btn_refresh_pv.setObjectName("SecondaryBtn")
        btn_refresh_pv.setFixedHeight(28)
        btn_refresh_pv.clicked.connect(self.refresh_region_preview)
        pv_header.addWidget(btn_refresh_pv, alignment=Qt.AlignmentFlag.AlignRight)
        pv_layout.addLayout(pv_header)

        self.preview_image_lbl = QLabel("Önizleme yükleniyor...")
        self.preview_image_lbl.setFixedHeight(96)
        self.preview_image_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_lbl.setStyleSheet(
            "background-color: #070A0F; border: 1px dashed #263248; border-radius: 8px; color: #64748B;"
        )
        pv_layout.addWidget(self.preview_image_lbl)
        top_split.addWidget(preview_card, stretch=5)

        # 2. RDR2 & Oyun Diyalog Test Stüdyosu
        sim_card = QFrame()
        sim_card.setObjectName("Card")
        sim_layout = QVBoxLayout(sim_card)
        sim_layout.setContentsMargins(16, 14, 16, 14)

        sim_header = QHBoxLayout()
        sim_title = QLabel("🤠  RDR2 & DİYALOG SİMÜLATÖRÜ (ANLIK TEST)")
        sim_title.setStyleSheet("color: #FBBF24; font-size: 11px; font-weight: 800;")
        sim_header.addWidget(sim_title)

        btn_next_quote = QPushButton("🎲 Başka RDR2 Repliği")
        btn_next_quote.setObjectName("SecondaryBtn")
        btn_next_quote.setFixedHeight(28)
        btn_next_quote.clicked.connect(self._cycle_sample_quote)
        sim_header.addWidget(btn_next_quote, alignment=Qt.AlignmentFlag.AlignRight)
        sim_layout.addLayout(sim_header)

        self.test_input = QLineEdit(RDR2_SAMPLE_QUOTES[0])
        sim_layout.addWidget(self.test_input)

        btn_run_sim = QPushButton("⚡  Seçili Motorla Çevir & Sinematik Overlay'de Göster")
        btn_run_sim.setObjectName("PrimaryActionBtn")
        btn_run_sim.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_run_sim.clicked.connect(self.run_test_translation)
        sim_layout.addWidget(btn_run_sim)

        top_split.addWidget(sim_card, stretch=6)
        layout.addLayout(top_split)

        # Alt Satır: Canlı Diyalog Akışı (Zengin Kart Görünümü)
        feed_card = QFrame()
        feed_card.setObjectName("Card")
        feed_v = QVBoxLayout(feed_card)
        feed_v.setContentsMargins(16, 14, 16, 14)

        feed_header = QHBoxLayout()
        feed_title = QLabel("📡  CANLI OYUN DİYALOG & ÇEVİRİ AKIŞI")
        feed_title.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 800;")
        feed_header.addWidget(feed_title)

        btn_clear_feed = QPushButton("🧹 Akışı Temizle")
        btn_clear_feed.setObjectName("SecondaryBtn")
        btn_clear_feed.setFixedHeight(28)
        btn_clear_feed.clicked.connect(lambda: self.feed_browser.clear())
        feed_header.addWidget(btn_clear_feed, alignment=Qt.AlignmentFlag.AlignRight)
        feed_v.addLayout(feed_header)

        self.feed_browser = QTextBrowser()
        self.feed_browser.setObjectName("FeedBrowser")
        self.feed_browser.setOpenExternalLinks(False)
        feed_v.addWidget(self.feed_browser, stretch=1)

        layout.addWidget(feed_card, stretch=1)

        self._append_welcome_banner()
        return page

    def _build_ai_profile_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 10, 12)
        layout.setSpacing(14)

        # Kart 1: Oyun Atmosfer Profili (RDR2 Özel mod dahil)
        prof_card = QFrame()
        prof_card.setObjectName("Card")
        pv = QVBoxLayout(prof_card)
        pv.setContentsMargins(18, 16, 18, 16)
        pv.setSpacing(10)

        pv.addWidget(self._section_label("🎭  OYUN ATMOSFER & TERİM PROFİLİ"))
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(list(GAME_PROFILES.keys()))
        if self.config_data.game_genre in GAME_PROFILES:
            self.genre_combo.setCurrentText(self.config_data.game_genre)
        self.genre_combo.currentTextChanged.connect(self._on_profile_changed)
        pv.addWidget(self.genre_combo)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setFixedHeight(92)
        self.prompt_preview.setStyleSheet("color: #94A3B8; font-size: 12px; background-color: #090D14;")
        pv.addWidget(self.prompt_preview)
        layout.addWidget(prof_card)

        # Kart 2: Yapay Zeka Çeviri Motoru ve API Anahtarları
        ai_card = QFrame()
        ai_card.setObjectName("Card")
        av = QVBoxLayout(ai_card)
        av.setContentsMargins(18, 16, 18, 16)
        av.setSpacing(10)

        av.addWidget(self._section_label("🤖  AKTİF ÇEVİRİ YAPAY ZEKA MOTORU"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_DISPLAY_TO_KEY.keys()))
        current_disp = PROVIDER_KEY_TO_DISPLAY.get(
            self.config_data.provider,
            list(PROVIDER_DISPLAY_TO_KEY.keys())[0],
        )
        self.provider_combo.setCurrentText(current_disp)
        av.addWidget(self.provider_combo)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(QLabel("Google Gemini API Anahtarı:"), 0, 0)
        self.gemini_key_edit = QLineEdit(self.config_data.gemini_api_key)
        self.gemini_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_edit.setPlaceholderText("AIzaSy... (Yerel config.local.json içinde saklanır)")
        grid.addWidget(self.gemini_key_edit, 0, 1)

        grid.addWidget(QLabel("Gemini Modeli:"), 0, 2)
        self.gemini_model_edit = QLineEdit(self.config_data.gemini_model)
        grid.addWidget(self.gemini_model_edit, 0, 3)

        grid.addWidget(QLabel("Yerel Ollama Adresi:"), 1, 0)
        self.ollama_url_edit = QLineEdit(self.config_data.ollama_url)
        grid.addWidget(self.ollama_url_edit, 1, 1)

        grid.addWidget(QLabel("Ollama Modeli:"), 1, 2)
        self.ollama_model_edit = QLineEdit(self.config_data.ollama_model)
        grid.addWidget(self.ollama_model_edit, 1, 3)

        grid.addWidget(QLabel("OpenAI API Anahtarı:"), 2, 0)
        self.openai_key_edit = QLineEdit(self.config_data.openai_api_key)
        self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key_edit.setPlaceholderText("sk-...")
        grid.addWidget(self.openai_key_edit, 2, 1)

        grid.addWidget(QLabel("OpenAI Modeli:"), 2, 2)
        self.openai_model_edit = QLineEdit(self.config_data.openai_model)
        grid.addWidget(self.openai_model_edit, 2, 3)

        av.addLayout(grid)

        btn_save_ai = QPushButton("💾  Yapay Zeka ve Oyun Profilini Kaydet")
        btn_save_ai.setObjectName("PrimaryActionBtn")
        btn_save_ai.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_save_ai.clicked.connect(self.save_settings)
        av.addWidget(btn_save_ai, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(ai_card)
        layout.addStretch()

        self._update_prompt_preview()
        scroll.setWidget(container)
        return scroll

    def _build_overlay_studio_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        card = QFrame()
        card.setObjectName("Card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(20, 18, 20, 18)
        cv.setSpacing(14)

        cv.addWidget(self._section_label("🎨  HAZIR SİNEMATİK ALTYAZI TEMALARI"))
        preset_row = QHBoxLayout()
        presets = [
            ("🤠 RDR2 Kehribar Sarısı", "#FFF3B0", "#0B0E14"),
            ("🎥 Sinema / Netflix Altın", "#FFE66D", "#101218"),
            ("❄️ Ambarino Saf Beyaz", "#FFFFFF", "#090D16"),
            ("⚡ Cyberpunk Neon Camgöbeği", "#00F5D4", "#08121E"),
        ]
        for title, fg, bg in presets:
            b = QPushButton(title)
            b.setObjectName("SecondaryBtn")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, f=fg, g=bg: self._apply_theme_preset(f, g))
            preset_row.addWidget(b)
        cv.addLayout(preset_row)

        # Boyut, Saydamlık ve Tarama Hızı Kaydırıcıları
        grid = QGridLayout()
        grid.setVerticalSpacing(16)
        grid.setHorizontalSpacing(18)

        self.lbl_font_val = QLabel(f"Altyazı Boyutu: {self.config_data.overlay_font_size} px")
        grid.addWidget(self.lbl_font_val, 0, 0)
        self.slider_font = QSlider(Qt.Orientation.Horizontal)
        self.slider_font.setRange(14, 36)
        self.slider_font.setValue(int(self.config_data.overlay_font_size))
        self.slider_font.valueChanged.connect(self._on_overlay_sliders_changed)
        grid.addWidget(self.slider_font, 0, 1)

        self.lbl_opac_val = QLabel(f"Arka Plan Opaklığı: %{int(self.config_data.overlay_opacity * 100)}")
        grid.addWidget(self.lbl_opac_val, 1, 0)
        self.slider_opac = QSlider(Qt.Orientation.Horizontal)
        self.slider_opac.setRange(35, 100)
        self.slider_opac.setValue(int(self.config_data.overlay_opacity * 100))
        self.slider_opac.valueChanged.connect(self._on_overlay_sliders_changed)
        grid.addWidget(self.slider_opac, 1, 1)

        self.lbl_interval_val = QLabel(f"Ekran Tarama Hızı: {self.config_data.scan_interval_ms} ms")
        grid.addWidget(self.lbl_interval_val, 2, 0)
        self.slider_interval = QSlider(Qt.Orientation.Horizontal)
        self.slider_interval.setRange(200, 1200)
        self.slider_interval.setSingleStep(50)
        self.slider_interval.setValue(int(self.config_data.scan_interval_ms))
        self.slider_interval.valueChanged.connect(self._on_overlay_sliders_changed)
        grid.addWidget(self.slider_interval, 2, 1)

        cv.addLayout(grid)

        self.chk_show_orig = QCheckBox(
            "Türkçe altyazının hemen üstünde orijinal İngilizce repliği de küçük italik satır olarak göster"
        )
        self.chk_show_orig.setChecked(bool(self.config_data.show_original_text))
        self.chk_show_orig.stateChanged.connect(self._on_overlay_sliders_changed)
        cv.addWidget(self.chk_show_orig)

        layout.addWidget(card)
        layout.addStretch()
        return page

    def _build_cache_and_hotkeys_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        hk_card = QFrame()
        hk_card.setObjectName("Card")
        hv = QVBoxLayout(hk_card)
        hv.setContentsMargins(18, 16, 18, 16)
        hv.setSpacing(8)
        hv.addWidget(self._section_label("⌨️  OYUN İÇİ GLOBAL KISAYOL TUŞLARI"))
        hv.addWidget(QLabel("•  Alt + R  —  Oyundaki altyazı bölgesini fareyle seçer (HUD Bölge Seçici)"))
        hv.addWidget(QLabel("•  Alt + T  —  Gerçek zamanlı yakalama ve çeviri motorunu Başlatır / Durdurur"))
        hv.addWidget(QLabel("•  Alt + L  —  Hayalet Modu (Click-Through) Kilitler veya Taşıma için Açar"))
        hv.addWidget(QLabel("•  Alt + C  —  Sürekli tarama kapalıyken ekrandaki kareyi tek tuşla çevirir"))
        layout.addWidget(hk_card)

        cache_card = QFrame()
        cache_card.setObjectName("Card")
        cv = QVBoxLayout(cache_card)
        cv.setContentsMargins(18, 16, 18, 16)

        header = QHBoxLayout()
        header.addWidget(self._section_label("💾  SQLITE YEREL ÇEVİRİ ÖNBELLEĞİ (0 ms GECİKME)"))
        btn_clear = QPushButton("🗑️ Tüm Önbelleği Sıfırla")
        btn_clear.setObjectName("DangerActionBtn")
        btn_clear.clicked.connect(self._clear_cache)
        header.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        cv.addLayout(header)

        self.cache_browser = QTextBrowser()
        self.cache_browser.setObjectName("FeedBrowser")
        cv.addWidget(self.cache_browser, stretch=1)

        layout.addWidget(cache_card, stretch=1)
        self._refresh_cache_table()
        return page

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #00F5D4; font-size: 12px; font-weight: 800; letter-spacing: 0.6px;")
        return lbl

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if index == 3:
            self._refresh_cache_table()

    def _append_welcome_banner(self) -> None:
        self.feed_browser.append(
            "<div style='background-color:#111927; border-left: 4px solid #00F5D4; padding:10px; border-radius:6px; margin-bottom:8px;'>"
            "<b style='color:#00F5D4;'>⚡ ReplikAI Pro v0.2 Komuta Merkezi Hazır</b><br>"
            "<span style='color:#94A3B8;'>• <b>Red Dead Redemption 2 (1899 Vahşi Batı)</b> profili aktif. Oyun dosyalarına dokunulmaz, Anti-Cheat %100 güvenlidir.<br>"
            "• Sağ üstteki <b>RDR2 & Diyalog Simülatörü</b> ile Arthur/Dutch repliklerini anında test edebilir veya <b>Alt+R</b> ile oyunda bölge seçebilirsiniz.</span>"
            "</div>"
        )

    def _cycle_sample_quote(self) -> None:
        self._quote_idx = (self._quote_idx + 1) % len(RDR2_SAMPLE_QUOTES)
        self.test_input.setText(RDR2_SAMPLE_QUOTES[self._quote_idx])

    def _on_profile_changed(self, profile_name: str) -> None:
        self.config_data.game_genre = profile_name
        profile = GAME_PROFILES.get(profile_name, {})
        if "subtitle_color" in profile:
            self.config_data.overlay_text_color = profile["subtitle_color"]
            self.overlay.apply_config()
        self.metric_profile[1].setText(profile_name.split(" (")[0])
        self._update_prompt_preview()
        self.config_data.save()

    def _update_prompt_preview(self) -> None:
        self.prompt_preview.setPlainText(self.pipeline.translator.build_system_prompt())

    def _apply_theme_preset(self, fg_hex: str, bg_hex: str) -> None:
        self.config_data.overlay_text_color = fg_hex
        self.config_data.overlay_bg_color = bg_hex
        self.config_data.save()
        self.overlay.apply_config()

    def _on_overlay_sliders_changed(self) -> None:
        self.config_data.overlay_font_size = int(self.slider_font.value())
        self.config_data.overlay_opacity = round(float(self.slider_opac.value()) / 100.0, 2)
        self.config_data.scan_interval_ms = int(self.slider_interval.value())
        self.config_data.show_original_text = bool(self.chk_show_orig.isChecked())

        self.lbl_font_val.setText(f"Altyazı Boyutu: {self.config_data.overlay_font_size} px")
        self.lbl_opac_val.setText(f"Arka Plan Opaklığı: %{int(self.config_data.overlay_opacity * 100)}")
        self.lbl_interval_val.setText(f"Ekran Tarama Hızı: {self.config_data.scan_interval_ms} ms")

        self.config_data.save()
        self.overlay.apply_config()

    def save_settings(self) -> None:
        self.config_data.provider = PROVIDER_DISPLAY_TO_KEY.get(
            self.provider_combo.currentText(), "free_fast"
        )
        self.config_data.game_genre = self.genre_combo.currentText()
        self.config_data.gemini_api_key = self.gemini_key_edit.text().strip()
        self.config_data.gemini_model = self.gemini_model_edit.text().strip() or "gemini-2.5-flash"
        self.config_data.ollama_url = self.ollama_url_edit.text().strip() or "http://localhost:11434"
        self.config_data.ollama_model = self.ollama_model_edit.text().strip() or "gemma2:9b"
        self.config_data.openai_api_key = self.openai_key_edit.text().strip()
        self.config_data.openai_model = self.openai_model_edit.text().strip() or "gpt-4o-mini"
        self.config_data.save()
        self.metric_profile[1].setText(self.config_data.game_genre.split(" (")[0])
        self._update_prompt_preview()

    def refresh_region_preview(self) -> None:
        try:
            img, _ = self.pipeline.capturer.capture_region(
                region=self.config_data.capture_region,
                preprocess=False,
                force=True,
            )
            if img is not None:
                qim = ImageQt(img.convert("RGBA"))
                pix = QPixmap.fromImage(qim).scaled(
                    380,
                    90,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_image_lbl.setPixmap(pix)
        except Exception:
            self.preview_image_lbl.setText("Önizleme alınamadı")

    def toggle_scanning(self) -> None:
        self.save_settings()
        active = self.pipeline.toggle()
        if active:
            self.btn_toggle_scan.setText("⏸  CANLI ÇEVİRİYİ DURDUR (Alt+T)")
            self.btn_toggle_scan.setObjectName("DangerActionBtn")
            self.status_pill.setText("●  DURUM: CANLI TARAMA AKTİF")
            self.status_pill.setStyleSheet(
                "background-color: rgba(0, 245, 212, 0.12); color: #00F5D4; "
                "border: 1px solid rgba(0, 245, 212, 0.45); border-radius: 8px; padding: 9px 12px; font-size: 11px; font-weight: 700;"
            )
        else:
            self.btn_toggle_scan.setText("▶  CANLI ÇEVİRİYİ BAŞLAT (Alt+T)")
            self.btn_toggle_scan.setObjectName("PrimaryActionBtn")
            self.status_pill.setText("●  DURUM: DURAKLATILDI")
            self.status_pill.setStyleSheet(
                "background-color: #131C2E; color: #FBBF24; border: 1px solid #263552; "
                "border-radius: 8px; padding: 9px 12px; font-size: 11px; font-weight: 700;"
            )
        self.btn_toggle_scan.style().unpolish(self.btn_toggle_scan)
        self.btn_toggle_scan.style().polish(self.btn_toggle_scan)

    def open_region_selector(self) -> None:
        was_running = self.pipeline.is_running
        if was_running:
            self.pipeline.stop()

        def _on_selected(region: Dict[str, int]) -> None:
            self.config_data.capture_region = region
            self.config_data.save()
            self.pipeline.capturer.reset_state()
            self.metric_region[1].setText(f"{region['width']}×{region['height']} px")
            self.refresh_region_preview()
            if was_running:
                self.pipeline.start()

        self._selector_window = RegionSelector(on_selected=_on_selected)

    def trigger_single_capture(self) -> None:
        self.save_settings()
        self.refresh_region_preview()
        threading.Thread(
            target=lambda: self.pipeline.process_single_frame(force=True),
            daemon=True,
        ).start()

    def toggle_overlay_lock(self) -> None:
        new_state = not self.config_data.overlay_click_through
        self.overlay.set_click_through(new_state)
        self.config_data.save()
        if new_state:
            self.btn_toggle_lock.setText("🔒  Hayalet Kilit: Aktif (Alt+L)")
        else:
            self.btn_toggle_lock.setText("🔓  Taşıma Modu: Açık (Alt+L)")

    def run_test_translation(self) -> None:
        self.save_settings()
        sample_text = self.test_input.text().strip()
        if not sample_text:
            return

        def _worker() -> None:
            t0 = time.perf_counter()
            cached = self.pipeline.cache.get(
                sample_text,
                target_lang=self.config_data.target_lang,
                genre=self.config_data.game_genre,
            )
            if cached:
                dt = (time.perf_counter() - t0) * 1000.0
                ev = TranslationEvent(
                    source_text=sample_text,
                    translated_text=cached,
                    provider="SQLite Önbellek ⚡",
                    from_cache=True,
                    latency_ms=dt,
                )
                self.signals.translation_ready.emit(ev)
                return

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
                self.signals.translation_ready.emit(ev)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_translation_ui(self, event: TranslationEvent) -> None:
        self.overlay.update_subtitle(
            translated=event.translated_text,
            original=event.source_text,
            provider=event.provider,
        )
        self.metric_latency[1].setText(f"{event.latency_ms:.0f} ms • {event.provider.split(' ')[0]}")
        self.metric_cache[1].setText(f"{self.pipeline.cache.count()} Kayıtlı Replik")

        ts = datetime.now().strftime("%H:%M:%S")
        badge_color = "#00F5D4" if event.from_cache else "#FBBF24"
        card_html = (
            f"<div style='background-color:#111827; border:1px solid #1F2937; border-left:4px solid {badge_color}; "
            f"padding:10px 14px; border-radius:8px; margin-bottom:8px;'>"
            f"<div style='font-size:11px; color:#64748B;'>"
            f"🕒 {ts} &nbsp;|&nbsp; <b style='color:{badge_color};'>{event.provider}</b> &nbsp;|&nbsp; ⚡ {event.latency_ms:.0f} ms"
            f"</div>"
            f"<div style='color:#94A3B8; font-size:12px; margin-top:4px;'><i>🇬🇧 {event.source_text}</i></div>"
            f"<div style='color:#FFF3B0; font-size:14px; font-weight:600; margin-top:3px;'>🇹🇷 {event.translated_text}</div>"
            f"</div>"
        )
        self.feed_browser.append(card_html)

    def _on_status_ui(self, status_text: str) -> None:
        self.status_pill.setText(f"●  DURUM: {status_text.upper()}")

    def _refresh_cache_table(self) -> None:
        self.cache_browser.clear()
        entries = self.pipeline.cache.recent_entries(limit=30)
        if not entries:
            self.cache_browser.setPlainText("Önbellekte henüz kayıtlı replik bulunmuyor.")
            return
        for src, tr, prov in entries:
            self.cache_browser.append(
                f"<div style='padding:6px; border-bottom:1px solid #1E293B;'>"
                f"<span style='color:#00F5D4; font-size:11px;'>[{prov}]</span> "
                f"<span style='color:#94A3B8;'>{src}</span><br>"
                f"<b style='color:#FFF3B0;'>→ {tr}</b></div>"
            )

    def _clear_cache(self) -> None:
        self.pipeline.cache.clear()
        self.metric_cache[1].setText("0 Kayıtlı Replik")
        self._refresh_cache_table()

    def _register_hotkeys(self) -> None:
        try:
            import keyboard

            keyboard.add_hotkey(
                self.config_data.hotkey_toggle_scan,
                lambda: self.signals.hotkey_toggle_scan.emit(),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_select_region,
                lambda: self.signals.hotkey_select_region.emit(),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_toggle_lock,
                lambda: self.signals.hotkey_toggle_lock.emit(),
            )
            keyboard.add_hotkey(
                self.config_data.hotkey_single_capture,
                lambda: self.signals.hotkey_single_capture.emit(),
            )
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self.pipeline.stop()
        try:
            import keyboard

            keyboard.unhook_all()
        except Exception:
            pass
        self.overlay.close()
        event.accept()
