@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ReplikAI Pro - Gercek Zamanli Oyun Ceviri Motoru
echo ReplikAI Pro (Yonetici / Oyun Ustu Katman Modu) baslatiliyor...
python main.py
