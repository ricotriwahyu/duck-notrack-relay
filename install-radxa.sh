#!/usr/bin/env bash
# ==============================================================================
# Installer Duck.ai Web Relay untuk Radxa SBC (ARM64 / aarch64)
# ==============================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${GREEN}     zexrtw API Relay — Installer Radxa & 9router   ${NC}"
echo -e "${BLUE}====================================================${NC}"

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}[Error] Jalankan script ini sebagai root atau dengan sudo:${NC}"
  echo "sudo bash install-radxa.sh"
  exit 1
fi

ARCH=$(uname -m)
echo -e "${BLUE}[*] Mendeteksi Arsitektur Sistem:${NC} ${YELLOW}$ARCH${NC}"

# 1. Update & Install dependencies
echo -e "${BLUE}[1/5] Memperbarui paket & menginstal Chromium dan Python...${NC}"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    chromium \
    fonts-liberation \
    fonts-noto-color-emoji \
    curl \
    procps

# 2. Setup Direktori Instalasi
INSTALL_DIR="/opt/duckrelay"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}[2/5] Menyiapkan direktori di $INSTALL_DIR...${NC}"
mkdir -p "$INSTALL_DIR"
cp -r "$CURRENT_DIR"/* "$INSTALL_DIR"/

cd "$INSTALL_DIR"

# 3. Setup Virtualenv
echo -e "${BLUE}[3/5] Membuat virtualenv Python dan memasang dependensi...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 4. Siapkan File .env jika belum ada
echo -e "${BLUE}[4/5] Menyiapkan konfigurasi .env...${NC}"
if [ ! -f ".env" ]; then
    cp example.env .env
    echo -e "${GREEN}[+] File .env dibuat dari example.env (Open access).${NC}"
fi

# 5. Konfigurasi Systemd Service
echo -e "${BLUE}[5/5] Memasang systemd service 'duckrelay.service'...${NC}"
cp duckrelay.service /etc/systemd/system/duckrelay.service
systemctl daemon-reload
systemctl enable duckrelay.service
systemctl restart duckrelay.service

# Ambil IP lokal
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "IP_RADXA")

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}🎉 Instalasi Berhasil Selesai!${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "Headless API Relay siap dihubungkan ke 9router:"
echo -e "  👉 Base URL (9router): ${YELLOW}http://${LOCAL_IP}:8080/v1${NC}"
echo -e "  👉 Health Check:       ${YELLOW}http://${LOCAL_IP}:8080/health${NC}"
echo -e "  👉 Models Catalog:     ${YELLOW}http://${LOCAL_IP}:8080/v1/models${NC}"
echo -e ""
echo -e "Model yang tersedia:"
echo -e "  - gpt-6-luna (Default), gpt-5.4-mini, claude-haiku-4-5"
echo -e "  - mistral-small-2603, tinfoil/gpt-oss-120b, tinfoil/gemma4-31b"
echo -e "  - zexrtw, zexrtw-uncensored (Direct HTTP, 0 RAM, Uncensored)"
echo -e ""
echo -e "Perintah manajemen:"
echo -e "  Status service:  ${BLUE}sudo systemctl status duckrelay${NC}"
echo -e "  Restart service: ${BLUE}sudo systemctl restart duckrelay${NC}"
echo -e "  Log real-time:   ${BLUE}sudo journalctl -u duckrelay -f${NC}"
echo -e "${GREEN}====================================================${NC}"
