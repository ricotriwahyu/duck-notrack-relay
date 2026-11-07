<div align="center">

# 🦆 zexrtw — API Relay for Duck.ai & NoTrack.ai

**OpenAI- and Anthropic-compatible API gateway with a headless browser pool, streaming, vision, image generation, and an autonomous agent — built for 9router, Docker, and ARM64 SBCs (Radxa / Raspberry Pi).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Playwright](https://img.shields.io/badge/Playwright-Headless-2ead63.svg)](https://playwright.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multiarch-2496ed.svg)](https://www.docker.com/)
[![ARM64](https://img.shields.io/badge/ARM64-Radxa%20Ready-ff6b6b.svg)](https://radxa.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Ringkasan

**zexrtw** adalah relay API yang menyatukan dua backend AI menjadi satu endpoint lokal:

| Backend | Cara kerja | Model |
|---|---|---|
| **Duck.ai** | Chromium headless via Playwright, worker pool + proxy rotation | GPT-6 Luna, GPT-5.4 mini, Claude Haiku 4.5, Mistral Small 4, gpt-oss 120B, Gemma 4 31B |
| **NoTrack.ai** | HTTP streaming langsung, tanpa browser (0 RAM Chromium) | `zexrtw` (uncensored) |

Semua model dibungkus dalam format **OpenAI** (`/v1/chat/completions`) dan **Anthropic** (`/v1/messages`), sehingga bisa langsung dipakai oleh 9router, Cline, Roo-Code, OpenWebUI, LibreChat, Claude Code, atau klien apa pun yang mendukung kedua protokol tersebut.

> Proyek ini **tidak berafiliasi** dengan DuckDuckGo, Duck.ai, atau NoTrack.ai. Gunakan dengan bijak dan patuhi ketentuan layanan masing-masing penyedia.

---

## ✨ Fitur

- **OpenAI-compatible**
  - `POST /v1/chat/completions` — streaming SSE & non-streaming
  - `GET /v1/models` — katalog model + alias router
  - `POST /v1/responses` — OpenAI Responses API
  - `POST /v1/images/generations` — format DALL·E
  - `GET /v1/images/content/{id}` — URL gambar hasil generasi (cached)
- **Anthropic-compatible**
  - `POST /v1/messages` — untuk Claude Code, Cline, dan klien Anthropic lainnya
- **Duck.ai browser pool**
  - Satu instance Chromium dipakai bersama banyak worker (default `DUCKAI_CONCURRENCY=3`)
  - Rotasi proxy otomatis saat kena ban `418 ERR_BN_LIMIT`
  - Solver VQD proof-of-work dan anti-fingerprint (`duck_challenge.py`)
  - Auto-reset worker yang rusak/ban tanpa mematikan server
- **NoTrack direct client**
  - Streaming cepat tanpa browser, hemat RAM
  - Session UID berputar otomatis
- **Vision & attachment bridge**
  - Gambar/PDF/dokumen di-upload ke sesi browser
  - Deteksi path file lokal di prompt lalu isinya dibaca otomatis
- **Autonomous agent** (`model: "agent"`)
  - ReAct loop yang menjalankan `bash`, `read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search` langsung di host
- **Operasional**
  - Bearer token / `x-api-key` opsional
  - `GET /health` — status uptime, platform, dan worker pool
  - Log ring buffer untuk debugging
  - Context trimming otomatis agar prompt muat di batas Duck.ai

---

## 🤖 Model

| Model ID | Provider | Kemampuan |
|---|---|---|
| `gpt-6-luna` *(default)* | OpenAI | Chat, reasoning, **vision**, **generate image** |
| `gpt-5.4-mini` | OpenAI | Chat cepat & efisien |
| `claude-haiku-4-5` | Anthropic | Analisis kode & teks |
| `mistral-small-2603` | Mistral AI | Multibahasa |
| `tinfoil/gpt-oss-120b` | Tinfoil / OSS | Penalaran mendalam |
| `tinfoil/gemma4-31b` | Google / Tinfoil | Ringan & cepat |
| `zexrtw` | NoTrack.ai | Uncensored, tanpa browser, sangat cepat |
| `agent` | NoTrack.ai + tools | Autonomous agent di host |

**Alias otomatis:** `gpt-6`, `gpt-4o`, `gpt-4o-mini`, `gpt-5.4`, `claude-3-5-haiku`, `claude-3-5-sonnet`, `claude`, `mistral`, `gemma`, `gpt-oss`, `notrack`, `dall-e-3`, `gpt-image-1/2`.

---

## 🔌 Endpoint

| Method | Path | Auth | Keterangan |
|---|---|---|---|
| `GET` | `/` | — | Info service & daftar endpoint |
| `GET` | `/health` | — | Health check, uptime, worker pool |
| `POST` | `/v1/chat/completions` | ✅ | Chat OpenAI (stream / non-stream) |
| `POST` | `/v1/responses` | ✅ | OpenAI Responses API |
| `GET` | `/v1/models` | ✅ | Daftar model |
| `POST` | `/v1/messages` | ✅ | Anthropic Messages API |
| `POST` | `/v1/images/generations` | ✅ | Generate gambar |
| `GET` | `/v1/images/content/{id}` | — | Ambil gambar hasil generasi |

Auth aktif hanya jika `DUCKAI_API_KEY` diisi. Kirim `Authorization: Bearer <key>` atau `x-api-key: <key>`.

---

## 🚀 Menjalankan

### Opsi 1 — Docker (paling praktis)

```bash
cp example.env .env      # opsional: isi API key / proxy
docker compose up -d --build
docker compose logs -f
```

Server: `http://localhost:8080` · Health: `http://localhost:8080/health`

### Opsi 2 — Native systemd di Radxa / Raspberry Pi (paling hemat RAM)

```bash
sudo git clone https://github.com/<USER>/<REPO>.git /opt/duckrelay
cd /opt/duckrelay
sudo bash install-radxa.sh
```

Script memasang Chromium + Python venv, lalu mengaktifkan service `duckrelay`.

```bash
sudo systemctl status duckrelay
sudo systemctl restart duckrelay
sudo journalctl -u duckrelay -f
```

### Opsi 3 — Manual (Windows / Linux / dev)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

`run.py` otomatis membuat `.env` dari `example.env`, memeriksa port, menjalankan uvicorn, dan menulis log ke `server.log`.

Butuh Chromium/Chrome terpasang. Jika path-nya tidak terdeteksi otomatis, set `DUCKAI_CHROME_PATH` (mis. `/usr/bin/chromium`).

---

## ⚙️ Konfigurasi

Semua lewat `.env` (lihat `example.env`).

| Variabel | Default | Keterangan |
|---|---|---|
| `DUCKAI_API_KEY` | *(kosong)* | Bearer token untuk `/v1/*`. Kosong = akses terbuka |
| `DUCKAI_BASE` | `https://duck.ai` | Base URL Duck.ai |
| `PUBLIC_BASE_URL` | `http://localhost:8080` | URL publik relay, dipakai untuk link gambar |
| `DUCKAI_MODEL` | `gpt-6-luna` | Model default |
| `DUCKAI_NEW_CHAT` | `0` | `1` = tab baru tiap request (stateless), `0` = reuse sesi |
| `DUCKAI_PROXIES` | *(kosong)* | Pool proxy dipisah koma, rotasi saat ban |
| `DUCKAI_CONCURRENCY` | `3` | Jumlah worker browser paralel |
| `DUCKAI_CHROME_PATH` | *(auto)* | Path binary Chromium/Chrome |
| `DUCKAI_WARM_MIN` / `DUCKAI_WARM_MAX` | `2.0` / `7.0` | Jeda warmup halaman (detik) |
| `DUCKAI_PREWARM` | `1` | Prewarm tab pertama saat startup |
| `DUCKAI_TOOL_ROUTING` | `0` | Intent → tool synthesis (eksperimental) |
| `AGENT_WORKSPACE` | `~` | Root direktori kerja model `agent` |
| `PORT` | `8080` | Port server |

---

## 🧩 Integrasi Klien

### 9router / OneAPI / LiteLLM
- **Base URL:** `http://<HOST>:8080/v1`
- **API Key:** isi apa saja jika `DUCKAI_API_KEY` kosong (mis. `sk-duck`)
- **Model:** `gpt-6-luna`, `claude-haiku-4-5`, `zexrtw`, `agent`, …

### Cline / Roo-Code
- Provider: **OpenAI Compatible**
- Base URL: `http://<HOST>:8080/v1`
- Model: `gpt-6-luna`

### OpenWebUI / LibreChat
```env
OPENAI_API_BASE_URL=http://<HOST>:8080/v1
OPENAI_API_KEY=sk-duck
```

### Claude Code
```bash
export ANTHROPIC_BASE_URL="http://<HOST>:8080"
export ANTHROPIC_API_KEY="sk-duck"
claude
```

### cURL
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-duck" \
  -d '{
    "model": "gpt-6-luna",
    "messages": [{"role": "user", "content": "Halo! Siapa kamu?"}],
    "stream": true
  }'
```

### Generate gambar
```bash
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-duck" \
  -d '{"model": "gpt-6-luna", "prompt": "kucing astronot di bulan", "n": 1}'
```

---

## 📁 Struktur Proyek

```
.
├── main.py              # FastAPI app: endpoint, auth, bridge, routing
├── duckai.py            # Browser pool Duck.ai (Playwright + Chromium)
├── duck_challenge.py    # Solver VQD proof-of-work & anti-fingerprint
├── notrack.py           # Klien HTTP langsung NoTrack.ai
├── agent.py             # Engine autonomous agent (ReAct + tools)
├── tools.py             # Parser tool-call & repair JSON
├── toolrouter.py        # Intent → tool routing (opt-in)
├── run.py               # Bootstrap runner lokal
├── Dockerfile
├── docker-compose.yml
├── install-radxa.sh     # Installer systemd untuk Radxa/ARM64
├── duckrelay.service    # Unit file systemd
├── requirements.txt
├── example.env          # Template konfigurasi
└── LICENSE              # MIT
```

---

## 🔐 Keamanan

- **Selalu set `DUCKAI_API_KEY`** jika relay bisa diakses dari jaringan publik.
- Model **`agent` mengeksekusi perintah shell di host**. Jangan pernah mengeksposnya ke internet tanpa isolasi (container/VM) dan pembatasan `AGENT_WORKSPACE`.
- Jangan commit `.env`, log, screenshot debug, atau kredensial proxy. Gunakan `example.env` sebagai template.
- Jika memakai proxy, pastikan kredensial proxy hanya ada di `.env`, bukan di source code.

---

## 📄 Lisensi

Dirilis di bawah lisensi **MIT** — lihat [LICENSE](LICENSE).

Proyek ini independen dan tidak berafiliasi dengan DuckDuckGo, Duck.ai, maupun NoTrack.ai. Semua merek dagang milik pemiliknya masing-masing.
