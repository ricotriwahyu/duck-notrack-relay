"""Headless Duck.ai chat client (real Chrome + UI-driven, no IP rotation).

WHY THIS DESIGN (root-cause analysis, not guesswork):
  The same machine/IP that the user browses Duck.ai with manually works fine,
  but our Playwright relay got 418 ERR_BN_LIMIT. Two findings from js-reverse:

  1. Playwright's *bundled Chromium* is fingerprinted (TLS/HTTP2/JA3 + UA carries
     "HeadlessChrome") and is hard-banned with ERR_BN_LIMIT on the FIRST request.
     Real Chrome (`channel="chrome"`) passes that layer and instead gets
     ERR_CHALLENGE - i.e. the server is willing to talk, just wants a valid
     client-generated challenge answer.

  2. Calling fetch('/duckchat/v1/chat') by hand (bypassing the app) yields
     ERR_CHALLENGE because x-fe-signals / x-vqd-hash-1 are only produced by the
     app's OWN send logic (real interaction telemetry + challenge JS). Driving
     the real UI - type into the textarea, trigger send - lets the app generate
     the full, valid request. That path returns the real answer ("PONG" in test).

  => The fix is NOT a new IP. It is: real Chrome + drive the app's own send,
     then read the assistant reply from the intercepted chat response.

  We tee the /duckchat/v1/chat RESPONSE (SSE) with a clone().body.getReader() so
  we receive it token-by-token as it streams (the app still reads the original).
  Each SSE line looks like `data: {"action":"success","message":"<token>",...}`
  and ends with `data: [DONE]`. Parsing on the wire is far more robust than
  scraping the DOM, and lets us expose true incremental streaming upstream.

SESSION REUSE (2026-09): the app natively sends structured multi-turn
  `messages[]` and keeps conversation context in the page. When a follow-up
  request's flattened prompt exactly EXTENDS the previous one, we type only the
  new turn into the same live page (no re-warm, no re-sending history). Anything
  else falls back to a fresh page with the full flattened prompt. Either way the
  model sees the whole conversation.

BODY REWRITE (experimental): the hook may patch the app's own outgoing chat
  body (reasoningEffort / toolChoice / canUseTools) before send. If the server
  binds its challenge to the body (ERR_CHALLENGE), we auto-disable rewriting for
  the process lifetime and retry unmodified.

Model catalog (live /duckchat/v1/models snapshot 2026-09-19) + alias map follow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import urllib.request
from typing import AsyncIterator, List, Optional

from playwright.async_api import async_playwright

from duck_challenge import (
    CHALLENGE_INIT_SCRIPT,
    DEFAULT_UA,
    JS_POLL_CHUNKS,
    JS_RESET_CAPTURE,
    REAL_USER_AGENTS,
    get_random_ua,
)

logger = logging.getLogger("duckai")

BASE = os.getenv("DUCKAI_BASE", "https://duck.ai")
REAL_UA = DEFAULT_UA
IMAGE_RESULT_PREFIX = "__ZEXRTW_IMAGE_RESULT__"


def find_chrome() -> Optional[str]:
    """Auto-detect system Chrome or Chromium binary across OSes, prioritizing Radxa / Linux ARM64."""
    env_p = os.getenv("DUCKAI_CHROME_PATH", "").strip()
    if env_p and os.path.exists(env_p):
        return env_p
    candidates = [
        # Linux (Radxa SBC, Debian, Ubuntu, Armbian, Alpine, Arch)
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/snap/bin/chromium",
        "/usr/lib/chromium/chromium",
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for cmd in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        w = shutil.which(cmd)
        if w:
            return w
    return None


CHROME_PATH = find_chrome()
# A real desktop Chrome UA (no "HeadlessChrome" marker). Version pinned to a common
# stable release; must NOT contain "HeadlessChrome".
REAL_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

# Static fallback catalog (snapshot 2026-11). The live list comes from
# fetch_model_catalog(); /v1/models and access checks prefer the live data.
MODEL_LABELS = {
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "mistral-small-2603": "Mistral Small 4",
    "tinfoil/gpt-oss-120b": "gpt-oss 120B",
    "tinfoil/gemma4-31b": "Gemma 4 31B",
    "zexrtw": "zexrtw",
    "agent": "zexrtw Autonomous Agent",
}
DEFAULT_MODEL = "gpt-6-luna"

MODEL_METADATA = {
    "gpt-6-luna": {
        "name": "GPT-6 Luna",
        "short_name": "6 Luna",
        "provider": "OpenAI",
        "badge": "Cocok untuk penggunaan sehari-hari",
        "description": "Model andalan untuk tugas harian, penalaran, dan pembuatan gambar.",
        "icon": "openai",
        "reasoning": ["none", "low"],
        "supports_images": True,
        "access": "free",
    },
    "gpt-5.4-mini": {
        "name": "GPT-5.4 mini",
        "short_name": "5.4 mini",
        "provider": "OpenAI",
        "badge": "Mantap, tapi mencapai batas lebih cepat",
        "description": "Cepat, efisien, dan lincah untuk percakapan umum.",
        "icon": "openai",
        "reasoning": ["none", "low", "medium"],
        "supports_images": True,
        "access": "free",
    },
    "claude-haiku-4-5": {
        "name": "Claude Haiku 4.5",
        "short_name": "Haiku 4.5",
        "provider": "Anthropic",
        "badge": "Mantap, tapi mencapai batas lebih cepat",
        "description": "Sangat cepat untuk coding, ringkasan, dan instruksi presisi.",
        "icon": "anthropic",
        "reasoning": ["none", "low"],
        "supports_images": True,
        "access": "free",
    },
    "mistral-small-2603": {
        "name": "Mistral Small 4",
        "short_name": "Mistral Small",
        "provider": "Mistral AI",
        "badge": "Cepat & ringkas",
        "description": "Model bahasa terbuka yang lincah dan berbobot dari Mistral.",
        "icon": "mistral",
        "reasoning": [],
        "supports_images": False,
        "access": "free",
    },
    "tinfoil/gpt-oss-120b": {
        "name": "gpt-oss 120B",
        "short_name": "gpt-oss",
        "provider": "Tinfoil / OSS",
        "badge": "Open-source berkapasitas besar",
        "description": "Model open-source 120B dengan penalaran mendalam.",
        "icon": "oss",
        "reasoning": ["low"],
        "supports_images": False,
        "access": "free",
    },
    "tinfoil/gemma4-31b": {
        "name": "Gemma 4 31B",
        "short_name": "Gemma 4",
        "provider": "Google / Tinfoil",
        "badge": "Model riset Google",
        "description": "Model cerdas dan efisien dari riset terbuka Google.",
        "icon": "google",
        "reasoning": ["none", "low"],
        "supports_images": True,
        "access": "free",
    },
    "zexrtw": {
        "name": "zexrtw",
        "short_name": "zexrtw",
        "provider": "zexrtw",
        "badge": "Uncensored · Tanpa Sensor",
        "description": "Model AI tanpa sensor moral dan batasan konten, respon kilat via direct HTTP.",
        "icon": "notrack",
        "reasoning": [],
        "supports_images": False,
        "access": "free",
    },
    "agent": {
        "name": "zexrtw Autonomous Agent",
        "short_name": "Agent",
        "provider": "zexrtw",
        "badge": "Autonomous Agent · Code & Shell",
        "description": "AI Agent otonom: bisa baca, tulis, edit file, dan jalankan terminal command secara otomatis.",
        "icon": "notrack",
        "reasoning": [],
        "supports_images": False,
        "access": "free",
    },
}

MODEL_ALIASES = {
    # Autonomous Agent
    "agent": "agent",
    "ze/agent": "agent",
    "zexrtw-agent": "agent",
    "ze-agent": "agent",

    # zexrtw Uncensored
    "zexrtw": "zexrtw",
    "zexrtw-uncensored": "zexrtw",
    "notrack": "zexrtw",
    "notrack-uncensored": "zexrtw",
    "notrack.ai": "zexrtw",
    "notrack-ai": "zexrtw",
    "uncensored": "zexrtw",

    # GPT-6 Luna
    "gpt-6": "gpt-6-luna",
    "gpt-6-luna": "gpt-6-luna",
    "6-luna": "gpt-6-luna",
    "luna": "gpt-6-luna",
    "gpt6": "gpt-6-luna",
    "gpt-5.6-luna": "gpt-6-luna",
    "gpt-5.6": "gpt-6-luna",
    "gpt-5.6-terra": "gpt-6-luna",
    "gpt-5.6-sol": "gpt-6-luna",

    # GPT-5.4 mini
    "gpt-5.4": "gpt-5.4-mini",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "5.4-mini": "gpt-5.4-mini",
    "gpt-5": "gpt-5.4-mini",
    "gpt-5.5": "gpt-5.4-mini",
    "gpt-4o": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-5.4-mini",
    "o3-mini": "gpt-5.4-mini",

    # Claude Haiku 4.5
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "haiku-4.5": "claude-haiku-4-5",
    "haiku": "claude-haiku-4-5",
    "claude-3-5-haiku": "claude-haiku-4-5",
    "claude-3-haiku": "claude-haiku-4-5",
    "claude-sonnet-4-6": "claude-haiku-4-5",
    "claude-sonnet-5-5": "claude-haiku-4-5",
    "claude-sonnet": "claude-haiku-4-5",
    "claude-3-5-sonnet": "claude-haiku-4-5",
    "claude-3.7-sonnet": "claude-haiku-4-5",
    "claude-opus-4-8": "claude-haiku-4-5",
    "claude-opus-5-5": "claude-haiku-4-5",
    "claude-opus": "claude-haiku-4-5",
    "claude": "claude-haiku-4-5",

    # Mistral Small 4
    "mistral-small": "mistral-small-2603",
    "mistral-small-2603": "mistral-small-2603",
    "mistral-small-4": "mistral-small-2603",
    "mistral": "mistral-small-2603",

    # gpt-oss 120B
    "gpt-oss": "tinfoil/gpt-oss-120b",
    "gpt-oss-120b": "tinfoil/gpt-oss-120b",
    "tinfoil/gpt-oss-120b": "tinfoil/gpt-oss-120b",
    "oss-120b": "tinfoil/gpt-oss-120b",

    # Gemma 4 31B
    "gemma": "tinfoil/gemma4-31b",
    "gemma-4": "tinfoil/gemma4-31b",
    "gemma4": "tinfoil/gemma4-31b",
    "gemma4-31b": "tinfoil/gemma4-31b",
    "tinfoil/gemma4-31b": "tinfoil/gemma4-31b",

    # Image generation runs through GPT-6 Luna's native GenerateImage tool
    # (backed by "GPT Image 2"); OpenAI image-model names alias onto it.
    "gpt-image-1": "gpt-6-luna",
    "gpt-image-2": "gpt-6-luna",
    "dall-e-3": "gpt-6-luna",
    "dall-e-2": "gpt-6-luna",
}

MODEL_FAMILY = {
    "gpt-6": "gpt-6-luna",
    "gpt-5.6": "gpt-6-luna",
    "gpt-5.4": "gpt-5.4-mini",
    "claude-haiku": "claude-haiku-4-5",
    "claude-sonnet": "claude-haiku-4-5",
    "claude-opus": "claude-haiku-4-5",
    "claude": "claude-haiku-4-5",
    "mistral": "mistral-small-2603",
    "gpt-oss": "tinfoil/gpt-oss-120b",
    "gemma": "tinfoil/gemma4-31b",
}


def resolve_model(name: str | None) -> str:
    """Map any user-facing model name to a real Duck.ai backend id."""
    if not name:
        return DEFAULT_MODEL
    n = name.strip()
    if n in MODEL_LABELS:
        return n
    low = n.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    # A live catalog id (fetched at startup) is passed through untouched.
    if low in _LIVE_MODELS:
        return _LIVE_MODELS[low]
    for k, v in MODEL_ALIASES.items():
        if k.lower() == low:
            return v
    for fam in sorted(MODEL_FAMILY, key=len, reverse=True):
        if low.startswith(fam):
            return MODEL_FAMILY[fam]
    return n


# id(lower) -> canonical id, refreshed by fetch_model_catalog() via note_live_models().
_LIVE_MODELS: dict = {}


def note_live_models(models: List[dict]) -> None:
    """Remember live catalog ids, modelNames, and shortNames so resolve_model() auto-detects them."""
    global _LIVE_MODELS
    out = {}
    for m in models:
        mid = (m.get("id") or "").strip()
        if not mid:
            continue
        out[mid.lower()] = mid
        for key in ("name", "modelName", "modelShortName"):
            val = (m.get(key) or "").strip()
            if val:
                out[val.lower()] = mid
                out[val.lower().replace(" ", "")] = mid
                out[val.lower().replace(" ", "-")] = mid
    if out:
        _LIVE_MODELS.update(out)


def fetch_model_catalog(timeout: float = 10.0) -> List[dict]:
    """Public GET /duckchat/v1/models (no tokens needed). Raises on network error."""
    url = f"{BASE}/duckchat/v1/models"
    req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": REAL_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    models = data.get("models") or []
    note_live_models(models)
    return models


class DuckAIError(Exception):
    pass


class DuckAIRateLimit(DuckAIError):
    pass


class _Null:
    banned = False


_NULL = _Null()


def _normalize_proxy(proxy: str | None) -> dict:
    """Convert a proxy URL (http/https/socks5, optional user:pass) into Playwright's proxy dict.

    Duck.ai issues a persistent per-IP ban (ERR_BN_LIMIT); a clean proxy is the fix.
    """
    if not proxy:
        return {}
    p = proxy.strip()
    m = re.match(
        r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://(?:(?P<user>[^:@]+):(?P<pwd>[^@]*)@)?(?P<host>.+)$",
        p,
    )
    if not m:
        # bare host:port without scheme -> assume http
        return {"server": f"http://{p}"}
    server = f"{m.group('scheme')}://{m.group('host')}"
    out: dict = {"server": server}
    if m.group("user"):
        out["username"] = m.group("user")
        out["password"] = m.group("pwd") or ""
    return out


# Re-export challenge scripts from isolated duck_challenge module
_INIT_SCRIPT = CHALLENGE_INIT_SCRIPT
_RESET_JS = JS_RESET_CAPTURE
_POLL_JS = JS_POLL_CHUNKS


def _parse_event(line: str) -> Optional[dict]:
    """Parse one SSE payload line into an event dict, or None if not a JSON object.

    Text turns use `{"message": ...}`; image turns use role=ui-component events
    carrying `data.b64Image`. Both flow through here so the driver stays generic.
    """
    line = line.strip()
    if not line.startswith("{"):
        if line.startswith("data:"):
            line = line[5:].strip()
        else:
            return None
    if line == "[DONE]" or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# Flipped off process-wide if a rewritten body ever draws ERR_CHALLENGE
# (i.e. Duck.ai binds its anti-bot answer to the request body).
_REWRITE_OK = True


# Used when the composer textarea exposes no maxlength attribute. 12K chars is
# verified to get an answer from duck.ai; 24K never replies.
_PROMPT_FALLBACK_LIMIT = 12000


# Page warmup. A fixed 7s sleep paid on EVERY turn dominated end-to-end latency
# (measured: ~10s to first token, of which 7s was this sleep). We now poll for
# readiness with WARM_MIN as a hard floor and WARM_MAX as the deadline, which
# keeps the anti-ban guarantee but lets a fast page skip straight to ready.
WARM_MIN = float(os.getenv("DUCKAI_WARM_MIN", "2.0"))
WARM_MAX = float(os.getenv("DUCKAI_WARM_MAX", "7.0"))

# Launch Chrome and warm a page at server startup instead of on first request.
# Set to 0 to boot lazy (no browser until someone actually asks).
PREWARM = os.getenv("DUCKAI_PREWARM", "1") not in ("0", "false", "False")


def _clamp_prompt(prompt: str, limit: int) -> str:
    """Smart truncation: keep system preamble (head) and the latest user message (tail).
    
    Agent clients like OpenCode send 100K+ context. Duck.ai only takes 12K.
    We must preserve the last user question intact or AI gets nothing useful.
    """
    if len(prompt) <= limit:
        return prompt
    marker = "\n…[earlier context truncated for brevity]…\n"
    if limit <= len(marker) + 100:
        return prompt[-limit:]
    # Give 70% to the tail (latest messages) and 30% to the head (system prompt)
    tail_n = int(limit * 0.7) - len(marker)
    head_n = limit - tail_n - len(marker)
    return prompt[:head_n] + marker + prompt[-tail_n:]


class _BrowserSession:
    """One persistent real-Chrome tab; UI-driven send; SSE response tee-streamed."""

    def __init__(self, model: str, proxy: Optional[str], timeout: float, new_chat: bool = False) -> None:
        self.model = model
        self.proxy = proxy
        self.timeout = timeout
        self.new_chat = new_chat
        self._pw = None
        self.browser = None
        self.ctx = None
        self.page = None
        self._last_prompt: Optional[str] = None
        self._current_model: Optional[str] = None
        self._ready = False
        self.banned = False
        # When the current page was opened - drives the warmup poll in _wait_warm.
        self._page_opened_at = 0.0
        # The reused page holds live conversation state, so only one send may drive
        # it at a time; concurrent clients queue instead of clobbering each other.
        self._lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._pw = await async_playwright().start()
        executable = find_chrome()
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-first-run",
            "--mute-audio",
            "--memory-pressure-off",
            "--js-flags=--max-old-space-size=2048",
        ]
        launch: dict = {"headless": True, "args": launch_args}
        if executable:
            launch["executable_path"] = executable
        else:
            launch["channel"] = "chrome"
        if self.proxy:
            launch["proxy"] = _normalize_proxy(self.proxy)
        try:
            self.browser = await self._pw.chromium.launch(**launch)
        except Exception as e:
            logger.warning("Chromium launch with specific channel/path failed (%s); falling back to default", e)
            launch.pop("executable_path", None)
            launch.pop("channel", None)
            self.browser = await self._pw.chromium.launch(**launch)
        self.ctx = await self.browser.new_context(user_agent=REAL_UA)
        await self.ctx.add_init_script(_INIT_SCRIPT)
        self._ready = True

    async def _drop_page(self) -> None:
        if self.page is not None:
            try:
                await self.page.close()
            except Exception:
                pass
            self.page = None
        self._last_prompt = None
        self._current_model = None

    async def _reset(self) -> None:
        """Tear the whole browser down after a transport-level failure so the
        next request rebuilds from scratch (dead browser/ctx would otherwise
        make every later call throw)."""
        await self._drop_page()
        try:
            if self.browser:
                await self.browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self.browser = None
        self.ctx = None
        self._pw = None
        self._ready = False

    async def _open_warmed_page(self):
        """Open a fresh page and warm it so the app mints its challenge/token JS.

        Duck.ai needs the first seconds of a fresh page to build its challenge
        answer; sending too early hits ERR_BN_LIMIT. 7s is the safe floor.

        We poll for readiness instead of sleeping the full 7s: the challenge
        token is ready well before 7s on a warm connection, so a ready-check
        cuts the common case to ~2s while the deadline still guards slow ones.
        """
        await self._ensure_ready()
        page = await self.ctx.new_page()
        await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        self._page_opened_at = asyncio.get_event_loop().time()
        await self._wait_warm(page)
        self.page = page
        self._last_prompt = None
        return page

    async def _wait_warm(self, page, budget: float = 7.0) -> None:
        """Block until the page looks ready to send, capped at `budget` seconds.

        Readiness signal: the composer textarea exists AND the page has been up
        for at least WARM_MIN seconds - Duck.ai refuses an instant send while
        its challenge JS is still booting. Falls back to the full budget if the
        probe never goes green, so behaviour can never get WORSE than before.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + budget
        while loop.time() < deadline:
            if loop.time() - self._page_opened_at >= WARM_MIN:
                try:
                    ta = await page.query_selector("textarea")
                    if ta is not None and await ta.is_enabled():
                        return
                except Exception:
                    return
            await page.wait_for_timeout(200)

    def _continuation(self, prompt: str) -> Optional[str]:
        """Text to type when `prompt` only ADDS one turn to the live conversation.

        Stateless clients (Claude Code et al.) re-send the whole messages[] each
        turn; our flatten() is deterministic, so when the new prompt starts with
        exactly what we typed last time, the delta is one `Human:` turn - the page
        already holds the rest as native context. Anything messier -> None (fresh
        full-prompt send), which is always correct, just slower.
        """
        if self._last_prompt is None or self.page is None:
            return None
        if not prompt.startswith(self._last_prompt):
            return None
        tail = prompt[len(self._last_prompt):].strip()
        marks = list(re.finditer(r"(?m)^Human: ", tail))
        if len(marks) != 1:
            return None
        new_text = tail[marks[0].end():].strip()
        return new_text or None

    async def _trigger_send(self, page, prompt: str, image_path: Optional[str] = None) -> None:
        # Upload image first if provided
        if image_path:
            try:
                fi = await page.query_selector("input[type='file']")
                if fi:
                    await fi.set_input_files(image_path)
                    await page.wait_for_timeout(1500)
                    logger.info("vision: uploaded %s to Duck.ai", image_path)
            except Exception as e:
                logger.warning("vision upload failed: %s", e)

        ta = await page.query_selector("textarea")
        if ta is None:
            raise DuckAIError("duck.ai textarea not found (page layout changed)")
        raw = await ta.get_attribute("maxlength")
        limit = int(raw) if raw and raw.isdigit() and int(raw) > 0 else _PROMPT_FALLBACK_LIMIT
        if len(prompt) > limit:
            logger.warning("prompt %d chars exceeds duck.ai limit %d; truncating",
                           len(prompt), limit)
            prompt = _clamp_prompt(prompt, limit)
        try:
            await ta.wait_for_element_state("enabled", timeout=5000)
            await ta.fill(prompt, timeout=5000)
        except Exception:
            await page.evaluate("""([val]) => {
                const ta = document.querySelector('textarea');
                if (ta) {
                    ta.removeAttribute('disabled');
                    ta.value = val;
                    ta.dispatchEvent(new Event('input', {bubbles: true}));
                    ta.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }""", [prompt])
        await page.wait_for_timeout(300)
        sent = False
        for btn in await page.query_selector_all("button"):
            t = (await btn.inner_text() or "").strip()
            a = await btn.get_attribute("aria-label") or ""
            if t in ("Ask", "Send", "问", "发送") or a in ("Ask", "Send", "问", "发送"):
                await btn.click(force=True)
                sent = True
                break
        if not sent:
            await ta.press("Enter")

    async def _iter_events(self, prompt: str, timeout: float, rewrite: Optional[dict], image_path: Optional[str] = None) -> AsyncIterator[dict]:
        global _REWRITE_OK
        target_model = resolve_model((rewrite or {}).get("model") or self.model)
        page_alive = self.page is not None and not self.page.is_closed()

        # If switching models on an existing page, drop it so a clean warmed tab opens
        if page_alive and self._current_model != target_model:
            await self._drop_page()
            page_alive = False

        text = None
        if page_alive and not self.new_chat:
            text = self._continuation(prompt)
        if text is None:
            if page_alive:
                await self._drop_page()
            page = await self._open_warmed_page()
            self._current_model = target_model
            text = prompt
        else:
            page = self.page

        rw = dict(rewrite or {})
        rw["model"] = target_model
        if target_model == "tinfoil/gpt-oss-120b":
            rw.setdefault("reasoningEffort", "low")

        await page.evaluate(_RESET_JS, rw if (rw and _REWRITE_OK) else None)
        try:
            await self._trigger_send(page, text, image_path=image_path)
        except Exception:
            await self._drop_page()
            raise

        buf = ""
        idx = 0
        saw_data = False
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            st = await page.evaluate(_POLL_JS, idx)
            idx = st["n"]
            if st.get("err"):
                logger.warning("duck.ai raw error string: %s", st["err"])
                if st["err"].startswith("Error:") or st["err"].startswith("TypeError"):
                    # pump() read-failure, not a server payload - surface as generic error
                    await self._drop_page()
                    raise DuckAIError(f"duck.ai stream read failed: {st['err']}")
                if rewrite and st.get("rewriteApplied") and "ERR_CHALLENGE" in st["err"]:
                    _REWRITE_OK = False
                    logger.warning("rewritten chat body drew ERR_CHALLENGE; body rewrite disabled")
                    await self._drop_page()
                    raise _RewriteRejected()
                self._raise_err(st["err"])
            new = st.get("text") or ""
            if new:
                saw_data = True
                buf += new
                # Emit every complete SSE line; keep the trailing partial for next round.
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    ev = _parse_event(line)
                    if ev is not None:
                        yield ev
            if st.get("done"):
                if buf.strip():
                    ev = _parse_event(buf)
                    if ev is not None:
                        yield ev
                if not saw_data:
                    await self._drop_page()
                    raise DuckAIError("Duck.ai closed the stream without a reply")
                self._last_prompt = prompt
                return
            await page.wait_for_timeout(200 if saw_data else 500)
        await self._drop_page()
        raise DuckAIError("timed out waiting for Duck.ai reply")

    async def send_stream_ui(self, prompt: str, timeout: float = 120.0, rewrite: Optional[dict] = None, image_path: Optional[str] = None) -> AsyncIterator[str]:
        """Drive the real UI to send, then stream the reply token-by-token.
        
        Handles both text tokens AND native image generation events (yielded as markdown).
        """
        for attempt in (0, 1):
            try:
                async with self._lock:
                    async for ev in self._iter_events(prompt, timeout, rewrite, image_path=image_path):
                        # 1. Plain text stream
                        msg = ev.get("message")
                        if msg:
                            yield msg

                        # 2. Native image generation tool call
                        if ev.get("role") == "tool-invocation" and ev.get("state") == "call":
                            try:
                                gp = json.loads(ev.get("toolArguments") or "{}").get("imageGenPrompt")
                                if gp:
                                    yield f"🎨 *Generating image*: {gp}\n\n"
                            except Exception:
                                pass

                        # 3. Native image generation success. Keep base64 inside
                        # the relay process: main.py converts this private marker
                        # into a short cached URL before anything reaches clients.
                        data = ev.get("data")
                        if isinstance(data, dict):
                            img = data.get("b64Image")
                            if img and data.get("status") == "success":
                                title = data.get("title") or "Generated Image"
                                payload = json.dumps({"title": title, "b64": img}, separators=(",", ":"))
                                yield IMAGE_RESULT_PREFIX + payload
                return
            except _RewriteRejected:
                if attempt == 0 and rewrite:
                    continue
                raise

    async def send_ui(self, prompt: str, timeout: float = 120.0, rewrite: Optional[dict] = None, image_path: Optional[str] = None) -> str:
        parts: List[str] = []
        async for tok in self.send_stream_ui(prompt, timeout, rewrite, image_path=image_path):
            parts.append(tok)
        return "".join(parts)

    _IMAGE_INSTRUCTION = (
        "请直接调用图像生成工具，根据以下描述生成一张图片，不要反问、不要解释过程。"
        "图像描述：{prompt}{size_hint}"
    )

    async def send_image_ui(
        self,
        prompt: str,
        timeout: float = 180.0,
        size: Optional[str] = None,
        rewrite_model: Optional[str] = None,
    ) -> dict:
        """Delegate to Duck.ai's native GenerateImage (GPT Image 2) and return the image.

        The chat turn makes the model invoke the tool; the SSE carries a
        tool-invocation "call" event (refined imageGenPrompt), then ui-component
        data events whose status=="success" frame holds the FULL base64 JPEG
        (earlier status=="partial" frames are lower-res previews - ignored).
        """
        hint = ""
        if size:
            dims = re.split(r"[x×]", str(size))
            try:
                w, h = int(dims[0]), int(dims[1])
                hint = "（构图：横版宽幅）" if w > h else "（构图：竖版）" if h > w else "（构图：方形）"
            except (ValueError, IndexError):
                pass
        instruction = self._IMAGE_INSTRUCTION.format(prompt=prompt, size_hint=hint)
        rewrite = {"model": rewrite_model} if rewrite_model else None

        result = {"b64": None, "title": None, "gen_prompt": None, "text": ""}
        for attempt in (0, 1):
            try:
                async with self._lock:
                    async for ev in self._iter_events(instruction, timeout, rewrite):
                        if ev.get("message"):
                            result["text"] += ev["message"]
                        if ev.get("role") == "tool-invocation" and ev.get("state") == "call":
                            try:
                                result["gen_prompt"] = json.loads(ev.get("toolArguments") or "{}").get("imageGenPrompt")
                            except json.JSONDecodeError:
                                pass
                        data = ev.get("data")
                        if isinstance(data, dict):
                            if data.get("type") == "image-title":
                                result["title"] = data.get("title") or result["title"]
                            img = data.get("b64Image")
                            if img and data.get("status") == "success":
                                result["b64"] = img
                break
            except _RewriteRejected:
                if attempt == 0 and rewrite:
                    continue
                raise
        if not result["b64"]:
            raise DuckAIError("Duck.ai returned no image (the model may have answered in text)")
        return result

    @staticmethod
    def _raise_err(text: str) -> None:
        m = re.search(r'"type"\s*:\s*"([^"]+)"', text)
        typ = m.group(1) if m else "error"
        if "ERR_BN_LIMIT" in text:
            raise DuckAIRateLimit("Duck.ai ERR_BN_LIMIT (IP/fingerprint banned)")
        if "ERR_CHALLENGE" in text:
            raise DuckAIError("Duck.ai ERR_CHALLENGE (challenge required)")
        raise DuckAIError(f"Duck.ai error: {typ}")

    async def close(self) -> None:
        try:
            if self.browser:
                await self.browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass


class _RewriteRejected(DuckAIError):
    """Internal: rewritten body got ERR_CHALLENGE; retry once unmodified."""


CONCURRENCY_WORKERS = int(os.getenv("DUCKAI_CONCURRENCY", "3"))


class DuckAISession:
    """Multi-Worker Concurrent Browser Pool; supports 3+ concurrent devices simultaneously."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        proxies: Optional[List[str]] = None,
        timeout: float = 120.0,
        new_chat: bool = False,
        concurrency: int = CONCURRENCY_WORKERS,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.new_chat = new_chat
        self.concurrency = max(1, min(concurrency, 6))
        if not proxies:
            self.proxies: List[Optional[str]] = [None]
        elif isinstance(proxies, str):
            self.proxies = [p.strip() or None for p in proxies.split(",")]
        else:
            self.proxies = [p.strip() or None for p in proxies]

        self._workers: List[_BrowserSession] = []
        self._available_workers: asyncio.Queue[_BrowserSession] = asyncio.Queue()
        for i in range(self.concurrency):
            pxy = self.proxies[i % len(self.proxies)]
            worker = _BrowserSession(self.model, pxy, self.timeout, new_chat=self.new_chat)
            self._workers.append(worker)
            self._available_workers.put_nowait(worker)

    async def _lease_worker(self) -> _BrowserSession:
        """Atomically reserve one worker for the complete request/stream."""
        worker = await self._available_workers.get()
        if worker.banned:
            # A reset rebuilds the browser while retaining this worker slot.
            await worker._reset()
            worker.banned = False
        return worker

    def _release_worker(self, worker: _BrowserSession) -> None:
        """Return a leased worker exactly once after its response completes."""
        self._available_workers.put_nowait(worker)

    async def prewarm(self) -> None:
        """Prewarm the primary worker tab ahead of time so the first caller skips startup."""
        if not PREWARM or not self._workers:
            return
        t0 = asyncio.get_event_loop().time()
        try:
            w = self._workers[0]
            await w._ensure_ready()
            await w._open_warmed_page()
            logger.info("prewarmed primary browser tab in %.1fs (multi-worker pool size: %d)",
                        asyncio.get_event_loop().time() - t0, len(self._workers))
        except Exception as e:
            logger.warning("prewarm failed (%s); serving lazily instead", e)

    async def send(self, prompt: str, model: Optional[str] = None, rewrite: Optional[dict] = None, image_path: Optional[str] = None, max_rotations: int = 2) -> str:
        target_model = resolve_model(model or (rewrite or {}).get("model") or self.model)
        rw = dict(rewrite or {})
        rw["model"] = target_model
        if target_model == "tinfoil/gpt-oss-120b":
            rw.setdefault("reasoningEffort", "low")

        worker = await self._lease_worker()
        try:
            return await worker.send_ui(prompt, self.timeout, rw, image_path=image_path)
        except Exception as e:
            await worker._reset()
            raise DuckAIError(f"browser transport failed: {e}")
        finally:
            self._release_worker(worker)

    async def send_stream(self, prompt: str, model: Optional[str] = None, rewrite: Optional[dict] = None, image_path: Optional[str] = None, max_rotations: int = 2) -> AsyncIterator[str]:
        """Stream the assistant reply token-by-token across multi-worker pool."""
        target_model = resolve_model(model or (rewrite or {}).get("model") or self.model)
        rw = dict(rewrite or {})
        rw["model"] = target_model
        if target_model == "tinfoil/gpt-oss-120b":
            rw.setdefault("reasoningEffort", "low")

        worker = await self._lease_worker()
        try:
            async for tok in worker.send_stream_ui(prompt, self.timeout, rw, image_path=image_path):
                yield tok
        except Exception as e:
            await worker._reset()
            raise DuckAIError(f"browser transport failed: {e}")
        finally:
            # Generator finally also runs when the client disconnects, so a
            # cancelled SSE cannot permanently consume a worker slot.
            self._release_worker(worker)

    async def send_image(self, prompt: str, size: Optional[str] = None, max_rotations: int = 2) -> dict:
        worker = await self._lease_worker()
        try:
            return await worker.send_image_ui(
                prompt, timeout=180.0, size=size, rewrite_model="gpt-6-luna"
            )
        except Exception as e:
            await worker._reset()
            raise DuckAIError(f"browser transport failed: {e}")
        finally:
            self._release_worker(worker)

    async def close(self) -> None:
        for w in self._workers:
            await w.close()
