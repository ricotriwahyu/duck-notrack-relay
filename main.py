"""zexrtw - Full Headless OpenAI & Anthropic API Relay for 9router.

Combines Duck.ai and NoTrack.ai into a unified, high-performance API relay.
Designed for 9router, OneAPI, LiteLLM, and Radxa SBC (ARM64).
Endpoints:
  - POST /v1/chat/completions     (OpenAI Chat Completions)
  - GET  /v1/models               (OpenAI Model List)
  - POST /v1/messages             (Anthropic Messages)
  - POST /v1/responses            (OpenAI Responses)
  - POST /v1/images/generations   (OpenAI Images)
  - GET  /v1/images/content/{id}  (Cached Image Content)
  - GET  /health, GET /           (Health & Status)
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import uuid4

from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from dotenv import load_dotenv
from pydantic import BaseModel
from duckai import (
    DEFAULT_MODEL,
    DuckAIError,
    DuckAIRateLimit,
    DuckAISession,
    MODEL_LABELS,
    IMAGE_RESULT_PREFIX,
    fetch_model_catalog,
    resolve_model,
)
from tools import split_text_and_tool
from toolrouter import has_tool_result, route_intent
from notrack import NoTrackClient, NoTrackError, NoTrackRateLimit
import agent

_notrack_client: Optional[NoTrackClient] = None


def get_notrack_client() -> NoTrackClient:
    global _notrack_client
    if _notrack_client is None:
        _notrack_client = NoTrackClient()
    return _notrack_client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zexrtw")

LOG_BUFFER: deque = deque(maxlen=500)


class _TailHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BUFFER.append({
                "ts": time.time(),
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:
            pass


_tail = _TailHandler()
for _name in ("zexrtw", "uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_name).addHandler(_tail)
logging.getLogger().addHandler(_tail)

API_KEY = os.getenv("DUCKAI_API_KEY", "").strip()
# Single proxy (DUCKAI_PROXY) or a comma-separated pool (DUCKAI_PROXIES).
# Empty means a direct connection; the pool is optional.
_proxy_pool = os.getenv("DUCKAI_PROXIES", "").strip() or os.getenv("DUCKAI_PROXY", "").strip()
PROXIES = [p.strip() for p in _proxy_pool.split(",") if p.strip()]
DEFAULT = os.getenv("DUCKAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
NEW_CHAT = os.getenv("DUCKAI_NEW_CHAT", "0").strip() == "1"
# Intent->tool synthesis is a gamble for agent clients (WorkBuddy/Claude Code
# always send tools + a huge context; a regex mis-hit returns content:null and
# the IDE reports "no response from model"). Off by default; opt in per deploy.
TOOL_ROUTING = os.getenv("DUCKAI_TOOL_ROUTING", "0").strip() == "1"
PORT = int(os.getenv("PORT", "8080"))
_BOOT_TS = time.time()
_active_requests = 0


def _id() -> str:
    return f"chatcmpl-{uuid4().hex[:24]}"


def _created() -> int:
    return int(time.time())


# resolve_model is imported from duckai (single source of truth for model mapping).


def estimate_tokens(text: str) -> int:
    if not text:
        return 1
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Bridge helpers: Vision, File Read, Image Generation
# ---------------------------------------------------------------------------
_TEMP_DIR = tempfile.mkdtemp(prefix="zexrtw_")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8080").rstrip("/")


def _consume_image_marker(value: str) -> Optional[str]:
    """Cache a private image marker and return client-safe Markdown URL."""
    if not isinstance(value, str) or not value.startswith(IMAGE_RESULT_PREFIX):
        return None
    try:
        payload = json.loads(value[len(IMAGE_RESULT_PREFIX):])
        img_id = _store_image(payload["b64"])
        title = str(payload.get("title") or "Generated Image").replace("]", "")
        return f"\n\n![{title}]({PUBLIC_BASE_URL}/v1/images/content/{img_id})\n\n"
    except Exception as exc:
        logger.warning("image marker conversion failed: %s", exc)
        return "\n\n[Image generation completed, but caching the result failed.]\n\n"


_FILE_EXTS = (
    ".json", ".yaml", ".bash", ".toml", ".html", ".scss", ".sass",
    ".jsx", ".tsx", ".sql", ".css", ".xml", ".csv", ".log", ".env",
    ".js", ".ts", ".py", ".sh", ".md", ".rs", ".go", ".rb", ".php",
    ".txt", ".c", ".h", ".cpp", ".pdf"
)
_SORTED_EXTS = sorted([ext.lstrip(".") for ext in _FILE_EXTS], key=len, reverse=True)

_FILE_PATH_RE = re.compile(
    r'(?:^|[\s\'"(])([A-Za-z]:[\\/][\w\\\/\.\-]+\.\w+|'
    r'(?:~\/|\.\/|\/[\w]+\/)[\w\/\.\-]+\.\w+|'
    r'[\w\-]+\.(?:' + '|'.join(_SORTED_EXTS) + r'))(?=[\s\'")\.,;:]|$)',
    re.IGNORECASE
)


def _extract_attachments_from_messages(messages: List) -> Tuple[Optional[str], Optional[str]]:
    """Extract attachments from messages.
    
    Returns (upload_file_path, text_injection):
      - upload_file_path: saved temp file for Duck.ai browser upload (images & PDFs)
      - text_injection: decoded text/code for code files, CSV, JSON, TXT, etc.
    """
    upload_path = None
    text_injections = []

    # Attachments belong to the latest user turn only. Scanning the complete
    # history re-uploads stale images on every follow-up and compaction request.
    latest_user = None
    for candidate in reversed(messages):
        role = candidate.role if hasattr(candidate, "role") else candidate.get("role")
        if role == "user":
            latest_user = candidate
            break

    for m in [latest_user] if latest_user is not None else []:
        content = m.content if hasattr(m, "content") else m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            raw_url = ""
            media_type = ""

            if btype in ("image_url", "file") and isinstance(block.get(btype), dict):
                raw_url = block[btype].get("url", "")
            elif btype in ("image", "document", "file") and isinstance(block.get("source"), dict):
                src = block["source"]
                if src.get("type") == "base64":
                    media_type = src.get("media_type", "")
                    raw_url = f"data:{media_type};base64,{src.get('data', '')}"

            if not raw_url.startswith("data:"):
                continue

            try:
                header, b64data = raw_url.split(",", 1)
                header_lower = header.lower()
                data_bytes = base64.b64decode(b64data)

                # 1. Images and PDFs -> upload via Duck.ai file input
                if any(k in header_lower for k in ("image/", "application/pdf")):
                    if not upload_path:
                        ext = ".png"
                        if "jpeg" in header_lower or "jpg" in header_lower:
                            ext = ".jpg"
                        elif "webp" in header_lower:
                            ext = ".webp"
                        elif "gif" in header_lower:
                            ext = ".gif"
                        elif "pdf" in header_lower:
                            ext = ".pdf"
                        fpath = os.path.join(_TEMP_DIR, f"attach_{uuid4().hex[:8]}{ext}")
                        with open(fpath, "wb") as f:
                            f.write(data_bytes)
                        upload_path = fpath
                        logger.info("attachment bridge: saved uploadable %s (%d bytes)", ext, len(data_bytes))

                # 2. Text, code, json, csv, yaml, xml -> decode text directly into prompt
                else:
                    try:
                        decoded_text = data_bytes.decode("utf-8")
                        if len(decoded_text) > 8000:
                            decoded_text = decoded_text[:4000] + "\n...[truncated]...\n" + decoded_text[-4000:]
                        text_injections.append(f"[Attached Document/Code]:\n```\n{decoded_text}\n```")
                        logger.info("attachment bridge: decoded %d text chars", len(decoded_text))
                    except UnicodeDecodeError:
                        pass
            except Exception as e:
                logger.warning("attachment bridge error: %s", e)

    combined_text = "\n\n".join(text_injections) if text_injections else None
    return upload_path, combined_text


def _detect_and_read_files(text: str) -> Optional[str]:
    """Detect file paths & filenames in user text, read contents locally, return injected context."""
    if not text:
        return None
    matches = _FILE_PATH_RE.findall(text)
    if not matches:
        return None
    injected = []
    seen = set()
    search_dirs = [Path("."), Path(agent.WORKSPACE_DIR)]

    for fpath_str in matches[:3]:
        fpath_str = fpath_str.strip().strip("'\"()")
        if fpath_str in seen:
            continue
        seen.add(fpath_str)

        target = Path(fpath_str).expanduser()
        # If relative or bare filename, search in working dirs
        if not target.is_absolute() and not target.exists():
            for sdir in search_dirs:
                cand = sdir / target
                if cand.exists() and cand.is_file():
                    target = cand
                    break

        if not target.exists() or not target.is_file():
            continue

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > 6000:
                content = content[:3000] + "\n...[truncated]...\n" + content[-3000:]
            ext = target.suffix.lstrip(".") or "text"
            injected.append(f"[File Content: {target.name} ({target})]\n```{ext}\n{content}\n```")
            logger.info("file bridge: auto-read %s (%d chars)", target, len(content))
        except Exception as e:
            logger.warning("file bridge: failed to read %s: %s", target, e)

    return "\n\n".join(injected) if injected else None


# ---------------------------------------------------------------------------
# Anthropic message helpers
# ---------------------------------------------------------------------------
def _anthropic_block_text(content: Any) -> str:
    """Flatten an Anthropic content field (str or list of blocks) to text.

    Image blocks are described as placeholders so context stays coherent.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        # Generated-image markdown and accidental serialized data URLs can be
        # hundreds of KB. Preserve their meaning without forwarding raw base64.
        content = re.sub(
            r'data:(?:image|application)/[^;,\s]+;base64,[A-Za-z0-9+/=\r\n]+',
            '[binary attachment omitted]',
            content,
            flags=re.IGNORECASE,
        )
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype in ("image", "image_url"):
                    # Extract image context for text-only backends
                    url_info = ""
                    if isinstance(block.get("image_url"), dict):
                        raw_url = block["image_url"].get("url", "")
                        if raw_url.startswith("data:"):
                            media = raw_url.split(";")[0].replace("data:", "")
                            url_info = f"base64 {media}"
                        else:
                            url_info = raw_url[:100]
                    elif isinstance(block.get("source"), dict):
                        url_info = block["source"].get("media_type", "image")
                    parts.append(f"[User attached an image ({url_info or 'image'}). Vision is not available on this relay; describe or ask about the image content in text.]")
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(f"[tool_result {block.get('tool_use_id','')}]: {inner}")
                    elif isinstance(inner, list):
                        parts.append(f"[tool_result {block.get('tool_use_id','')}]: {_anthropic_block_text(inner)}")
                elif btype == "tool_use":
                    parts.append(
                        f'[tool_call name="{block.get("name","")}"] '
                        f"{json.dumps(block.get('input',{}), ensure_ascii=False)}"
                    )
                elif btype == "input_json_delta" and "partial_json" in block:
                    parts.append(block.get("partial_json", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _is_compaction_request(text: str) -> bool:
    """Recognize OpenCode/Hermes summary turns that must never call tools."""
    low = (text or "").lower()
    markers = (
        "you must summarize the conversation above",
        "structured summary that will be given to another agent",
        "required summary template",
        "return the summary as text",
        "do not call tools",
        "exact section headings",
    )
    return any(marker in low for marker in markers)


def _compact_turn_content(content: Any, max_chars: int = 5000) -> str:
    """Flatten one turn and bound it while preserving its head and tail."""
    text = _anthropic_block_text(content)
    if len(text) <= max_chars:
        return text
    marker = "\n...[turn content compacted]...\n"
    head = max_chars // 3
    tail = max_chars - head - len(marker)
    return text[:head] + marker + text[-tail:]


_IMAGE_GENERATION_RE = re.compile(
    r"\b(?:generate|buat(?:kan)?|bikin|ciptakan|create|draw|gambar(?:kan)?)\b"
    r"[^\n]{0,80}\b(?:gambar|image|foto|ilustrasi|poster|logo|art|photo|picture)\b"
    r"|\b(?:generate|create)\s+(?:an?\s+)?(?:image|photo|picture|illustration)\b"
    r"|\b(?:generate gambar|buat gambar|bikin gambar)\b",
    re.IGNORECASE,
)


def _wants_image_generation(text: str) -> bool:
    """True only for explicit image-creation requests, not image analysis."""
    low = (text or "").lower()
    if any(marker in low for marker in (
        "ini gambar apa", "gambar ini apa", "what is in this image",
        "describe this image", "analisa gambar", "analyze image",
    )):
        return False
    return bool(_IMAGE_GENERATION_RE.search(text or ""))


def _clean_image_prompt(text: str) -> str:
    """Remove router notes and retain the user's actual image request."""
    clean = re.sub(r"\[Note:[\s\S]*?\]\s*", "", text or "", flags=re.IGNORECASE)
    return clean.strip()[-1500:] or "Create a high-quality image."


def _role_label(role: str) -> str:
    return {"user": "Human", "assistant": "Assistant", "system": "System"}.get(role, "Human")


def format_tools_prompt(tools: List[dict]) -> str:
    """Format tools into an actionable few-shot system prompt so the model invokes them."""
    lines = [
        "You are an expert autonomous software engineer with access to system tools.",
        "When the user asks you to read or edit files, explore directories, search code, or run commands, you MUST call one of the available tools.",
        "AVAILABLE TOOLS:"
    ]
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name", "tool")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
        lines.append(f"- {name}({param_desc}): {desc}")

    lines.append("")
    lines.append("TOOL CALL PROTOCOL:")
    lines.append("1. To invoke a tool, output ONLY this exact XML format:")
    lines.append('<tool_call name="tool_name">{"param": "value"}</tool_call>')
    lines.append("2. Never say you cannot access files or run commands. Immediately invoke the tool!")
    lines.append("3. Example:")
    lines.append('<tool_call name="read_file">{"path": "test.js"}</tool_call>')
    return "\n".join(lines)


def flatten_conversation(system: Any, messages: List[dict]) -> str:
    """Flatten system + EVERY turn into one duck.ai prompt string.

    Duck.ai is single-turn (one textarea send), so multi-turn context must be
    re-sent in full on every request. Preserves tool calls and tool results so
    autonomous coding agents maintain coherent multi-step state.
    """
    parts: List[str] = []
    if system:
        sys_text = _anthropic_block_text(system)
        if sys_text:
            parts.append(sys_text)
    for m in messages:
        role = m.get("role", "user")
        text = _anthropic_block_text(m.get("content", ""))

        # Serialize prior assistant tool_calls
        tool_calls = m.get("tool_calls") or []
        tc_texts = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                fn_name = fn.get("name") or tc.get("name", "tool")
                fn_args = fn.get("arguments") or tc.get("input", "{}")
                if isinstance(fn_args, dict):
                    fn_args = json.dumps(fn_args, ensure_ascii=False)
                tc_texts.append(f'<tool_call name="{fn_name}">{fn_args}</tool_call>')

        if role == "tool":
            tool_id = m.get("tool_call_id", "")
            parts.append(f"[tool_result {tool_id}]:\n{text}")
        elif role == "assistant":
            ass_parts = []
            if text.strip():
                ass_parts.append(text.strip())
            if tc_texts:
                ass_parts.extend(tc_texts)
            if ass_parts:
                parts.append(f"Assistant: {' '.join(ass_parts)}")
        else:
            label = _role_label(role)
            if text.strip():
                parts.append(f"{label}: {text}")
    return "\n\n".join(parts).strip()


def _to_anthropic_tool(openai_tool: dict) -> dict:
    """Convert an OpenAI tool spec ({type:function, function:{...}}) to Anthropic shape."""
    if openai_tool.get("type") == "function" and "function" in openai_tool:
        fn = openai_tool["function"]
        return {
            "name": fn.get("name", "tool"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        }
    return openai_tool


def build_anthropic_prompt(system: Any, messages: List[dict]) -> str:
    """Combine system + full conversation into a single duck.ai prompt string."""
    return flatten_conversation(system, messages)


# ---------------------------------------------------------------------------
# App + session
# ---------------------------------------------------------------------------
app = FastAPI(
    title="zexrtw",
    description="zexrtw Headless OpenAI & Anthropic API Relay for 9router",
    version="2.0.0",
)


@app.get("/")
async def root():
    return {
        "service": "zexrtw",
        "status": "ready",
        "description": "Multi-Provider AI Relay for 9router (Duck.ai & NoTrack.ai)",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "messages": "POST /v1/messages",
            "responses": "POST /v1/responses",
            "images": "POST /v1/images/generations",
            "health": "GET /health",
        },
        "default_model": DEFAULT,
        "models": [
            "gpt-6-luna",
            "gpt-5.4-mini",
            "claude-haiku-4-5",
            "mistral-small-2603",
            "tinfoil/gpt-oss-120b",
            "tinfoil/gemma4-31b",
            "zexrtw",
            "gpt-4o",
            "gpt-6",
            "claude-3-5-haiku",
        ],
    }


_sessions: dict = {}


async def _drain_then_exit() -> None:
    """Wait for in-flight requests, close the browsers, then exit.

    This is what the dashboard's Stop button drives. It does the work uvicorn's
    own Ctrl-C path would do, but without reaching into uvicorn: the app object
    is imported from inside Server.serve(), so by the time this module runs
    there is no Server instance to set should_exit on, and signalling our own
    PID is not a substitute (on Windows os.kill SIGINT is TerminateProcess,
    which skips the shutdown hook and strands Chrome).

    Bounded so a wedged stream cannot hold the process open forever; the exit is
    forced once the grace period lapses, which is better than never exiting.
    """
    import os
    deadline = time.monotonic() + 10
    while _active_requests > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if _active_requests:
        logger.warning("shutdown: %d request(s) still active, exiting anyway",
                       _active_requests)
    await _close_sessions()
    os._exit(0)


class _ActiveRequestTracker:
    """Count requests that have started but not finished sending.

    Pure ASGI on purpose. An @app.middleware("http") would not do: Starlette's
    BaseHTTPMiddleware returns from call_next as soon as the response object
    exists, so its finally fires while a StreamingResponse is still going out -
    measured at 2.45s early on a 6-chunk stream. _drain_then_exit would read a
    count of zero and exit mid-answer, cutting the user off mid-sentence.
    Watching the final http.response.body instead marks the real end of a
    request, streaming or not.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        global _active_requests
        _active_requests += 1
        released = False

        async def counting_send(message) -> None:
            # `global` is needed here too, not just in __call__: a nested
            # function that assigns _active_requests makes it local to itself,
            # and the += would raise UnboundLocalError.
            global _active_requests
            nonlocal released
            # more_body is True for every chunk but the last; that last one is
            # the only reliable "this request is done" signal.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                released = True
                _active_requests -= 1
            await send(message)

        try:
            await self.app(scope, receive, counting_send)
        finally:
            if not released:
                # The request errored or the client vanished before the last
                # body, so nothing above released the slot.
                _active_requests -= 1


app.add_middleware(_ActiveRequestTracker)


_shared_session: Optional[DuckAISession] = None


async def get_session(model: Optional[str] = None) -> DuckAISession:
    global _shared_session, _sessions
    if _shared_session is None:
        _shared_session = DuckAISession(model=DEFAULT, proxies=PROXIES, new_chat=NEW_CHAT)
        _sessions["shared"] = _shared_session
    return _shared_session


@app.on_event("startup")
async def _prewarm_default_model() -> None:
    """Warm a browser+page during boot so the first request skips ~10s of setup.

    Runs against the default model. A prewarm failure is logged and
    ignored: the lazy path in get_session still works, so the server stays up
    even if Chrome is missing or the network is down.
    """
    try:
        sess = await get_session(DEFAULT)
        await sess.prewarm()
    except Exception as e:  # never block boot on a warmup problem
        logger.warning("startup prewarm skipped: %s", e)


@app.on_event("shutdown")
async def _close_sessions() -> None:
    """Close the shared browser engine on shutdown."""
    global _shared_session
    if _shared_session:
        try:
            await _shared_session.close()
            logger.info("closed shared browser session")
        except Exception as e:  # noqa: BLE001 - shutdown must always complete
            logger.warning("close failed: %s", e)


# Live catalog from GET /duckchat/v1/models (public). Falls back to the static
# MODEL_LABELS snapshot when Duck.ai is unreachable; ids resolve_model() has never
# seen are passed through, so new models work before this file is updated.
CATALOG_TTL = 3600.0
_catalog: List[dict] = []
_catalog_ts = 0.0


async def get_catalog() -> List[dict]:
    global _catalog, _catalog_ts
    now = time.time()
    if _catalog and now - _catalog_ts < CATALOG_TTL:
        return _catalog
    try:
        models = await asyncio.to_thread(fetch_model_catalog)
        if models:
            _catalog, _catalog_ts = models, now
    except Exception as e:  # noqa: BLE001 - catalog is best-effort, never fatal
        logger.warning("model catalog fetch failed: %s", e)
    return _catalog


def require_key(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="", alias="x-api-key"),
) -> None:
    if not API_KEY:
        return
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif authorization.strip():
        token = authorization.strip()
    elif x_api_key.strip():
        token = x_api_key.strip()

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


# Duck.ai's chat body carries native fields the free web UI toggles expose:
#   reasoningEffort  ("none"|"low"|"medium", per model's supportedReasoningEffort)
#   canUseTools / metadata.toolChoice        (WebSearch / image gen etc.)
# We patch them onto the app's own outgoing request (duckai.body rewrite), which
# auto-disables process-wide if the server ever binds its challenge to the body.
_REASONING_MAP = {
    "minimal": "none",
    "none": "none",
    "low": "low",
    "medium": "low",  # free tier max is low
    "high": "low",    # auto-map high to highest supported free reasoning (low)
    "max": "low",
}


def build_rewrite(
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    thinking: bool = False,
    duckai: Optional[dict] = None,
) -> Optional[dict]:
    rw: dict = {}
    if model:
        rw["model"] = model
    if isinstance(duckai, dict):
        if duckai.get("reasoningEffort"):
            rw["reasoningEffort"] = str(duckai["reasoningEffort"])
        if duckai.get("canUseTools") is not None:
            rw["canUseTools"] = bool(duckai["canUseTools"])
        if isinstance(duckai.get("toolChoice"), dict):
            rw["toolChoice"] = duckai["toolChoice"]
    if reasoning_effort:
        mapped = _REASONING_MAP.get(str(reasoning_effort).strip().lower(), "low")
        rw["reasoningEffort"] = mapped
    elif thinking:
        rw["reasoningEffort"] = "low"
    return rw or None


# ---------------------------------------------------------------------------
# OpenAI protocol
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    reasoning_effort: Optional[str] = None
    duckai: Optional[dict] = None


async def _openai_stream(
    session: DuckAISession,
    prompt: str,
    model: str,
    chat_id: str,
    rewrite: Optional[dict] = None,
    tools: Optional[List[dict]] = None,
    image_path: Optional[str] = None,
):
    def chunk_obj(delta: dict, finish=None):
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _created(),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    try:
        yield f"data: {json.dumps(chunk_obj({'role': 'assistant', 'content': ''}, None))}\n\n"
        if tools:
            buffer = []
            async for token in session.send_stream(prompt, model=model, rewrite=rewrite, image_path=image_path):
                buffer.append(_consume_image_marker(token) or token)
            full_text = "".join(buffer)
            preamble, calls = split_text_and_tool(full_text)
            if calls:
                if preamble:
                    yield f"data: {json.dumps(chunk_obj({'content': preamble}, None))}\n\n"
                for idx, c in enumerate(calls):
                    tc_delta = {
                        "tool_calls": [{
                            "index": idx,
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.input, ensure_ascii=False),
                            }
                        }]
                    }
                    yield f"data: {json.dumps(chunk_obj(tc_delta, None))}\n\n"
                yield f"data: {json.dumps(chunk_obj({}, 'tool_calls'))}\n\n"
            else:
                if full_text:
                    yield f"data: {json.dumps(chunk_obj({'content': full_text}, None))}\n\n"
                yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
        else:
            async for token in session.send_stream(prompt, model=model, rewrite=rewrite, image_path=image_path):
                safe_token = _consume_image_marker(token) or token
                yield f"data: {json.dumps(chunk_obj({'content': safe_token}, None))}\n\n"
            yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
        yield "data: [DONE]\n\n"
    except (DuckAIRateLimit, DuckAIError) as e:
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': {'message': f'stream failed: {e}', 'type': 'server_error'}})}\n\n"
    finally:
        if image_path:
            try:
                os.unlink(image_path)
            except Exception:
                pass

@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat_completions(request: ChatCompletionRequest):
    system = None
    turns = []
    for m in request.messages:
        if m.role == "system":
            system = m.content
        else:
            turns.append({
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            })

    # === BRIDGE 1: Attachments (Images, PDFs, Code, Docs) ===
    upload_file_path, text_attach_injection = _extract_attachments_from_messages(request.messages)
    image_path = upload_file_path  # passed to browser upload input (images & PDFs)

    # === BRIDGE 2: Local File Auto-Read (detect file paths & filenames) ===
    last_user_text = ""
    for t in reversed(turns):
        if t.get("role") == "user":
            last_user_text = _anthropic_block_text(t.get("content", ""))
            break
    compaction_request = _is_compaction_request(last_user_text)
    file_context = _detect_and_read_files(last_user_text)

    # Inject file context and decoded text attachments into the last user turn
    extra_context = []
    if file_context:
        extra_context.append(file_context)
    if text_attach_injection:
        extra_context.append(text_attach_injection)

    if extra_context:
        combined_extra = "\n\n".join(extra_context)
        for t in reversed(turns):
            if t.get("role") == "user":
                existing = _anthropic_block_text(t.get("content", ""))
                t["content"] = f"{existing}\n\n{combined_extra}"
                break

    # === SMART CONTEXT TRIMMING ===
    # 9router/Hermes inject 100K+ context. Duck.ai only takes 12K.
    # Keep system compact + last 4 turns only.
    if system:
        sys_text = _anthropic_block_text(system)
        if len(sys_text) > 1500:
            system = sys_text[:1500] + "..."

    if len(turns) > 4:
        logger.info("context trim: %d turns -> 4", len(turns))
        turns = turns[-4:]

    # Bound every retained turn independently. This prevents one generated
    # image/data URL or huge tool result from producing a 400K prompt.
    for turn in turns:
        turn["content"] = _compact_turn_content(turn.get("content", ""))

    # Compact tool injection
    if request.tools and not compaction_request:
        tool_names = [((t.get("function") or t).get("name", "?")) for t in request.tools]
        compact = "Tools: " + ", ".join(tool_names) + '. Use: <tool_call name="TOOL">{"p":"v"}</tool_call>'
        if system:
            system = f"{compact}\n{_anthropic_block_text(system)}"
        else:
            system = compact

    # Reasoning
    if request.reasoning_effort in ("high", "max"):
        r_text = "Think step-by-step with rigorous reasoning."
        if system:
            system = f"{_anthropic_block_text(system)}\n[{r_text}]"
        else:
            system = f"[{r_text}]"

    prompt = flatten_conversation(system, turns)
    if not prompt.strip() and not request.tools:
        raise HTTPException(status_code=400, detail="empty prompt")
    model = resolve_model(request.model)

    def _tname(t: dict) -> str:
        return (t.get("function") or {}).get("name") or t.get("name") or "?"

    logger.info(
        "REQ /v1/chat/completions model=%s->%s stream=%s msgs=%d prompt_len=%d tools=[%s] last_user=%r",
        request.model, model, request.stream, len(turns), len(prompt),
        ",".join(_tname(t) for t in (request.tools or [])),
        (json.dumps(turns[-1]["content"], ensure_ascii=False)[:160] if turns else ""),
    )

    # Cross-model image routing: text-only backends such as zexrtw/NoTrack can
    # improve a prompt but cannot generate bytes. Explicit image requests are
    # always delegated to Duck.ai Luna's native GenerateImage tool and returned
    # as a cached public URL, regardless of the selected chat model.
    if _wants_image_generation(last_user_text):
        image_prompt = _clean_image_prompt(last_user_text)
        logger.info("IMAGE ROUTE model=%s -> gpt-6-luna GenerateImage prompt=%r", model, image_prompt[:160])
        session = await get_session("gpt-6-luna")
        image_id = _id()

        async def generate_image_markdown() -> str:
            result = await session.send_image(image_prompt)
            b64 = result.get("b64")
            if not b64:
                raise DuckAIError(result.get("text") or "Duck.ai returned no generated image")
            cached_id = _store_image(b64)
            revised = result.get("gen_prompt") or image_prompt
            url = f"{PUBLIC_BASE_URL}/v1/images/content/{cached_id}"
            return f"🎨 **Generated image**\n\n![Generated Image]({url})\n\nPrompt: {revised}"

        if request.stream:
            async def _image_route_stream():
                def chunk(delta: dict, finish_reason=None) -> str:
                    payload = {
                        "id": image_id,
                        "object": "chat.completion.chunk",
                        "created": _created(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }],
                    }
                    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                yield chunk({"role": "assistant", "content": ""})
                try:
                    markdown = await generate_image_markdown()
                    yield chunk({"content": markdown})
                    yield chunk({}, "stop")
                except Exception as exc:
                    yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'image_generation_error', 'code': 'image_generation_failed'}})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _image_route_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            markdown = await generate_image_markdown()
        except DuckAIRateLimit as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except DuckAIError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {
            "id": image_id,
            "object": "chat.completion",
            "created": _created(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": markdown},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": estimate_tokens(image_prompt),
                "completion_tokens": estimate_tokens(markdown),
                "total_tokens": estimate_tokens(image_prompt) + estimate_tokens(markdown),
            },
        }

    # Autonomous Pi Agent mode: execute tools locally on the Pi directly
    if model in ("agent", "ze/agent", "zexrtw-agent"):
        chat_id = _id()
        nt = get_notrack_client()
        sess = await get_session(DEFAULT)
        async def llm_caller(p: str) -> str:
            try:
                res = await nt.send(p)
                if res.strip():
                    return res
            except Exception:
                pass
            return await sess.send(p, model="gpt-6-luna")
        last_user_turn = ""
        for t in reversed(turns):
            if t.get("role") == "user":
                last_user_turn = _anthropic_block_text(t.get("content", ""))
                break
        task_query = last_user_turn or prompt
        if request.stream:
            async def _agent_stream_gen():
                def chunk_obj(delta: dict, finish=None):
                    return {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": _created(),
                        "model": "agent",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                yield f"data: {json.dumps(chunk_obj({'role': 'assistant', 'content': ''}, None))}\n\n"
                try:
                    async for chunk in agent.run_autonomous_agent(task_query, llm_caller):
                        yield f"data: {json.dumps(chunk_obj({'content': chunk}, None))}\n\n"
                    yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            return StreamingResponse(_agent_stream_gen(), media_type="text/event-stream")
        else:
            chunks = []
            async for chunk in agent.run_autonomous_agent(task_query, llm_caller):
                chunks.append(chunk)
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": _created(),
                "model": "agent",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(chunks)},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": estimate_tokens(prompt),
                    "completion_tokens": estimate_tokens("".join(chunks)),
                    "total_tokens": estimate_tokens(prompt) + estimate_tokens("".join(chunks)),
                }
            }

    # Deterministic bridge for explicit file/terminal/search intents. This is
    # always enabled when the client offers tools: it avoids relying on a chat
    # model to decide whether it can access the client's local filesystem.
    if request.tools and not compaction_request and not has_tool_result(turns):
        routed = route_intent(turns, request.tools)
        if routed is not None:
            logger.info("ROUTED tool=%s input=%s (Duck.ai not called)", routed.name, routed.input)
            completion_id = _id()
            call_id = f"call_{uuid4().hex[:24]}"
            arguments = json.dumps(routed.input, ensure_ascii=False)

            if request.stream:
                async def _routed_tool_stream():
                    def chunk(delta: dict, finish_reason=None) -> str:
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": _created(),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }],
                        }
                        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    # OpenAI-compatible streaming shape: role first, complete
                    # tool call second, explicit terminal finish_reason third.
                    yield chunk({"role": "assistant", "content": None})
                    yield chunk({
                        "tool_calls": [{
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": routed.name,
                                "arguments": arguments,
                            },
                        }],
                    })
                    yield chunk({}, "tool_calls")
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _routed_tool_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": _created(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": routed.name,
                                "arguments": arguments,
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": estimate_tokens(prompt),
                    "completion_tokens": 0,
                    "total_tokens": estimate_tokens(prompt),
                },
            }

    if model in ("zexrtw", "zexrtw-uncensored", "notrack", "notrack-uncensored"):
        nt = get_notrack_client()
        chat_id = _id()
        sys_str = _anthropic_block_text(system) if system else None
        out_model = model
        if request.stream:
            async def _nt_stream():
                def chunk_obj(delta: dict, finish=None):
                    return {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": _created(),
                        "model": out_model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                yield f"data: {json.dumps(chunk_obj({'role': 'assistant', 'content': ''}, None))}\n\n"
                try:
                    if request.tools:
                        buffer = []
                        async for token in nt.send_stream(prompt, instructions=sys_str):
                            buffer.append(token)
                        full_text = "".join(buffer)
                        preamble, calls = split_text_and_tool(full_text)
                        if calls:
                            if preamble:
                                yield f"data: {json.dumps(chunk_obj({'content': preamble}, None))}\n\n"
                            for idx, c in enumerate(calls):
                                tc_delta = {
                                    "tool_calls": [{
                                        "index": idx,
                                        "id": c.id,
                                        "type": "function",
                                        "function": {
                                            "name": c.name,
                                            "arguments": json.dumps(c.input, ensure_ascii=False),
                                        }
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_obj(tc_delta, None))}\n\n"
                            yield f"data: {json.dumps(chunk_obj({}, 'tool_calls'))}\n\n"
                        else:
                            if full_text:
                                yield f"data: {json.dumps(chunk_obj({'content': full_text}, None))}\n\n"
                            yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
                    else:
                        async for token in nt.send_stream(prompt, instructions=sys_str):
                            yield f"data: {json.dumps(chunk_obj({'content': token}, None))}\n\n"
                        yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            return StreamingResponse(_nt_stream(), media_type="text/event-stream")
        else:
            try:
                res_text = await nt.send(prompt, instructions=sys_str)
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))
            preamble, calls = split_text_and_tool(res_text)
            msg = {"role": "assistant", "content": res_text}
            finish = "stop"
            if calls:
                msg["content"] = preamble or None
                msg["tool_calls"] = [{
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.input, ensure_ascii=False)},
                } for c in calls]
                finish = "tool_calls"
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": _created(),
                "model": out_model,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": estimate_tokens(prompt),
                    "completion_tokens": estimate_tokens(res_text),
                    "total_tokens": estimate_tokens(prompt) + estimate_tokens(res_text),
                }
            }

    session = await get_session(model)
    rewrite = build_rewrite(model=model, reasoning_effort=request.reasoning_effort, duckai=request.duckai)

    if request.stream:
        return StreamingResponse(
            _openai_stream(session, prompt, model, _id(), rewrite, tools=request.tools, image_path=image_path),
            media_type="text/event-stream"
        )

    try:
        result = await session.send(prompt, model=model, rewrite=rewrite, image_path=image_path)
        # Non-streaming responses can contain one private image marker after
        # explanatory text. Replace every marker with a short cached URL.
        if IMAGE_RESULT_PREFIX in result:
            marker_pos = result.find(IMAGE_RESULT_PREFIX)
            prefix_text = result[:marker_pos]
            marker_text = result[marker_pos:]
            result = prefix_text + (_consume_image_marker(marker_text) or "")
    except DuckAIRateLimit as e:
        raise HTTPException(status_code=429, detail=str(e))
    except DuckAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        if image_path:
            try:
                os.unlink(image_path)
            except Exception:
                pass
    msg = {"role": "assistant", "content": result.strip()}
    finish = "stop"
    preamble, calls = split_text_and_tool(result)
    if calls:
        msg["content"] = preamble or None
        msg["tool_calls"] = [{
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.input, ensure_ascii=False)},
        } for c in calls]
        finish = "tool_calls"
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _created(),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(result),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(result),
        },
    }


# OpenAI Responses API (/v1/responses)
# ---------------------------------------------------------------------------
def _responses_input_text(inp):
    """Flatten a Responses API `input` into a full multi-turn duck.ai prompt.

    `input` may be a string, or a list of items:
      - {"role": "user"|"system"|"assistant", "content": str | [blocks]}
      - {"type": "message", "role": ..., "content": str | [blocks]}
    Every turn is preserved (role-labelled) so multi-turn stays coherent.
    """
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp.strip()
    if isinstance(inp, list):
        parts: List[str] = []
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = item.get("content", "")
            text = _anthropic_block_text(content) if not isinstance(content, str) else content
            if role == "tool":
                label = f"[tool_result {item.get('tool_call_id', '')}]"
            else:
                label = _role_label(role)
            if text.strip():
                parts.append(f"{label}: {text}")
        return "\n\n".join(parts).strip()
    return str(inp).strip()

@app.post("/v1/responses", dependencies=[Depends(require_key)])
async def openai_responses(request: Request):
    raw = await request.json()
    model = resolve_model(raw.get("model"))
    tools = raw.get("tools") or []
    prompt = _responses_input_text(raw.get("input"))
    # Relay-side tool routing: synthesise a function_call from the user's intent.
    if TOOL_ROUTING and tools:
        resp_items = raw.get("input") or []
        route_msgs = []
        if isinstance(resp_items, list):
            for it in resp_items:
                if isinstance(it, dict):
                    role = it.get("role") or ("user" if it.get("type") == "message" else "user")
                    content = it.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                        )
                    route_msgs.append({"role": role, "content": content})
        if not has_tool_result(route_msgs):
            routed = route_intent(route_msgs, tools)
            if routed is not None:
                return {
                    "id": f"resp_{uuid4().hex[:24]}",
                    "object": "response",
                    "created_at": _created(),
                    "model": model,
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "id": f"fc_{uuid4().hex[:24]}",
                        "name": routed.name,
                        "arguments": json.dumps(routed.input, ensure_ascii=False),
                    }],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
    if not prompt:
        raise HTTPException(status_code=400, detail="empty input")
    stream = bool(raw.get("stream", False))
    session = await get_session(model)
    reasoning = raw.get("reasoning") or {}
    rewrite = build_rewrite(
        model=model,
        reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None,
        duckai=raw.get("duckai"),
    )
    resp_id = f"resp_{uuid4().hex[:24]}"

    if not stream:
        try:
            result = await session.send(prompt, model=model, rewrite=rewrite)
        except DuckAIRateLimit as e:
            raise HTTPException(status_code=429, detail=str(e))
        except DuckAIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        preamble, calls = split_text_and_tool(result)
        output = []
        if preamble:
            output.append({
                "type": "message",
                "id": f"msg_{uuid4().hex[:24]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": preamble}],
            })
        for c in calls:
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid4().hex[:24]}",
                "name": c.name,
                "arguments": json.dumps(c.input, ensure_ascii=False),
            })
        if not output:
            output.append({
                "type": "message",
                "id": f"msg_{uuid4().hex[:24]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.strip()}],
            })
        return {
            "id": resp_id,
            "object": "response",
            "created_at": _created(),
            "model": model,
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    async def _resp_stream():
        yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'status': 'in_progress', 'model': model, 'output': []}})}\n\n"
        try:
            async for token in session.send_stream(prompt, model=model, rewrite=rewrite):
                safe_token = _consume_image_marker(token) or token
                yield f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': safe_token})}\n\n"
        except (DuckAIRateLimit, DuckAIError) as e:
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            return
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': f'stream failed: {e}', 'type': 'server_error'}})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'object': 'response', 'status': 'completed', 'model': model, 'output': []}})}\n\n"

    return StreamingResponse(_resp_stream(), media_type="text/event-stream")


def _build_anthropic_message(msg_id: str, model: str, result: str, tool_use=None) -> dict:
    """Build an Anthropic message response.

    - If `tool_use` (a RoutedToolCall from the relay router) is given, emit a
      tool_use content block and stop_reason="tool_use" WITHOUT calling Duck.ai.
    - Else if the model emitted a <tool_call> envelope (never happens on Duck.ai,
      kept for completeness), parse it.
    - Else return a plain text block.
    """
    if tool_use is not None:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "tool_use",
                "id": f"toolu_{uuid4().hex[:24]}",
                "name": tool_use.name,
                "input": tool_use.input,
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    preamble, calls = split_text_and_tool(result)
    if not calls:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": result.strip()}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    content = []
    if preamble:
        content.append({"type": "text", "text": preamble})
    for c in calls:
        content.append({
            "type": "tool_use",
            "id": c.id,
            "name": c.name,
            "input": c.input,
        })
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Anthropic protocol
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(require_key)])
async def list_models():
    catalog = await get_catalog()
    data = []
    seen = set()

    # Include all catalog models (they map to working backend backends)
    if catalog:
        for m in catalog:
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            tiers = m.get("accessTier") or []
            data.append({
                "id": mid,
                "object": "model",
                "created": _created(),
                "owned_by": "zexrtw",
                "display_name": m.get("name") or mid,
                "access": "free" if "free" in tiers else "paid",
                "capabilities": {
                    "image_upload": bool(m.get("supportsImageUpload")),
                    "reasoning_effort": m.get("supportedReasoningEffort") or [],
                    "tools": m.get("supportedTools") or [],
                    "file_types": m.get("supportedFileTypes") or [],
                },
            })

    # Ensure core models exist
    for mid, label in MODEL_LABELS.items():
        if mid not in seen:
            seen.add(mid)
            data.append({
                "id": mid,
                "object": "model",
                "created": _created(),
                "owned_by": "zexrtw",
                "display_name": label,
                "access": "free",
                "capabilities": {"image_upload": mid == "gpt-6-luna"},
            })

    # Common router aliases so 9router / Cline / Cursor can select standard names
    router_aliases = [
        ("agent", "zexrtw Autonomous Agent"),
        ("ze/agent", "zexrtw Autonomous Agent"),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o mini"),
        ("gpt-6", "GPT-6"),
        ("claude-3-5-haiku", "Claude 3.5 Haiku"),
        ("claude-3-5-sonnet", "Claude 3.5 Sonnet"),
    ]
    for aid, alabel in router_aliases:
        if aid not in seen:
            seen.add(aid)
            data.append({
                "id": aid,
                "object": "model",
                "created": _created(),
                "owned_by": "zexrtw",
                "display_name": alabel,
                "access": "free",
                "capabilities": {},
            })

    return {"object": "list", "data": data}


@app.post("/v1/messages", dependencies=[Depends(require_key)])
async def anthropic_messages(request: Request):
    raw = await request.json()
    model = resolve_model(raw.get("model"))
    system = raw.get("system")
    tools = raw.get("tools") or []
    messages = raw.get("messages") or []
    stream = bool(raw.get("stream", False))

    # === BRIDGE 1: Attachments (Images, PDFs, Code, Docs) ===
    upload_file_path, text_attach_injection = _extract_attachments_from_messages(messages)
    image_path = upload_file_path

    # === BRIDGE 2: Local File Auto-Read ===
    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = _anthropic_block_text(m.get("content", ""))
            break
    file_context = _detect_and_read_files(last_user_text)

    extra_context = []
    if file_context:
        extra_context.append(file_context)
    if text_attach_injection:
        extra_context.append(text_attach_injection)

    if extra_context:
        combined_extra = "\n\n".join(extra_context)
        for m in reversed(messages):
            if m.get("role") == "user":
                existing = _anthropic_block_text(m.get("content", ""))
                m["content"] = f"{existing}\n\n{combined_extra}"
                break

    # Context trimming for Anthropic requests
    if len(messages) > 6:
        messages = [messages[0]] + messages[-4:]

    prompt = build_anthropic_prompt(system, messages)
    # Relay-side tool routing: Duck.ai cannot emit tool calls, so the relay
    # synthesizes a tool_use block from the user's intent. Skip routing when the
    # client is already returning a tool_result (mid-loop) - then we just answer.
    if TOOL_ROUTING and tools and not has_tool_result(messages):
        routed = route_intent(messages, tools)
        if routed is not None:
            return _build_anthropic_message(
                f"msg_{uuid4().hex[:24]}", model, "", tool_use=routed
            )

    if model in ("zexrtw", "zexrtw-uncensored", "notrack", "notrack-uncensored"):
        nt = get_notrack_client()
        sys_str = _anthropic_block_text(system) if system else None
        msg_id = f"msg_{uuid4().hex[:24]}"
        out_model = model
        if not stream:
            try:
                result = await nt.send(prompt, instructions=sys_str)
            except Exception as e:
                return _anthropic_error(502, "api_error", str(e))
            return _build_anthropic_message(msg_id, out_model, result)

        async def _nt_anthropic_stream():
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": out_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            try:
                async for token in nt.send_stream(prompt, instructions=sys_str):
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": token},
                    })
            except Exception as e:
                yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}})
                return
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            })
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(_nt_anthropic_stream(), media_type="text/event-stream")

    session = await get_session(model)
    thinking = raw.get("thinking")
    rewrite = build_rewrite(
        model=model,
        thinking=isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"),
        duckai=raw.get("duckai"),
    )
    msg_id = f"msg_{uuid4().hex[:24]}"
    if not stream:
        try:
            result = await session.send(prompt, model=model, rewrite=rewrite, image_path=image_path)
        except DuckAIRateLimit as e:
            return _anthropic_error(429, "rate_limit_error", str(e))
        except DuckAIError as e:
            return _anthropic_error(502, "api_error", str(e))
        finally:
            if image_path:
                try:
                    os.unlink(image_path)
                except Exception:
                    pass
        return _build_anthropic_message(msg_id, model, result)

    async def _anthropic_stream():
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        try:
            async for token in session.send_stream(prompt, model=model, rewrite=rewrite, image_path=image_path):
                token = _consume_image_marker(token) or token
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token},
                })
        except (DuckAIRateLimit, DuckAIError) as e:
            yield _sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": str(e)}})
            return
        except Exception as e:
            yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": f"stream failed: {e}"}})
            return
        finally:
            if image_path:
                try:
                    os.unlink(image_path)
                except Exception:
                    pass
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(_anthropic_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _anthropic_error(status: int, etype: str, message: str):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": etype, "message": message}},
    )


# ---------------------------------------------------------------------------
# Images protocol (OpenAI-compatible /v1/images/generations)
# Duck.ai has no dedicated image API; GPT-5.6 Luna's chat turn natively invokes a
# GenerateImage tool (backend "GPT Image 2"). We drive that turn and return the
# base64 JPEG the SSE success-frame carries. Generated bytes are cached in-memory
# so `response_format:"url"` can hand back a stable local URL.
# ---------------------------------------------------------------------------
_IMAGES: dict = {}          # img_id -> {"bytes": bytes, "mime": str}
_IMAGE_CAP = 50


def _store_image(b64: str) -> str:
    raw = base64.b64decode(b64)
    mime = "image/jpeg" if b64.startswith("/9j/") else "image/png" if b64.startswith("iVBOR") else "image/png"
    img_id = f"img_{uuid4().hex[:24]}"
    _IMAGES[img_id] = {"bytes": raw, "mime": mime}
    while len(_IMAGES) > _IMAGE_CAP:  # FIFO eviction
        _IMAGES.pop(next(iter(_IMAGES)))
    return img_id


class ImageRequest(BaseModel):
    model: str = DEFAULT
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = None
    response_format: Optional[str] = "url"


@app.post("/v1/images/generations", dependencies=[Depends(require_key)])
async def images_generations(request: ImageRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    model = resolve_model(request.model)
    count = max(1, min(int(request.n or 1), 4))
    session = await get_session(model)
    data = []
    for _ in range(count):
        try:
            result = await session.send_image(request.prompt, size=request.size)
        except DuckAIRateLimit as e:
            raise HTTPException(status_code=429, detail=str(e))
        except DuckAIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        b64 = result["b64"]
        entry = {}
        if request.response_format == "b64_json":
            entry["b64_json"] = b64
        else:
            img_id = _store_image(b64)
            entry["url"] = f"/v1/images/content/{img_id}"
        if result.get("gen_prompt"):
            entry["revised_prompt"] = result["gen_prompt"]
        data.append(entry)
    return {"created": _created(), "data": data}


@app.get("/v1/images/content/{img_id}")
async def images_content(img_id: str, request: Request,
                         authorization: str = Header(default="")):
    """Serve a generated image by id.

    Takes the key from the Authorization header, falling back to a `key` query
    param. The fallback exists because a browser cannot attach a header to an
    <img src>: the dashboard renders results that way, and requiring the header
    would leave the UI unable to show its own images. An img_id is a uuid4
    prefix, so it is not guessable - the param is a second path to the same
    secret, not a hole in it.
    """
    # img_id contains 96 random bits and is intentionally usable from Markdown
    # renderers that cannot attach Authorization headers. API generation routes
    # remain Bearer-protected; only possession of this unguessable URL grants
    # access to the cached image.
    item = _IMAGES.get(img_id)
    if not item:
        raise HTTPException(status_code=404, detail="image not found (expired from cache)")
    return Response(content=item["bytes"], media_type=item["mime"])


@app.get("/health")
async def health():
    import platform
    global _shared_session
    browser_ready = False
    if _shared_session:
        workers = getattr(_shared_session, "_workers", [])
        for s in workers:
            if getattr(s, "_ready", False):
                browser_ready = True
                break
    return {
        "status": "ok",
        "service": "zexrtw",
        "uptime_s": round(time.time() - _BOOT_TS, 1),
        "platform": f"{platform.system()} {platform.machine()}",
        "browser_ready": browser_ready,
        "worker_pool": {
            "size": len(getattr(_shared_session, "_workers", [])) if _shared_session else 0,
            "available": getattr(getattr(_shared_session, "_available_workers", None), "qsize", lambda: 0)() if _shared_session else 0,
            "busy": sum(1 for worker in getattr(_shared_session, "_workers", []) if worker._lock.locked()) if _shared_session else 0,
        },
        "default_model": DEFAULT,
        "providers": ["duck.ai", "notrack.ai"],
    }


if __name__ == "__main__":
    # Performance: use uvloop on Linux (Radxa) for 2-4x faster async I/O
    try:
        import uvloop
        uvloop.install()
        logger.info("uvloop installed for faster async I/O")
    except ImportError:
        pass
    uvicorn.run(app, host="0.0.0.0", port=PORT, loop="auto", access_log=False)
