"""NoTrack.ai Uncensored AI Relay Client.

High-throughput direct HTTP streaming client for NoTrack (https://notrack.ai/chat).
Supports both httpx (connection-pooled async) and standard urllib fallback.
Zero browser RAM, optimized for 9router and Radxa SBC (ARM64).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import urllib.request
import uuid
from typing import AsyncIterator, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore
    HAS_HTTPX = False

logger = logging.getLogger("notrack")

NOTRACK_BASE_URL = "https://notrack.ai/chat"
NOTRACK_DISPATCH_URL = "https://notrack.ai/api/dispatch"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
]


class NoTrackError(Exception):
    pass


class NoTrackRateLimit(NoTrackError):
    pass


NOTRACK_MAX_INPUT = 3900  # server hard limit 4000 chars, keep margin


def _clamp_notrack(prompt: str) -> str:
    """Clamp prompt to NoTrack's 4000 char limit, keeping head+tail."""
    if len(prompt) <= NOTRACK_MAX_INPUT:
        return prompt
    marker = "\n…[context truncated]…\n"
    head = max(200, NOTRACK_MAX_INPUT // 5)
    tail = NOTRACK_MAX_INPUT - head - len(marker)
    return prompt[:head] + marker + prompt[-tail:]


class NoTrackClient:
    """Non-blocking HTTP client for notrack.ai with cookie session minting."""

    def __init__(self, default_persona: str = "normal", timeout: float = 60.0):
        self.default_persona = default_persona
        self.timeout = timeout
        self.uid: Optional[str] = None
        self._client: Optional[object] = None
        self._ua = random.choice(USER_AGENTS)

    async def _ensure_session(self) -> None:
        """Fetch legitimate anonymous cookie from notrack.ai."""
        if self.uid:
            return
        await self.refresh_session()

    async def refresh_session(self) -> None:
        """Get a fresh session cookie directly from the landing page."""
        client = await self._get_httpx_client()
        if HAS_HTTPX and client:
            try:
                client.cookies.clear()
            except Exception:
                pass
        self._ua = random.choice(USER_AGENTS)
        self.uid = str(uuid.uuid4())
        headers = {"User-Agent": self._ua}
        try:
            if HAS_HTTPX and client:
                res = await client.get(NOTRACK_BASE_URL, headers=headers)
                cookie_uid = client.cookies.get("uid")
                if cookie_uid:
                    self.uid = cookie_uid
                    logger.info("Minted fresh NoTrack session UID: %s", self.uid)
                    return
        except Exception as e:
            logger.warning("Failed to refresh NoTrack session via httpx: %s", e)

        # Fallback to UUID
        self.uid = str(uuid.uuid4())

    async def _get_httpx_client(self):
        if not HAS_HTTPX:
            return None
        if self._client is None or getattr(self._client, "is_closed", True):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                follow_redirects=True,
            )
        return self._client

    def _headers(self, custom_uid: Optional[str] = None) -> dict:
        uid = custom_uid or self.uid or str(uuid.uuid4())
        return {
            "Content-Type": "application/json",
            "User-Agent": self._ua,
            "Origin": "https://notrack.ai",
            "Referer": "https://notrack.ai/chat",
            "Cookie": f"uid={uid}",
            "Accept": "text/event-stream",
            "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    async def send_stream(
        self,
        prompt: str,
        persona: Optional[str] = None,
        instructions: Optional[str] = None,
        chat_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> AsyncIterator[str]:
        """Stream tokens incrementally from NoTrack SSE endpoint with auto-rotation on limits."""
        await self._ensure_session()

        # NoTrack counts instructions + user_input together against 4000 char limit.
        # Merge instructions into prompt, then clamp the combined result.
        if instructions and instructions.strip():
            # Compact the system prompt: keep only the essential first 600 chars
            compact_instr = instructions.strip()
            if len(compact_instr) > 600:
                compact_instr = compact_instr[:600] + "…"
            prompt = f"[System: {compact_instr}]\n\n{prompt}"
            instructions = None  # don't send separately

        prompt = _clamp_notrack(prompt)

        if HAS_HTTPX:
            async for token in self._stream_httpx(prompt, persona, instructions, chat_id, max_retries):
                yield token
        else:
            async for token in self._stream_urllib(prompt, persona, instructions, chat_id, max_retries):
                yield token

    async def _stream_httpx(
        self,
        prompt: str,
        persona: Optional[str] = None,
        instructions: Optional[str] = None,
        chat_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> AsyncIterator[str]:
        client = await self._get_httpx_client()
        for attempt in range(max_retries + 1):
            payload = {
                "user_input": prompt,
                "mode": "usual",
                "model": "C",
                "persona": persona or self.default_persona,
                "max_turns": 1,
                "chat_id": chat_id,
                "attachments": [],
                "regenerate": False,
                "edit": False,
                "via": "typed",
            }
            if instructions and instructions.strip():
                payload["instructions"] = instructions.strip()

            headers = self._headers()
            rate_limited = False
            got_tokens = False

            try:
                async with client.stream(
                    "POST",
                    NOTRACK_DISPATCH_URL,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code == 429:
                        rate_limited = True
                        await self.refresh_session()
                    elif response.status_code != 200:
                        err_text = await response.aread()
                        raise NoTrackError(f"HTTP {response.status_code}: {err_text.decode('utf-8', errors='ignore')[:200]}")
                    else:
                        async for raw_line in response.aiter_lines():
                            line = raw_line.strip()
                            if not line.startswith("data: "):
                                continue
                            raw_json = line[6:].strip()
                            if raw_json == "[DONE]":
                                break
                            try:
                                ev = json.loads(raw_json)
                                ev_type = ev.get("type")
                                if ev_type == "delta":
                                    chunk = ev.get("chunk")
                                    if chunk:
                                        got_tokens = True
                                        yield chunk
                                elif ev_type in ("message", "consensus"):
                                    if not got_tokens:
                                        msg_content = ev.get("content")
                                        if msg_content:
                                            got_tokens = True
                                            yield msg_content
                                elif ev_type == "done":
                                    break
                                elif ev_type == "error":
                                    if ev.get("code") == "ratelimit":
                                        rate_limited = True
                                        await self.refresh_session()
                                        break
                                    raise NoTrackError(ev.get("message") or "Unknown error")
                            except json.JSONDecodeError:
                                continue

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                logger.warning("NoTrack network retry %d/%d: %s", attempt + 1, max_retries, e)
                if attempt == max_retries:
                    raise NoTrackError(f"NoTrack connection failed: {e}")
                await asyncio.sleep(0.5)
                continue

            if got_tokens:
                return

            if rate_limited:
                if attempt < max_retries:
                    logger.warning("NoTrack rate limit hit; refreshed session and retrying...")
                    await asyncio.sleep(0.5)
                    continue
                raise NoTrackRateLimit("NoTrack rate limit exceeded after retries")

            return

    async def _stream_urllib(
        self,
        prompt: str,
        persona: Optional[str] = None,
        instructions: Optional[str] = None,
        chat_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> AsyncIterator[str]:
        for attempt in range(max_retries + 1):
            payload = {
                "user_input": prompt,
                "mode": "usual",
                "model": "C",
                "persona": persona or self.default_persona,
                "max_turns": 1,
                "chat_id": chat_id,
                "attachments": [],
                "regenerate": False,
                "edit": False,
                "via": "typed",
            }
            if instructions and instructions.strip():
                payload["instructions"] = instructions.strip()

            headers = self._headers()
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(NOTRACK_DISPATCH_URL, data=data, headers=headers)

            queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def sync_worker():
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        for raw_line in resp:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line.startswith("data: "):
                                continue
                            raw_json = line[6:].strip()
                            if raw_json == "[DONE]":
                                break
                            try:
                                ev = json.loads(raw_json)
                                loop.call_soon_threadsafe(queue.put_nowait, ev)
                                if ev.get("type") == "done":
                                    break
                            except json.JSONDecodeError:
                                continue
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except urllib.error.HTTPError as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "_error", "code": e.code, "msg": str(e)})
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "_error", "code": 500, "msg": str(e)})

            asyncio.create_task(asyncio.to_thread(sync_worker))

            got_tokens = False
            rate_limited = False
            error_msg = None

            while True:
                item = await queue.get()
                if item is None:
                    break
                ev_type = item.get("type")
                if ev_type == "delta":
                    chunk = item.get("chunk")
                    if chunk:
                        got_tokens = True
                        yield chunk
                elif ev_type in ("message", "consensus"):
                    if not got_tokens:
                        msg_content = item.get("content")
                        if msg_content:
                            got_tokens = True
                            yield msg_content
                elif ev_type == "_error":
                    code = item.get("code")
                    if code == 429:
                        rate_limited = True
                        self.uid = str(uuid.uuid4())
                    else:
                        error_msg = item.get("msg")
                elif ev_type == "error":
                    if item.get("code") == "ratelimit":
                        rate_limited = True
                        self.uid = str(uuid.uuid4())
                    else:
                        error_msg = item.get("message")

            if got_tokens:
                return

            if rate_limited and attempt < max_retries:
                logger.warning("NoTrack rate limit hit; retrying...")
                await asyncio.sleep(0.5)
                continue

            if rate_limited:
                raise NoTrackRateLimit("NoTrack rate limit exceeded")
            if error_msg:
                raise NoTrackError(f"NoTrack error: {error_msg}")
            return

    async def send(
        self,
        prompt: str,
        persona: Optional[str] = None,
        instructions: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        parts = []
        async for token in self.send_stream(prompt, persona, instructions, chat_id):
            parts.append(token)
        return "".join(parts)

    async def close(self) -> None:
        if HAS_HTTPX and self._client and not getattr(self._client, "is_closed", True):
            await self._client.aclose()
            self._client = None
