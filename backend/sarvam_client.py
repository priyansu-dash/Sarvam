"""
Thin async wrapper around the Sarvam AI REST API.

All Sarvam traffic goes through this module. The API key is read from the
environment once and is never logged or returned to callers.

We use raw REST via httpx (rather than the `sarvamai` SDK) so the request
contracts match the live API reference exactly — see DECISIONS.md.
"""

from __future__ import annotations

import asyncio
import html
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, AsyncIterator
import zipfile

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("sarvam")

BASE_URL = "https://api.sarvam.ai"

# ---- Model / voice defaults (one TTS model + its own voice catalogue) ----
CHAT_MODEL = "sarvam-105b"
STT_MODEL = "saaras:v4"
TTS_MODEL = "bulbul:v4-flash"
TTS_DEFAULT_SPEAKER = "shubh_en_narration"
TTS_REST_MAX_CHARS = 2500
TTS_STREAM_MAX_CHARS = 3500

RETRY_STATUSES = {429, 503}
MAX_RETRIES = 4


class SarvamError(Exception):
    """Normalised error for any failed Sarvam call."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _api_key() -> str:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise SarvamError(500, "missing_api_key",
                          "SARVAM_API_KEY is not configured on the server.")
    return key


def _headers() -> dict[str, str]:
    return {"api-subscription-key": _api_key()}


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=BASE_URL, timeout=httpx.Timeout(180.0, connect=15.0))
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _error_from_response(resp: httpx.Response) -> SarvamError:
    """Map a Sarvam error response to a friendly SarvamError (never leaks the key)."""
    status = resp.status_code
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or err.get("code") or ""
            detail = detail or body.get("detail") or body.get("title") or ""
            if not isinstance(detail, str):
                detail = json.dumps(detail)[:300]
    except ValueError:
        detail = resp.text[:300]

    mapping = {
        400: ("bad_request", "Sarvam rejected the request"),
        402: ("payment_required", "Sarvam billing / credits issue"),
        403: ("invalid_api_key", "Sarvam API key is invalid or lacks access"),
        404: ("not_found", "Sarvam resource not found"),
        409: ("conflict", "Sarvam job is not ready yet"),
        413: ("file_too_large", "File is too large for Sarvam"),
        422: ("unprocessable", "Sarvam could not process the input"),
        429: ("rate_limited", "Sarvam rate limit hit — please retry shortly"),
        503: ("service_unavailable", "Sarvam service is overloaded — please retry"),
    }
    code, base = mapping.get(status, ("upstream_error", f"Sarvam returned HTTP {status}"))
    msg = f"{base}: {detail}" if detail else base
    return SarvamError(status, code, msg)


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Send a request with retry + exponential backoff on 429/503 and network errors."""
    delay = 2.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client().request(method, url, headers=_headers(), **kwargs)
        except httpx.TransportError as exc:
            if attempt == MAX_RETRIES:
                raise SarvamError(502, "network_error", f"Could not reach Sarvam: {exc.__class__.__name__}")
            log.warning("Network error calling %s (attempt %d): %s", url, attempt + 1, exc)
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                raise _error_from_response(resp)
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
            log.warning("Sarvam %s on %s — retrying in %.1fs", resp.status_code, url, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)
    raise SarvamError(500, "unreachable", "Retry loop exited unexpectedly")


# --------------------------------------------------------------------------
# Chat completion (Sarvam-105B)
# --------------------------------------------------------------------------
async def chat(messages: list[dict[str, str]], *, max_tokens: int = 4096,
               reasoning_effort: str | None = "low", temperature: float = 0.2,
               model: str = CHAT_MODEL) -> dict[str, Any]:
    """Returns {content, reasoning, usage, finish_reason}."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Explicit null disables reasoning; omitted would default to "medium".
        "reasoning_effort": reasoning_effort,
    }
    resp = await _request("POST", "/v1/chat/completions", json=payload)
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": msg.get("reasoning_content"),
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }


# --------------------------------------------------------------------------
# Document Intelligence (Sarvam Vision) — async job flow
# --------------------------------------------------------------------------
TERMINAL_STATUSES = {"completed", "partially_completed", "failed", "rejected"}


def _html_to_md(content: str) -> str:
    """Fallback converter if Sarvam outputs HTML instead of pure Markdown."""
    s = content
    s = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: f"\n\n{'#' * int(m.group(1))} {m.group(2).strip()}\n\n", s, flags=re.S | re.I)
    s = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", s, flags=re.S | re.I)
    s = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", s, flags=re.S | re.I)
    s = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", s, flags=re.S | re.I)
    s = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


async def digitise_create(pdf_bytes: bytes, filename: str, language: str = "en-IN", output_format: str = "md") -> str:
    """Create a digitise job. NOTE: param is `language`, not `language_code`."""
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data = {"language": language, "output_format": output_format}
    resp = await _request("POST", "/doc-ai/v1/job/digitise", files=files, data=data)
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise SarvamError(502, "bad_upstream_response", "Digitise did not return a job_id")
    return job_id


async def digitise_status(job_id: str) -> dict[str, Any]:
    resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/status")
    return resp.json()


async def digitise_results_markdown(job_id: str) -> tuple[str, str]:
    """Fetch terminal results as Markdown. Returns (markdown, status).
    Handles both ZIP downloads (standard for file jobs) and JSON responses,
    with automatic fallback from format=md to format=html if the job produced HTML."""
    try:
        resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/results", params={"format": "md"})
    except SarvamError as exc:
        if "cannot serve md" in exc.message.lower() or "html" in exc.message.lower():
            log.warning("Job %s produced HTML; falling back to format=html", job_id)
            resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/results", params={"format": "html"})
        else:
            raise

    content = resp.content
    # 1. ZIP archive response (standard format for file-producing digitise jobs)
    if content.startswith(b"PK\x03\x04") or "zip" in resp.headers.get("content-type", "").lower():
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # Prefer .md files first
                md_files = [n for n in z.namelist() if n.endswith(".md")]
                if md_files:
                    texts = [z.read(n).decode("utf-8", errors="replace").strip() for n in sorted(md_files)]
                    return "\n\n".join(t for t in texts if t), "completed"
                # Fallback to .html files
                html_files = [n for n in z.namelist() if n.endswith(".html")]
                if html_files:
                    texts = [_html_to_md(z.read(n).decode("utf-8", errors="replace")) for n in sorted(html_files)]
                    return "\n\n".join(t for t in texts if t), "completed"
        except zipfile.BadZipFile as e:
            log.error("Corrupted zip returned for job %s: %s", job_id, e)

    # 2. JSON structured response fallback
    try:
        body = resp.json()
        pages: list[tuple[int, str]] = []
        for doc in body.get("documents") or []:
            for i, page in enumerate(doc.get("pages") or []):
                p_text = page.get("content") or ""
                if "<" in p_text and ">" in p_text:
                    p_text = _html_to_md(p_text)
                pages.append((page.get("page_number") or i + 1, p_text))
        pages.sort(key=lambda p: p[0])
        md = "\n\n".join(p[1].strip() for p in pages if p[1].strip())
        return md, body.get("status", "")
    except Exception:
        # 3. Plain text fallback
        return content.decode("utf-8", errors="replace"), "completed"


async def digitise_wait(job_id: str, *, poll_interval: float = 3.0, timeout: float = 600.0,
                        on_progress=None) -> dict[str, Any]:
    """Poll status until terminal or timeout. Calls on_progress(status_dict) each poll."""
    elapsed = 0.0
    while True:
        status = await digitise_status(job_id)
        if on_progress:
            on_progress(status)
        if status.get("status") in TERMINAL_STATUSES:
            return status
        if elapsed >= timeout:
            raise SarvamError(504, "ocr_timeout", f"OCR job {job_id} did not finish within {int(timeout)}s")
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval


# --------------------------------------------------------------------------
# Speech-to-Text (Saaras)
# --------------------------------------------------------------------------
async def transcribe(audio: bytes, filename: str, content_type: str, *,
                     keyterms: list[str] | None = None, mode: str = "codemix",
                     language_code: str | None = None) -> dict[str, Any]:
    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": STT_MODEL, "mode": mode}
    if language_code:
        data["language_code"] = language_code
    if keyterms:
        # Sent as a JSON-encoded array in one form field (max 50 terms, 64 chars each).
        clean = [k[:64] for k in keyterms if k and k.strip()][:50]
        if clean:
            data["keyterms"] = json.dumps(clean)
    resp = await _request("POST", "/speech-to-text", files=files, data=data)
    body = resp.json()
    return {"transcript": body.get("transcript", ""), "language_code": body.get("language_code")}


# --------------------------------------------------------------------------
# Text-to-Speech (Bulbul)
# --------------------------------------------------------------------------
async def tts_rest(text: str, *, language_code: str = "en-IN", speaker: str = TTS_DEFAULT_SPEAKER,
                   codec: str = "mp3", pace: float = 1.0) -> str:
    """Returns base64 audio (single chunk; text must be <= 2500 chars)."""
    payload = {
        "text": text[:TTS_REST_MAX_CHARS],
        "language_code": language_code,
        "speaker": speaker,
        "model": TTS_MODEL,
        "output_audio_codec": codec,
        "pace": pace,
    }
    resp = await _request("POST", "/text-to-speech", json=payload)
    audios = resp.json().get("audios") or []
    return "".join(audios)


async def tts_stream(text: str, *, language_code: str = "en-IN", speaker: str = TTS_DEFAULT_SPEAKER,
                     codec: str = "mp3", pace: float = 1.0) -> AsyncIterator[bytes]:
    """Yield raw audio bytes from the HTTP streaming TTS endpoint."""
    payload = {
        "text": text[:TTS_STREAM_MAX_CHARS],
        "language_code": language_code,
        "speaker": speaker,
        "model": TTS_MODEL,
        "output_audio_codec": codec,
        "pace": pace,
    }
    async with client().stream("POST", "/text-to-speech/stream", json=payload, headers=_headers()) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            raise _error_from_response(resp)
        async for chunk in resp.aiter_bytes():
            if chunk:
                yield chunk
