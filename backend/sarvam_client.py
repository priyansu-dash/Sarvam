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

BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")

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
#
#   POST /doc-ai/v1/job/digitise          -> {job_id, status, run_id}
#   GET  /doc-ai/v1/job/{id}/status       -> {status, usage{pages_*}, ...}
#   GET  /doc-ai/v1/job/{id}/results      -> {documents:[{pages:[{page_number, content}]}], ...}
#   GET  /doc-ai/v1/job/{id}/download-url -> {method, url, headers, expires_at}  (ZIP/file)
#
# Results are only served for terminal jobs (409 before that).
# --------------------------------------------------------------------------
TERMINAL_STATUSES = {"completed", "partially_completed", "failed", "rejected"}

# `format` on /results picks the output representation. Try in order of usefulness and only
# move on when the server says the format is unsupported (400/415/422).
_RESULT_FORMATS: tuple[str | None, ...] = ("md", "json", "html", None)
_FORMAT_REJECTED = {400, 415, 422}

_HTML_HINT = re.compile(r"</?(p|h[1-6]|div|table|tr|td|th|ul|ol|li|br|span|strong|em|b|i|sup|sub)\b", re.I)


def _looks_like_html(s: str) -> bool:
    return bool(_HTML_HINT.search(s))


def _html_to_md(content: str) -> str:
    """Convert the HTML Sarvam Vision emits (headings, paragraphs, lists, tables) to Markdown."""
    s = content
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>",
               lambda m: f"\n\n{'#' * int(m.group(1))} {m.group(2).strip()}\n\n", s, flags=re.S | re.I)
    # Tables: one row per line, cells separated by " | ".
    s = re.sub(r"</t[hd]>\s*", " | ", s, flags=re.I)
    s = re.sub(r"</tr>", "\n", s, flags=re.I)
    s = re.sub(r"</?(table|thead|tbody|tfoot)[^>]*>", "\n", s, flags=re.I)
    s = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", s, flags=re.S | re.I)
    s = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", s, flags=re.S | re.I)
    s = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", s, flags=re.S | re.I)
    s = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r" \| *\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _normalise_text(s: str) -> str:
    s = s or ""
    return _html_to_md(s) if _looks_like_html(s) else s.strip()


def _block_to_md(block: dict[str, Any]) -> str:
    """Render one JSON layout block ({text, tag, bbox}) as Markdown."""
    text = (block.get("text") or block.get("content") or "").strip()
    if not text:
        return ""
    tag = str(block.get("tag") or block.get("type") or "").lower()
    m = re.fullmatch(r"h([1-6])", tag)
    if m:
        return f"{'#' * int(m.group(1))} {text}"
    if tag in {"title", "heading", "section-header", "section_header"}:
        return f"## {text}"
    if tag in {"list-item", "list_item", "li"}:
        return f"- {text}"
    return text


def _page_text(page: dict[str, Any]) -> str:
    content = page.get("content")
    if isinstance(content, str) and content.strip():
        return _normalise_text(content)
    for key in ("markdown", "md", "html", "text"):
        val = page.get(key)
        if isinstance(val, str) and val.strip():
            return _normalise_text(val)
    blocks = page.get("blocks")
    if isinstance(blocks, list):
        parts = [_block_to_md(b) for b in blocks if isinstance(b, dict)]
        return "\n\n".join(p for p in parts if p)
    return ""


def _text_from_json(body: Any) -> str:
    """Pull page text out of a digitise results payload, in page order."""
    if isinstance(body, list):
        return "\n\n".join(t for t in (_text_from_json(b) for b in body) if t)
    if not isinstance(body, dict):
        return ""
    pages: list[tuple[int, str]] = []
    docs = body.get("documents")
    if isinstance(docs, dict):
        docs = [docs]
    for d_idx, doc in enumerate(docs or []):
        for i, page in enumerate((doc or {}).get("pages") or []):
            if not isinstance(page, dict):
                continue
            txt = _page_text(page)
            if txt.strip():
                pages.append((d_idx * 100_000 + int(page.get("page_number") or i + 1), txt))
    if not pages and isinstance(body.get("pages"), list):  # per-page file inside a ZIP
        for i, page in enumerate(body["pages"]):
            if isinstance(page, dict):
                txt = _page_text(page)
                if txt.strip():
                    pages.append((int(page.get("page_number") or i + 1), txt))
    if pages:
        pages.sort(key=lambda p: p[0])
        return "\n\n".join(p[1].strip() for p in pages)
    return _page_text(body)  # single-document payload with top-level content/markdown


def _natural_key(name: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _text_from_zip(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n]
        for ext in (".md", ".html", ".htm", ".json", ".txt"):
            group = sorted((n for n in names if n.lower().endswith(ext)), key=_natural_key)
            if not group:
                continue
            texts: list[str] = []
            for n in group:
                raw = z.read(n).decode("utf-8", errors="replace")
                if ext == ".json":
                    try:
                        texts.append(_text_from_json(json.loads(raw)))
                    except ValueError:
                        continue
                else:
                    texts.append(_normalise_text(raw))
            joined = "\n\n".join(t for t in texts if t.strip())
            if joined.strip():
                return joined
    return ""


def _parse_payload(content: bytes, content_type: str = "") -> tuple[str, str]:
    """Turn any results/download body (JSON, ZIP, Markdown, HTML, text) into (markdown, status)."""
    if content[:4] == b"PK\x03\x04" or "zip" in content_type.lower():
        try:
            return _text_from_zip(content), ""
        except zipfile.BadZipFile as exc:
            log.error("Corrupted ZIP from Sarvam: %s", exc)
            return "", ""
    raw = content.decode("utf-8", errors="replace")
    try:
        body = json.loads(raw)
    except ValueError:
        return _normalise_text(raw), ""  # plain Markdown / HTML body
    status = body.get("status", "") if isinstance(body, dict) else ""
    return _text_from_json(body), status


async def digitise_create(pdf_bytes: bytes, filename: str, language: str = "en-IN") -> str:
    """Create a digitise job. Field is `language` (BCP-47), not `language_code`; omitting it makes
    Sarvam treat the document as Indic. Output format is chosen at /results time."""
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data = {"language": language}
    resp = await _request("POST", "/doc-ai/v1/job/digitise", files=files, data=data)
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise SarvamError(502, "bad_upstream_response", "Digitise did not return a job_id")
    return job_id


async def digitise_status(job_id: str) -> dict[str, Any]:
    resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/status")
    return resp.json()


async def _download_output(job_id: str) -> str:
    """Fallback: get a presigned download URL, fetch it (usually a ZIP), and parse it."""
    resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/download-url")
    info = resp.json()
    url = info.get("url")
    if not url:
        raise SarvamError(502, "bad_upstream_response", "download-url returned no url")
    # Presigned URL: use a plain client so our subscription key is never sent to the storage host.
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0), follow_redirects=True) as dl:
        r = await dl.request((info.get("method") or "GET").upper(), url, headers=info.get("headers") or {})
    if r.status_code >= 400:
        raise SarvamError(502, "download_failed", f"Could not download OCR output (HTTP {r.status_code}).")
    text, _ = _parse_payload(r.content, r.headers.get("content-type", ""))
    return text


async def digitise_results_markdown(job_id: str) -> tuple[str, str]:
    """Fetch a terminal job's text as Markdown. Returns (markdown, status).
    Tries /results with several `format` values (and none); if every attempt is empty,
    falls back to the /download-url ZIP."""
    status = ""
    for fmt in _RESULT_FORMATS:
        try:
            resp = await _request("GET", f"/doc-ai/v1/job/{job_id}/results",
                                  params={"format": fmt} if fmt else None)
        except SarvamError as exc:
            if exc.status in _FORMAT_REJECTED:
                log.warning("Job %s: /results format=%s rejected (%s); trying next", job_id, fmt, exc.message)
                continue
            raise
        text, st = _parse_payload(resp.content, resp.headers.get("content-type", ""))
        status = st or status
        if text.strip():
            log.info("Job %s: parsed %d chars from /results (format=%s)", job_id, len(text), fmt)
            return text, status or "completed"
        log.warning("Job %s: /results (format=%s) held no text; head=%r", job_id, fmt, resp.content[:200])

    log.warning("Job %s: falling back to /download-url", job_id)
    text = await _download_output(job_id)
    return text, status or "completed"


async def digitise_wait(job_id: str, *, poll_interval: float = 3.0, timeout: float = 600.0,
                        on_progress=None) -> dict[str, Any]:
    """Poll status until terminal or timeout. Calls on_progress(status_dict) each poll."""
    elapsed = 0.0
    while True:
        status = await digitise_status(job_id)
        if on_progress:
            on_progress(status)
        if (status.get("status") or "").lower() in TERMINAL_STATUSES:
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
