"""
FastAPI backend for the Sarvam Research Paper Assistant.

Browser -> this backend -> Sarvam API. The API key lives only in the server env.

Routes
  GET  /api/health                 key configured? (never returns the key)
  POST /api/upload                 PDF -> starts chunked OCR in background -> {paper_id,...}
  GET  /api/papers/{paper_id}      OCR progress / result (preview when ready)
  POST /api/summarize              {paper_id | text} -> structured summary
  POST /api/tts                    {text,...} -> audio/mpeg (REST, chunked <=2500 chars)
  POST /api/tts/stream             {text,...} -> {stream_url} (short-lived token)
  GET  /api/tts/stream/{token}     streamed audio/mpeg proxied from Sarvam
  POST /api/transcribe             audio (<=30s) -> {transcript, language_code}
  POST /api/ask                    {paper_id, question} -> grounded {answer}
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure backend directory is in sys.path so sibling imports always resolve
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

import paper_store
import pdf_chunker
import sarvam_client as sarvam
import text_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_AUDIO_MB = 10
MAX_AUDIO_SECONDS = 30

# LLM context budgeting (sarvam-105b = 128K tokens). Leave room for output + reasoning.
SINGLE_PASS_TOKEN_BUDGET = 95_000
SINGLE_PASS_CHAR_BUDGET = int(SINGLE_PASS_TOKEN_BUDGET * text_utils.CHARS_PER_TOKEN)
MAP_CHUNK_CHARS = 90_000
QA_CHAR_BUDGET = SINGLE_PASS_CHAR_BUDGET

# One OCR pipeline at a time: Doc AI is rate-limited to 10 req/min.
OCR_SEMAPHORE = asyncio.Semaphore(1)
OCR_MIN_GAP_SECONDS = 6.0
OCR_POLL_SECONDS = 5.0
OCR_CHUNK_TIMEOUT = 900.0
_bg_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key or key == "your_key_here":
        log.warning("SARVAM_API_KEY is not set — copy .env.example to .env and add your key.")
    yield
    await sarvam.aclose()


app = FastAPI(title="Sarvam Research Assistant", lifespan=lifespan)


# --------------------------------------------------------------------------
# Error handling — one consistent JSON shape, never leaks secrets/stack traces
# --------------------------------------------------------------------------
def error_json(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _client_status(upstream: int) -> int:
    if upstream in (400, 413, 422):
        return 400
    if upstream == 404:
        return 404
    if upstream == 409:
        return 409
    if upstream == 429:
        return 429
    if upstream == 504:
        return 504
    if upstream == 500:
        return 500
    return 502  # 402/403/5xx upstream -> bad gateway


@app.exception_handler(sarvam.SarvamError)
async def _sarvam_error(_: Request, exc: sarvam.SarvamError):
    log.error("Sarvam error %s (%s): %s", exc.status, exc.code, exc.message)
    return error_json(_client_status(exc.status), exc.code, exc.message)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return error_json(exc.status_code, detail.get("code", "error"), detail.get("message", ""))
    return error_json(exc.status_code, "http_error", str(detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return error_json(422, "invalid_request", f"{loc}: {first.get('msg', 'invalid input')}".strip(": "))


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return error_json(500, "internal_error", "Something went wrong on the server.")


def fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def require_ready_paper(paper_id: str) -> paper_store.Paper:
    paper = paper_store.get(paper_id)
    if paper is None:
        raise fail(404, "paper_not_found", "Unknown paper_id. Upload the paper again.")
    if paper.state != "ready":
        raise fail(409, "paper_not_ready", f"Paper is not ready yet (state: {paper.state}).")
    return paper


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    key = os.getenv("SARVAM_API_KEY", "").strip()
    return {
        "ok": True,
        "key_configured": bool(key and key != "your_key_here"),
        "models": {"chat": sarvam.CHAT_MODEL, "stt": sarvam.STT_MODEL, "tts": sarvam.TTS_MODEL},
    }


# --------------------------------------------------------------------------
# Phase 2 — Upload + chunked OCR (Document Intelligence)
# --------------------------------------------------------------------------
@app.post("/api/upload", status_code=202)
async def upload(file: UploadFile = File(...)):
    name = file.filename or "paper.pdf"
    if not (name.lower().endswith(".pdf") or file.content_type == "application/pdf"):
        raise fail(400, "unsupported_file", "Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise fail(400, "empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise fail(413, "file_too_large", f"PDF exceeds the {MAX_UPLOAD_MB} MB limit.")
    if not data.startswith(b"%PDF"):
        raise fail(400, "unsupported_file", "That file does not look like a valid PDF.")

    try:
        chunks = pdf_chunker.split_pdf(data)
    except pdf_chunker.PdfChunkError as exc:
        raise fail(400, "bad_pdf", str(exc))

    paper = paper_store.create(name)
    paper.page_count = sum(c[2] for c in chunks)
    paper.chunk_count = len(chunks)
    paper.stage = "Waiting for OCR slot"
    task = asyncio.create_task(_run_ocr(paper, chunks))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    log.info("Paper %s: %d pages -> %d chunk(s)", paper.paper_id, paper.page_count, paper.chunk_count)
    return paper.public()


@app.get("/api/papers/{paper_id}")
async def paper_status(paper_id: str):
    paper = paper_store.get(paper_id)
    if paper is None:
        raise fail(404, "paper_not_found", "Unknown paper_id.")
    return paper.public()


async def _run_ocr(paper: paper_store.Paper, chunks: list[tuple[bytes, int, int]]) -> None:
    """Sequentially OCR each <=10-page chunk and re-assemble Markdown in page order."""
    async with OCR_SEMAPHORE:
        paper.state = "processing"
        parts: list[tuple[int, str]] = []
        failed_chunks = 0
        last_create = 0.0
        try:
            for idx, (chunk_bytes, start, count) in enumerate(chunks):
                first, last = start + 1, start + count
                label = f"pages {first}–{last}"

                # Space out job creation to respect the 10 req/min limit.
                gap = OCR_MIN_GAP_SECONDS - (time.monotonic() - last_create)
                if gap > 0 and idx > 0:
                    await asyncio.sleep(gap)
                last_create = time.monotonic()

                paper.stage = f"Uploading chunk {idx + 1}/{len(chunks)} ({label})"
                fname = f"{Path(paper.filename).stem}_p{first}-{last}.pdf"
                job_id = await sarvam.digitise_create(chunk_bytes, fname, language="en-IN")

                def progress(st: dict[str, Any], i=idx, lbl=label):
                    u = st.get("usage") or {}
                    done, total = u.get("pages_processed"), u.get("pages_total")
                    extra = f" · {done}/{total} pages" if total else ""
                    paper.stage = f"OCR chunk {i + 1}/{len(chunks)} ({lbl}) · {st.get('status', '…')}{extra}"

                status = await sarvam.digitise_wait(job_id, poll_interval=OCR_POLL_SECONDS,
                                                    timeout=OCR_CHUNK_TIMEOUT, on_progress=progress)
                st = (status.get("status") or "").lower()
                if st in ("failed", "rejected"):
                    failed_chunks += 1
                    paper.warnings.append(f"OCR {st} for {label}; those pages are missing.")
                    parts.append((start, f"\n\n> [Pages {first}–{last} could not be extracted]\n\n"))
                    log.warning("Paper %s chunk %d %s", paper.paper_id, idx, st)
                else:
                    md, _ = await sarvam.digitise_results_markdown(job_id)
                    if st == "partially_completed":
                        u = status.get("usage") or {}
                        paper.warnings.append(
                            f"OCR partially completed for {label} "
                            f"({u.get('pages_failed', '?')} page(s) failed).")
                        log.warning("Paper %s chunk %d partially completed", paper.paper_id, idx)
                    parts.append((start, md))
                paper.chunks_done = idx + 1

            if failed_chunks == len(chunks):
                raise sarvam.SarvamError(502, "ocr_failed", "OCR failed for every page of the document.")

            parts.sort(key=lambda p: p[0])  # ascending page order
            paper.text = "\n\n".join(p[1].strip() for p in parts if p[1].strip())
            if not paper.text.strip():
                raise sarvam.SarvamError(502, "ocr_empty", "OCR finished but returned no text.")

            paper.keyterms = text_utils.extract_keyterms(paper.text)
            paper.state = "ready"
            paper.stage = "Ready"
            paper_store.save(paper)
            log.info("Paper %s ready: %d chars", paper.paper_id, len(paper.text))
        except sarvam.SarvamError as exc:
            paper.state, paper.stage = "failed", "Failed"
            paper.error = {"code": exc.code, "message": exc.message}
            log.error("Paper %s OCR failed: %s", paper.paper_id, exc.message)
        except Exception as exc:  # noqa: BLE001
            paper.state, paper.stage = "failed", "Failed"
            paper.error = {"code": "internal_error", "message": "OCR pipeline crashed on the server."}
            log.exception("Paper %s OCR crashed: %s", paper.paper_id, exc)


# --------------------------------------------------------------------------
# Phase 1 — Summarisation (Sarvam-105B), map-reduce for long papers
# --------------------------------------------------------------------------
SUMMARY_SYSTEM = (
    "You are an expert research explainer. You read academic papers and explain them clearly "
    "to a technically literate reader. Use only information present in the provided paper text; "
    "never invent results, numbers, or citations. If something is not stated, say so."
)

SUMMARY_FORMAT = """Produce a structured Markdown summary with exactly these sections:

## TL;DR
Two or three sentences capturing the paper's core contribution.

## Problem
What problem is addressed and why it matters.

## Method
How the authors approach it — architecture, algorithm, data, or experimental setup.

## Key Concepts
A bulleted list of the important concepts/terms, each with a one-line plain-English explanation.

## Findings
The main results, including key numbers where the paper reports them.

## Limitations & Open Questions
Stated limitations, plus any obvious caveats.

Be precise and concise. Do not include a preamble before the first heading."""


class SummarizeIn(BaseModel):
    paper_id: str | None = None
    text: str | None = Field(default=None, max_length=2_000_000)
    force: bool = False


@app.post("/api/summarize")
async def summarize(body: SummarizeIn):
    paper = None
    if body.paper_id:
        paper = require_ready_paper(body.paper_id)
        if paper.summary and not body.force:
            return {"summary": paper.summary, "cached": True, "strategy": "cached"}
        text = paper.text
    elif body.text and body.text.strip():
        text = body.text
    else:
        raise fail(400, "missing_input", "Provide either paper_id or text.")

    if len(text) <= SINGLE_PASS_CHAR_BUDGET:
        strategy = "single-pass"
        summary = await _summarize_once(text)
    else:
        strategy = "map-reduce"
        summary = await _summarize_map_reduce(text)

    if paper:
        paper.summary = summary
        paper_store.save(paper)
    return {"summary": summary, "cached": False, "strategy": strategy}


async def _summarize_once(text: str) -> str:
    res = await sarvam.chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"PAPER TEXT:\n\"\"\"\n{text}\n\"\"\"\n\n{SUMMARY_FORMAT}"},
        ],
        max_tokens=8000,
        reasoning_effort="low",
    )
    return _ensure_content(res, "summary")


async def _summarize_map_reduce(text: str) -> str:
    parts = text_utils.split_text(text, MAP_CHUNK_CHARS)
    log.info("Map-reduce summary over %d parts", len(parts))
    notes: list[str] = []
    for i, part in enumerate(parts, 1):
        res = await sarvam.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content":
                    f"This is part {i} of {len(parts)} of a research paper.\n\nPART TEXT:\n\"\"\"\n{part}\n\"\"\"\n\n"
                    "Write detailed bullet-point notes on this part: problem statements, methods, definitions "
                    "of key concepts, reported results with numbers, and stated limitations. Notes only."},
            ],
            max_tokens=3000,
            reasoning_effort=None,
        )
        notes.append(f"### Notes on part {i}\n{_ensure_content(res, 'notes')}")
    combined = "\n\n".join(notes)
    res = await sarvam.chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content":
                "Below are section-by-section notes covering an entire research paper.\n\n"
                f"{combined}\n\n{SUMMARY_FORMAT}"},
        ],
        max_tokens=8000,
        reasoning_effort="low",
    )
    return _ensure_content(res, "summary")


def _ensure_content(res: dict[str, Any], what: str) -> str:
    content = res.get("content") or ""
    if not content:
        # Reasoning tokens can exhaust max_tokens and leave empty content (§7.5).
        raise sarvam.SarvamError(502, "empty_completion",
                                 f"The model returned no {what} (finish_reason={res.get('finish_reason')}). Try again.")
    if res.get("finish_reason") == "length":
        content += "\n\n*(Output truncated at the token limit.)*"
    return content


# --------------------------------------------------------------------------
# Phase 3 — Text-to-Speech
# --------------------------------------------------------------------------
class TTSIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=60_000)
    language_code: str = "en-IN"
    speaker: str = sarvam.TTS_DEFAULT_SPEAKER
    pace: float = Field(default=1.0, ge=0.5, le=2.0)


@app.post("/api/tts")
async def tts(body: TTSIn):
    """REST TTS. Long text is split at sentence boundaries into <=2500-char calls; the decoded
    MP3 segments are concatenated (MP3 frames concatenate cleanly) and returned as audio/mpeg."""
    prose = text_utils.strip_markdown(body.text)
    if not prose:
        raise fail(400, "empty_text", "Nothing to speak after cleaning the text.")
    pieces = text_utils.split_for_tts(prose, max_chars=sarvam.TTS_REST_MAX_CHARS - 100)

    sem = asyncio.Semaphore(3)

    async def one(piece: str) -> bytes:
        async with sem:
            b64 = await sarvam.tts_rest(piece, language_code=body.language_code,
                                        speaker=body.speaker, codec="mp3", pace=body.pace)
            return base64.b64decode(b64)  # REST returns base64 — must decode (§7.6)

    segments = await asyncio.gather(*(one(p) for p in pieces))
    audio = b"".join(segments)
    return Response(content=audio, media_type="audio/mpeg",
                    headers={"X-TTS-Segments": str(len(pieces)), "Cache-Control": "no-store"})


# Short-lived tokens so a plain <audio src> GET can stream progressively.
_stream_tokens: dict[str, tuple[float, dict[str, Any]]] = {}
STREAM_TOKEN_TTL = 600


@app.post("/api/tts/stream")
async def tts_stream_create(body: TTSIn):
    prose = text_utils.strip_markdown(body.text)
    if not prose:
        raise fail(400, "empty_text", "Nothing to speak after cleaning the text.")
    now = time.time()
    for k in [k for k, (ts, _) in _stream_tokens.items() if now - ts > STREAM_TOKEN_TTL]:
        _stream_tokens.pop(k, None)
    token = secrets.token_urlsafe(16)
    _stream_tokens[token] = (now, {
        "text": prose[: sarvam.TTS_STREAM_MAX_CHARS],
        "language_code": body.language_code,
        "speaker": body.speaker,
        "pace": body.pace,
    })
    return {"stream_url": f"/api/tts/stream/{token}", "truncated": len(prose) > sarvam.TTS_STREAM_MAX_CHARS}


@app.get("/api/tts/stream/{token}")
async def tts_stream_play(token: str):
    entry = _stream_tokens.get(token)
    if not entry or time.time() - entry[0] > STREAM_TOKEN_TTL:
        raise fail(404, "stream_expired", "Audio stream link expired.")
    params = entry[1]
    gen = sarvam.tts_stream(params["text"], language_code=params["language_code"],
                            speaker=params["speaker"], pace=params["pace"], codec="mp3")
    # Pull the first chunk eagerly so upstream errors become proper JSON errors.
    try:
        first = await gen.__anext__()
    except StopAsyncIteration:
        raise fail(502, "empty_audio", "Sarvam returned no audio.")

    async def body():
        yield first
        async for chunk in gen:
            yield chunk

    return StreamingResponse(body(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# Phase 4 — Speech-to-Text
# --------------------------------------------------------------------------
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), paper_id: str | None = Form(default=None)):
    audio = await file.read()
    if not audio:
        raise fail(400, "empty_audio", "No audio received.")
    if len(audio) > MAX_AUDIO_MB * 1024 * 1024:
        raise fail(413, "audio_too_large", "Audio file is too large.")

    ctype = file.content_type or "audio/wav"
    duration = _wav_duration(audio)
    if duration is not None and duration > MAX_AUDIO_SECONDS + 0.5:
        raise fail(400, "audio_too_long",
                   f"Recording is {duration:.0f}s; the limit is {MAX_AUDIO_SECONDS}s. Ask a shorter question.")

    keyterms: list[str] = []
    if paper_id:
        paper = paper_store.get(paper_id)
        if paper and paper.state == "ready":
            keyterms = paper.keyterms

    res = await sarvam.transcribe(audio, file.filename or "question.wav", ctype, keyterms=keyterms)
    if not res["transcript"].strip():
        raise fail(422, "no_speech", "Couldn't hear a question — try speaking closer to the mic.")
    return {**res, "duration": duration}


def _wav_duration(data: bytes) -> float | None:
    """Duration of a PCM WAV from its header, or None if not a parseable WAV."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    try:
        pos = 12
        byte_rate = None
        while pos + 8 <= len(data):
            cid = data[pos:pos + 4]
            size = int.from_bytes(data[pos + 4:pos + 8], "little")
            if cid == b"fmt ":
                byte_rate = int.from_bytes(data[pos + 16:pos + 20], "little")
            elif cid == b"data" and byte_rate:
                return min(size, len(data) - pos - 8) / byte_rate
            pos += 8 + size + (size & 1)
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------
# Phase 5 — Grounded Q&A
# --------------------------------------------------------------------------
QA_SYSTEM = (
    "You are a voice tutor helping a student test their understanding of ONE research paper. "
    "Answer strictly from the PAPER TEXT provided. If the paper does not cover the question, say "
    "plainly that the paper doesn't address it, and optionally mention what it does cover that is related. "
    "Your answer will be read aloud, so: reply in 2 to 5 short, natural spoken sentences; no Markdown, "
    "no bullet points, no tables, no LaTeX; spell out symbols in words. "
    "If the student states their own understanding, tell them whether it is correct and gently fix mistakes."
)


class AskIn(BaseModel):
    paper_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    reset_history: bool = False


@app.post("/api/ask")
async def ask(body: AskIn):
    paper = require_ready_paper(body.paper_id)
    if body.reset_history:
        paper.history.clear()

    context = text_utils.select_relevant(paper.text, body.question, QA_CHAR_BUDGET)
    partial = len(context) < len(paper.text)

    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content":
            f"PAPER TEXT{' (most relevant excerpts)' if partial else ''}:\n\"\"\"\n{context}\n\"\"\""},
        {"role": "assistant", "content": "I've read the paper. Ask me anything about it."},
    ]
    messages += paper.history[-8:]  # short memory so follow-ups work
    messages.append({"role": "user", "content": body.question.strip()})

    res = await sarvam.chat(messages, max_tokens=2500, reasoning_effort="low", temperature=0.3)
    answer = _ensure_content(res, "answer")

    paper.history += [{"role": "user", "content": body.question.strip()},
                      {"role": "assistant", "content": answer}]
    paper.history = paper.history[-16:]
    return {"answer": answer, "used_excerpts": partial}


@app.post("/api/papers/{paper_id}/reset")
async def reset_history(paper_id: str):
    paper = require_ready_paper(paper_id)
    paper.history.clear()
    return {"ok": True}


# --------------------------------------------------------------------------
# Frontend (served from the same origin -> no CORS, key never in the browser)
# --------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
