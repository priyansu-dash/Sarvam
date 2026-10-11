# DECISIONS.md — Architectural and Implementation Decisions

This document records key design decisions and trade-offs made while building the Sarvam Research Paper Assistant.

---

## 1. REST Client via HTTPX vs. Sarvam Python SDK

- **Decision**: Implemented backend communication using asynchronous `httpx` with connection pooling rather than exclusively relying on the `sarvamai` SDK wrapper.
- **Rationale**:
  - The Sarvam live API contracts (notably `POST /doc-ai/v1/job/digitise`, `GET /doc-ai/v1/job/{id}/results?format=md`, and `POST /text-to-speech/stream`) have precise parameter expectations (`language="en-IN"`, `output_format="md"`, model IDs `bulbul:v4-flash` vs `bulbul:v3`).
  - Native `httpx` client allows direct control over chunked HTTP streaming (`POST /text-to-speech/stream`), granular status code mapping, and configurable exponential backoff retry loops on HTTP 429 (rate limits) and 503 (service overloaded).

---

## 2. In-Memory PDF Chunking for Document Intelligence

- **Decision**: Created `backend/pdf_chunker.py` using `pypdf` to split uploaded PDFs into sequential chunks of $\le 10$ pages entirely in memory (`io.BytesIO`).
- **Rationale**:
  - Sarvam Doc AI enforces a strict 10-page maximum per submission (`400 invalid_request_error`).
  - Academic research papers routinely run from 12 to 50+ pages.
  - Processing chunks in memory avoids temporary file litter while retaining page indexes (`start_page_index`, `page_count`) to ensure extracted Markdown is stitched together in strict ascending page order.

---

## 3. Rate-Limiting and Async OCR Pipeline

- **Decision**:
  - `POST /api/upload` parses and splits the PDF, creates a paper entry, spawns a background asyncio task, and immediately returns HTTP 202 with `paper_id`.
  - Added an `asyncio.Semaphore(1)` and an enforced minimum delay of 6.0 seconds between Document AI job submissions.
  - The client polls `GET /api/papers/{paper_id}` every 2 seconds for chunk-by-chunk progress.
- **Rationale**:
  - Sarvam Doc AI is rate-limited to 10 requests per minute.
  - Document OCR is asynchronous; polling client-side prevents blocking HTTP request threads or tripping proxy/browser timeouts on long papers.

---

## 4. Dual TTS Strategy: Segmented REST vs. Streamed Audio

- **Decision**:
  - **Summary Read-Aloud**: Uses `POST /api/tts` (REST). Text is stripped of markdown/code, divided into sentence-bounded chunks $\le 2,400$ characters, synthesized via `bulbul:v4-flash` with bounded parallelism (semaphore of 3), base64-decoded, and joined into a single valid MP3 stream.
  - **Voice Q&A Spoken Answers**: Uses `POST /api/tts/stream` to issue a short-lived token ($TTL = 600s$), and proxies chunked audio via `GET /api/tts/stream/{token}` directly from Sarvam's streaming endpoint.
- **Rationale**:
  - Summaries are long ($>2,500$ characters) and benefit from full caching and seeking controls (`<audio controls>`).
  - Conversational Q&A answers need minimal time-to-first-byte latency; streaming delivers speech playback within milliseconds of LLM response completion.

---

## 5. Speech-to-Text Push-to-Talk and Audio Conditioning

- **Decision**:
  - Audio capture in the browser records via `MediaRecorder`, but on release decodes the audio and resamples it to 16 kHz mono 16-bit PCM WAV using `AudioContext` and `OfflineAudioContext`.
  - Enforced a client-side 30-second ring countdown timer with automatic stop.
  - Extracted technical terms, acronyms, and CamelCase identifiers from the paper are passed as `keyterms` (up to 50 terms) to `saaras:v4`.
- **Rationale**:
  - 16 kHz 16-bit PCM mono is Sarvam's recommended format for peak transcription accuracy.
  - Sync REST STT enforces a hard 30-second limit; client-side visual timer and auto-cutoff prevent failed uploads.
  - Technical terms in research papers (e.g., *FlashAttention*, *RoPE*, *BLEU*) are frequently misheard unless biased via `keyterms`.

---

## 6. Token Budgeting & Long Document Handling

- **Decision**:
  - For papers within the context budget ($<95,000$ tokens / $\approx 304,000$ chars), summarization runs in a single prompt with `max_tokens=8000` and `reasoning_effort="low"`.
  - For longer documents, a map-reduce pipeline splits the paper into 90,000-char blocks, extracts structured notes, and compiles the final summary.
  - For Q&A, `text_utils.select_relevant` scores chunks lexically (preserving the abstract/intro chunk) to keep the prompt well within context limits.
- **Rationale**:
  - Reasoning tokens in `sarvam-105b` count against `max_tokens` and are billed. Setting generous limits and lowering `reasoning_effort` prevents truncated or empty responses while ensuring deep comprehension.

---

## 7. Storage and Persistence

- **Decision**: Extracted papers are stored in-memory in `backend/paper_store.py` with an on-disk JSON cache at `data/papers/{paper_id}.json`.
- **Rationale**:
  - Preserves extracted text across server restarts and allows browser page reloads to resume paper sessions without re-running expensive OCR jobs.
  - Chat history remains session-only to keep paper files clean and scoped.

---

## 8. Security and Deployment

- **Decision**:
  - The API key exists only in `.env` (gitignored). Frontend never accesses or receives the key.
  - The FastAPI server serves frontend static assets directly from `/`, eliminating CORS issues and cross-origin complexity.
