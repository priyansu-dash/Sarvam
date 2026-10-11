# AGENT_CONTEXT.md — Sarvam-Powered Research Paper Assistant

> **Read this file fully before writing any code.**
> This is the single source of truth for building the project. Follow the phases in order.
> When something is ambiguous, make the smallest reasonable decision, note it in a `DECISIONS.md`, and move on.

---

## 1. Project Goal

Build a small web application with **two core features**, both powered by the **Sarvam AI API stack**:

1. **Research Paper Summarizer**
   - User uploads a research paper (PDF, or images of pages).
   - OCR/Document Intelligence extracts the full text.
   - Sarvam LLM reads the extracted text and produces a structured summary of the paper's concepts.
   - The summary can be **read aloud** using Sarvam Text-to-Speech.

2. **Voice-Enabled Q&A Bot for Concept Testing**
   - User **speaks** a question about the paper.
   - Speech-to-Text transcribes the spoken question.
   - Sarvam LLM answers, grounded in the paper's extracted text (the paper is the "context").
   - The answer is **spoken back** to the user with Text-to-Speech.
   - This is a conversational loop: listen → think → speak.

The **Sarvam LLM (Sarvam-105B) is the reasoning core** of the whole system. OCR feeds it, and TTS speaks its output.

---

## 2. CRITICAL ARCHITECTURE CONSTRAINT (read first)

**Do NOT call the Sarvam API directly from browser JavaScript.**

Two reasons, both hard blockers:

1. **Secret exposure.** Every Sarvam call needs the header `api-subscription-key: <YOUR_KEY>`. If the frontend makes the call, the key ships to every visitor's browser and can be stolen. Never do this.
2. **CORS / server-side design.** The API is designed for server-side use.

**Therefore the architecture is: browser → your backend → Sarvam API → your backend → browser.**

```
┌─────────────────────┐        ┌──────────────────────────┐        ┌────────────────────┐
│  Frontend (browser) │  HTTP  │  Backend (server)        │ HTTPS  │  Sarvam AI API     │
│  - upload PDF       │ ─────► │  - holds API key         │ ─────► │  api.sarvam.ai     │
│  - record audio     │        │  - orchestrates calls    │        │  - Doc AI (OCR)    │
│  - play audio       │ ◄───── │  - keeps paper text      │ ◄───── │  - Chat (LLM)      │
│  - show summary     │        │    as session context    │        │  - STT / TTS       │
└─────────────────────┘        └──────────────────────────┘        └────────────────────┘
```

The API key lives **only** in a backend environment variable. The frontend never sees it.

---

## 3. Recommended Tech Stack

Use these defaults unless there is a strong reason not to. Keep it simple.

| Layer | Choice | Why |
|---|---|---|
| Backend | **Python 3.11+ with FastAPI + Uvicorn** | Sarvam's official Python SDK (`sarvamai`) is first-class; async fits I/O-heavy API calls |
| Sarvam SDK | `sarvamai` (Python) | Official SDK; wraps chat, STT, TTS, Doc AI |
| Frontend | **Plain HTML + vanilla JS + CSS** (single page), OR **Next.js** if you want a framework | Keep it small; no build step needed for the vanilla version |
| Audio capture | Browser `MediaRecorder` API | Records mic audio to a Blob, no libraries needed |
| Audio playback | HTML `<audio>` element playing a data URL / blob URL | Simple |
| HTTP client (backend) | SDK where available, `httpx`/`requests` for raw REST | — |
| Config | `python-dotenv` reading a `.env` file | Keeps secrets out of code |

> If you prefer Node/TypeScript end-to-end, use the `sarvamai` npm package (`npm install sarvamai@latest`) and Express/Fastify. The API contract is identical. Pick ONE stack and stay consistent.

---

## 4. Sarvam API Reference (the exact contracts to code against)

**Base URL:** `https://api.sarvam.ai`
**Auth:** header `api-subscription-key: <YOUR_SARVAM_API_KEY>`
(The Chat endpoint *also* accepts `Authorization: Bearer <key>` for OpenAI-compatible tooling; prefer `api-subscription-key` everywhere for consistency.)
**Get a key:** create one in the Sarvam AI dashboard.

> **Tip for the agent:** Sarvam publishes a live docs MCP server at `https://docs.sarvam.ai/_mcp/server` and a full docs dump at `https://docs.sarvam.ai/llms-full.txt`. Connect the MCP server (or read the txt) to verify any endpoint detail before implementing it. The docs are the ground truth if anything here disagrees.

### 4.1 Chat Completion — the reasoning core (Sarvam-105B)

- **Endpoint:** `POST https://api.sarvam.ai/v1/chat/completions`
- **Models:** `sarvam-105b` (128K context — use this for paper summarization and Q&A), `sarvam-105b-conversations` (32K context — a lighter conversational variant, good for the voice loop).
- **Request body:** `messages` (list of `{role, content}`), `model` (required), plus optional `max_tokens` (default **2048** — raise it for long summaries), `temperature`, `n`, `response_format`.
- **Reasoning:** Sarvam-105B has hybrid reasoning. `reasoning_effort` accepts `low` / `medium` / `high`; thinking is on by default at `low`. Reasoning tokens are returned in `choices[0].message.reasoning_content` and **count toward completion tokens (they are billed)**. Disable with `reasoning_effort=None` when you want speed/cost over depth.
- **Response:** `choices[0].message.content` holds the answer; `usage` holds token counts.

```python
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

response = client.chat.completions(
    model="sarvam-105b",
    messages=[
        {"role": "system", "content": "You are a research assistant. Answer strictly from the provided paper."},
        {"role": "user", "content": f"PAPER TEXT:\n{paper_text}\n\nSummarise the key concepts."},
    ],
    max_tokens=2000,
)
print(response.choices[0].message.content)
```

cURL equivalent:

```bash
curl -X POST https://api.sarvam.ai/v1/chat/completions \
  -H "api-subscription-key: $SARVAM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sarvam-105b","messages":[{"role":"user","content":"What is the capital of India?"}]}'
```

> There is also `POST /v2/chat/completions` serving open-weight models (`glm5.3`, `gemma4`, `deepseekv4.1-flash`). **This is in beta and requires per-key whitelisting.** Do not depend on it for the MVP; use `/v1` with `sarvam-105b`.

### 4.2 Document Intelligence — OCR / paper reading (Sarvam Vision)

This is the OCR layer. It is an **async, job-based** flow: create → poll → download.

- **Digitise (full-document OCR):** `POST https://api.sarvam.ai/doc-ai/v1/job/digitise`
  - multipart form: `file` (PDF/PNG/JPG/ZIP), `language` (e.g. `en-IN`). Digitise takes `file` + `language` (not `output_format`).
- **Poll status:** `GET https://api.sarvam.ai/doc-ai/v1/job/{job_id}/status` — wait for `completed` / `partially_completed`.
- **Read output:** `GET https://api.sarvam.ai/doc-ai/v1/job/{job_id}/results` — returns JSON `documents[].pages[].content` (try formats md, json, html, or none).
- **Fetch download fallback:** `GET https://api.sarvam.ai/doc-ai/v1/job/{job_id}/download-url` — presigned ZIP download if results endpoint returns no text.

```python
with open("paper.pdf", "rb") as f:
    job = client.doc_ai.digitise(
        file=[("paper.pdf", f, "application/pdf")],
        language="en-IN",
    )

status = client.doc_ai.get_status(job_id=job.job_id)   # poll until completed
# read output from GET /results (try md/json/html/none), falling back to GET /download-url (presigned ZIP)
```

**Hard limits you MUST handle:**
- **Max 10 pages per PDF** (`400 invalid_request_error` if exceeded). Research papers are often longer, so **always split the PDF into ≤10-page chunks before calling Digitise** — see §7.1 for the required strategy.
- **Max 200 MB per file.**
- **Rate limit: 10 requests/minute.**
- Parameter names matter: use `language` (NOT `language_code` — that is ignored and defaults to Hindi). Default language when omitted is Hindi, so **always pass `language="en-IN"` for English papers.**
- Digitise takes `file` + `language`; read output from `GET /results` (JSON `documents[].pages[].content`, try md/json/html/none), falling back to `GET /download-url` (presigned ZIP).

> **Legacy note:** an older flow exists under `POST /doc-digitization/job/v1` (create → upload-files → start → status → download-files). Use the newer `/doc-ai/v1` `digitise` flow. If the SDK's `doc_ai` group is unavailable in the installed version, fall back to the legacy REST flow.

### 4.3 Speech-to-Text — capturing the user's voice question (Saaras)

- **Endpoint:** `POST https://api.sarvam.ai/speech-to-text` (multipart form)
- **Models:** `saaras:v4` (default, recommended; adds keyterm prompting), `saaras:v3` (stable).
- **`mode`:** `transcribe` (default, original language), `translate` (to English), `verbatim`, `translit` (romanised), `codemix` (keeps English words in English — **best for natural chat transcripts**).
- **`keyterms`:** optional list (up to 50 terms, 64 chars each) to bias recognition toward names/technical terms — **supported on `saaras:v4` only.** Pass paper-specific terms here to improve accuracy.
- **`language_code`:** optional; auto-detected if omitted.
- **Response:** `transcript`, `language_code`, `request_id`.

```python
response = client.speech_to_text.transcribe(
    file=open("question.wav", "rb"),
    model="saaras:v4",
    mode="codemix",
    keyterms=["transformer", "attention", "BLEU"],   # terms from the paper
)
print(response.transcript)
```

**Limits:** the **synchronous REST endpoint accepts audio up to 30 seconds.** For voice Q&A, use push-to-talk and record short utterances (<30s). For longer audio, use the Batch API or split into ≤30s chunks. Prefer **mono, 16 kHz, WAV** (16-bit PCM) for best accuracy; 8 kHz telephony is fine too.

### 4.4 Text-to-Speech — speaking the summary and the answers (Bulbul)

Three transports — choose by latency need:

| Transport | Endpoint | Returns | Best for |
|---|---|---|---|
| REST | `POST /text-to-speech` | JSON, base64 audio in `audios[]` | Offline files, pre-generated summaries |
| HTTP stream | `POST /text-to-speech/stream` | chunked audio bytes | Playable before completion — **use for the voice loop** |
| WebSocket | `wss://api.sarvam.ai/text-to-speech/ws` | `{"type":"audio"}` messages | High-volume interactive agents |

- **Models:** `bulbul:v4-flash` (latest, 200+ persona voices, recommended) or `bulbul:v3` (stable, 30+ classic voices; the API default if `model` is omitted).
- **`speaker`:** a voice ID. For `bulbul:v3` use short names like `shubh`, `priya`, `ritu`; for `bulbul:v4-flash` use persona IDs like `simran_en_customer`, `ritu_hi_customer`. **Case-sensitive, lowercase.** Default if omitted on v4-flash is `shubh_enhi_ads`.
- **`language_code`:** e.g. `en-IN`, `hi-IN`. **Required on REST and HTTP streaming.**
- **Other params:** `speech_sample_rate` (REST allows up to 48000; streaming capped at 24000), `output_audio_codec` (`wav`, `mp3`, `flac`, `aac`), `pace` (0.5–2.0), `enable_preprocessing` (normalise numbers/dates — recommended).
- **Limits:** max **2,500 characters** per REST request. No SSML support. **Romanised Indic input degrades quality** — use native script for Indic words.

```python
# REST: returns base64 audio, decode before writing
audio = client.text_to_speech.convert(
    text="Here is the summary of the paper.",
    language_code="en-IN",
    model="bulbul:v4-flash",
    speaker="simran_en_customer",
)
# audio is base64; decode to bytes then write .wav

# Streaming: iterate chunks and pipe to the response for low latency
for chunk in client.text_to_speech.convert_stream(
    text="Here is the summary...",
    language_code="en-IN",
    speaker="shubh",
    model="bulbul:v3",
    output_audio_codec="mp3",
):
    ...
```

> **Important:** the REST response is **base64-encoded**. In JS, `Buffer.from(response.audios.join(""), "base64")` before writing; writing the raw string produces a corrupted file.

---

## 5. Repository Structure (target)

```
sarvam-research-assistant/
├── AGENT_CONTEXT.md          # this file
├── DECISIONS.md              # log of choices you make
├── README.md                 # setup + run instructions
├── .env.example              # SARVAM_API_KEY=your_key_here   (NEVER commit .env)
├── .gitignore                # must include .env
├── backend/
│   ├── main.py               # FastAPI app + routes
│   ├── sarvam_client.py      # thin wrapper around the sarvamai SDK
│   ├── pdf_chunker.py        # splits a PDF into <=10-page chunks (pypdf)
│   ├── paper_store.py        # in-memory (or sqlite) store: paper_id -> extracted text
│   └── requirements.txt
└── frontend/
    ├── index.html            # single page: upload + summary + voice Q&A
    ├── app.js                # fetch calls to backend, MediaRecorder, audio playback
    └── styles.css
```

---

## 6. Build Plan — Phases and Tasks

Do these in order. Each phase should end in something you can run and see working.

### Phase 0 — Scaffold & secrets
- [ ] Create the repo structure above.
- [ ] `pip install sarvamai fastapi uvicorn python-dotenv httpx pypdf`
- [ ] Create `.env.example` with `SARVAM_API_KEY=your_key_here`; add `.env` to `.gitignore`.
- [ ] `sarvam_client.py`: read key from env, instantiate `SarvamAI(api_subscription_key=...)` once, expose helper functions.
- [ ] **Smoke test:** a script that calls Chat Completion with a hardcoded question and prints the reply. This proves auth works before anything else.

### Phase 1 — Backend: chat + summarization
- [ ] Route `POST /api/summarize` accepting `{ paper_id }` or raw text; builds a prompt and calls Chat Completion with `model="sarvam-105b"`.
- [ ] Prompt design: a `system` message setting the assistant as a research explainer; a `user` message containing the paper text (truncate/summarise in chunks if it exceeds context — see §7) and an instruction to produce a structured summary (problem, method, key concepts, findings, limitations).
- [ ] Set `max_tokens` generously (e.g. 2000+) — the default 2048 can truncate.
- [ ] Return `{ summary, reasoning (optional) }` to the frontend.

### Phase 2 — Backend: OCR ingestion (Document Intelligence)
- [ ] Route `POST /api/upload` accepting a PDF (multipart).
- [ ] **Split the PDF into ≤10-page chunks first** (`pdf_chunker.py`, using `pypdf`). This is mandatory — the API rejects any file over 10 pages. See §7.1.
- [ ] For each chunk: call `doc_ai.digitise(...)` with `file` and `language="en-IN"`, poll `get_status` until terminal, read output from `GET /results` (trying md/json/html/none), falling back to `GET /download-url` (presigned ZIP).
- [ ] Process chunks **sequentially** (the endpoint is rate-limited to 10 requests/minute) and **concatenate the Markdown in original page order** into one document.
- [ ] Store the concatenated text keyed by a generated `paper_id` (start with an in-memory dict; note it resets on restart).
- [ ] Return `{ paper_id, page_count, chunk_count, char_count, preview }`.

### Phase 3 — Backend: TTS for the summary
- [ ] Route `POST /api/tts` accepting `{ text, language_code, speaker }`.
- [ ] Call TTS REST for pre-generated summaries; return audio bytes (decode the base64) as `audio/wav` (or return base64 to the frontend — pick one and document it).
- [ ] For the voice loop, add `POST /api/tts/stream` that proxies `text-to-speech/stream` chunks to the client for low latency.

### Phase 4 — Backend: STT for the voice question
- [ ] Route `POST /api/transcribe` accepting an audio blob (multipart).
- [ ] Call `speech_to_text.transcribe(model="saaras:v4", mode="codemix", keyterms=[...])`.
- [ ] Return `{ transcript, language_code }`.

### Phase 5 — Backend: the Q&A loop
- [ ] Route `POST /api/ask` accepting `{ paper_id, question }`.
- [ ] Build a grounded prompt: system message + the stored paper text + the user's question. Instruct the model to answer **only from the paper** and to say when the paper does not cover something.
- [ ] Return `{ answer }`.
- [ ] (Optional) Maintain short conversation history so follow-ups work.

### Phase 6 — Frontend
- [ ] **Upload + summary panel:** file input → `POST /api/upload` → show preview → button to `POST /api/summarize` → render summary → "Read aloud" button → `POST /api/tts` → play audio.
- [ ] **Voice Q&A panel:** "Hold to talk" button using `MediaRecorder` → on release, POST the blob to `/api/transcribe` → show transcript → POST `/api/ask` → show answer → auto-call `/api/tts` → play answer.
- [ ] Show loading/error states for every network call. Async OCR can take a while — show a progress indicator while polling.
- [ ] Keep the UI clean and minimal; no framework required.

### Phase 7 — Polish & hardening
- [ ] Centralise error handling on the backend (map Sarvam `4xx`/`5xx` to friendly JSON).
- [ ] Add basic request validation and file-size checks.
- [ ] Write `README.md` with run instructions.
- [ ] End-to-end test: upload a paper → get summary → hear it → ask a spoken question → hear the answer.

---

## 7. Known Constraints, Gotchas, and Required Workarounds

Address each of these explicitly in the code.

1. **10-page cap on Document Intelligence — chunk every PDF.** The API rejects any PDF over 10 pages (`400 invalid_request_error`), and research papers routinely exceed that. **The strategy is fixed: split the PDF into ≤10-page chunks, OCR each chunk, then concatenate the Markdown back in page order.** Do not attempt to upload a full paper as one file.

   Implement a `pdf_chunker.py` helper:

   ```python
   from pypdf import PdfReader, PdfWriter

   def split_pdf(path, pages_per_chunk=10):
       reader = PdfReader(path)
       total = len(reader.pages)
       chunks = []
       for start in range(0, total, pages_per_chunk):
           writer = PdfWriter()
           for p in range(start, min(start + pages_per_chunk, total)):
               writer.add_page(reader.pages[p])
           out = f"{path}.chunk_{start // pages_per_chunk}.pdf"
           with open(out, "wb") as f:
               writer.write(f)
           chunks.append((out, start))   # keep the start index for ordering
       return chunks
   ```

   Then per chunk: run `doc_ai.digitise(...)` → poll → download → read the `.md`, and **join the chunk outputs back in ascending page order** (chunk 2's pages continue after chunk 1's). Process chunks **sequentially** to respect the 10 requests/minute limit, and add a short delay between jobs. If a chunk returns `partially_completed`, still use its output but log the gap.

2. **Async OCR latency.** A `digitise` job is not instant. Poll `get_status` with a sensible interval and a max timeout; surface progress to the UI. Do not block the request thread indefinitely — consider returning a `job_id` and letting the frontend poll, or poll server-side with a timeout.

3. **Long papers vs. LLM context.** `sarvam-105b` has a 128K-token context window — usually enough for a paper's text, but not for a whole book. If the extracted text is too large, do a **map-reduce summarization**: summarise each section/chunk separately, then summarise the summaries. Guard against silently truncating.

4. **30-second cap on sync STT.** Design the voice UI around short utterances (push-to-talk). Reject or warn on recordings longer than 30s, or move to the Batch/streaming API later.

5. **Reasoning tokens are billed and count toward `max_tokens`.** If you set `max_tokens` low on `sarvam-105b`, reasoning can consume the budget and leave little or no visible `content`. Set `max_tokens` high enough, or lower `reasoning_effort` for the voice loop where latency matters.

6. **TTS returns base64.** Decode it. Also keep TTS text under 2,500 chars per REST call — chunk longer summaries at sentence boundaries.

7. **Parameter-name traps:** Document AI `digitise` uses `language` (not `language_code`), taking `file` + `language`. Output is read from `GET /results` (JSON `documents[].pages[].content`, trying md/json/html/none), falling back to `GET /download-url` (presigned ZIP). TTS/STT use `language_code`. Don't mix them up.

8. **Rate limits:** Document AI is 10 req/min. Add retry-with-backoff on `429` and `503`. Handle `403` as an invalid/missing key.

9. **Secrets:** the API key must never appear in frontend code, logs, or git. Read it from the environment only.

10. **Voice choice must match the model.** `bulbul:v3` speakers (short names) and `bulbul:v4-flash` persona IDs are not interchangeable. Pick one model and use its voice catalogue.

---

## 8. Environment & Secrets

`.env` (never committed):
```
SARVAM_API_KEY=your_key_here
```
Load it with `python-dotenv` at startup. Provide `.env.example` with a placeholder. **Never hardcode the key anywhere.**

---

## 9. Error Handling Contract

- Wrap every Sarvam call; catch the SDK's `ApiError`.
- Map: `400` → bad input (tell the user what's wrong); `403` → invalid key (server config issue); `429` → rate limited (retry with backoff); `503` → service overloaded (retry with backoff).
- Return a consistent JSON error shape from the backend, e.g. `{ "error": { "code": "...", "message": "..." } }`.
- Never leak the raw API key or full stack traces to the frontend.

---

## 10. Definition of Done (acceptance criteria)

- [ ] A user can upload a research paper PDF of any length; papers over 10 pages are split into ≤10-page chunks, OCR'd chunk by chunk, and re-assembled in page order into one accurate extracted-text summary.
- [ ] The summary can be played back as natural speech.
- [ ] A user can hold a button, speak a question about the paper, and hear a spoken answer.
- [ ] The answer is grounded in the uploaded paper (it declines gracefully for out-of-scope questions).
- [ ] The API key exists only in backend environment variables.
- [ ] `README.md` explains setup and how to run both frontend and backend.
- [ ] Chunked OCR (§7.1) is implemented and verified end-to-end on a paper longer than 10 pages.
- [ ] Every other constraint in §7 is either handled or explicitly documented in `DECISIONS.md`.

---

## 11. Quick Reference Card

| Capability | Endpoint | Model | Key params |
|---|---|---|---|
| Reasoning / chat | `POST /v1/chat/completions` | `sarvam-105b` | `messages`, `max_tokens`, `reasoning_effort` |
| OCR / paper reading | `POST /doc-ai/v1/job/digitise` | Sarvam Vision | `file`, `language="en-IN"` |
| Speech → text | `POST /speech-to-text` | `saaras:v4` | `file`, `mode="codemix"`, `keyterms` |
| Text → speech (REST) | `POST /text-to-speech` | `bulbul:v4-flash` | `text`, `language_code`, `speaker` |
| Text → speech (stream) | `POST /text-to-speech/stream` | `bulbul:v4-flash` | `text`, `language_code`, `speaker`, `output_audio_codec` |
| Auth (all) | header | — | `api-subscription-key: <KEY>` |

**Base URL:** `https://api.sarvam.ai`
**Docs:** `https://docs.sarvam.ai` · full dump: `https://docs.sarvam.ai/llms-full.txt` · docs MCP: `https://docs.sarvam.ai/_mcp/server`
