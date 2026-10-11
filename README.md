# Sarvam Paper Lab — Research Paper Assistant

A web application powered by the **Sarvam AI** stack that allows users to ingest academic research papers, produce structured reasoning-driven summaries read aloud with Text-to-Speech, and interact via a conversational voice-enabled Q&A loop.

---

## Features

1. **Research Paper Summarizer**
   - **Any-Length PDF Support**: Automatic in-memory chunking into ≤10-page blocks to comply with Sarvam Document AI limits.
   - **Document Intelligence (Sarvam Vision)**: Asynchronous OCR extraction preserving reading order, headings, and formatting in Markdown.
   - **Reasoning Summarization (Sarvam-105B)**: Generates structured summaries (TL;DR, Problem, Method, Key Concepts, Findings, Limitations) using single-pass or map-reduce depending on paper length.
   - **High-Fidelity Read-Aloud (Bulbul:v4-flash)**: Sentence-bounded text chunking and multi-voice TTS synthesis.

2. **Voice-Enabled Q&A Bot for Concept Testing**
   - **Push-to-Talk Speech Capture (MediaRecorder)**: Browser captures microphone input, resamples to 16 kHz mono 16-bit PCM WAV in real time, with a 30-second ring indicator.
   - **Speech-to-Text (Saaras:v4)**: Accurate technical transcription enhanced with paper-derived keyterm biasing (`keyterms`).
   - **Grounded Conversational Reasoning (Sarvam-105B)**: Answers strictly grounded in the paper context, keeping short conversational history.
   - **Low-Latency Spoken Responses (Bulbul streaming)**: Proxied chunked audio streaming for natural conversational tempo.

---

## Architecture

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐           ┌─────────────────────────┐
│       Frontend (Browser)        │   HTTP    │       FastAPI Backend           │  HTTPS    │       Sarvam AI API     │
│  - Push-to-Talk (16kHz PCM WAV) │ ────────► │  - Holds SARVAM_API_KEY (.env)  │ ────────► │  - Document AI (OCR)    │
│  - Segmented OCR Progress       │           │  - In-Memory PDF Chunker        │           │  - Sarvam-105B (Chat)   │
│  - Markdown & Audio Player      │ ◄──────── │  - Rate-limiting & Token Budget │ ◄──────── │  - Saaras:v4 (STT)      │
│  - Pipeline Status Console      │           │  - Paper Cache (data/papers/)   │           │  - Bulbul:v4-flash (TTS)│
└─────────────────────────────────┘           └─────────────────────────────────┘           └─────────────────────────┘
```

> **Security Note:** The Sarvam API subscription key is stored only on the backend in `.env`. The frontend never accesses or exposes the secret.

---

## Repository Layout

All commands and file references are run from the project root directory:

```text
.env                      <- created from .env.example (gitignored)
.env.example
.gitignore
README.md  DECISIONS.md  AGENT_CONTEXT.md
backend/
  main.py
  sarvam_client.py
  pdf_chunker.py
  paper_store.py
  text_utils.py
  smoke_test.py
  test_integration.py
  requirements.txt
frontend/
  index.html
  app.js
  styles.css
data/papers/              (storage directory for cached extracted text)
```

---

## Quickstart & Setup

> **Note:** All commands must be run from the project root directory.

### 1. Prerequisites
- Python 3.11+
- A valid Sarvam AI API subscription key from the [Sarvam Dashboard](https://dashboard.sarvam.ai/).

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Configure Secrets
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Edit `.env` and set your key:
```env
SARVAM_API_KEY=your_actual_sarvam_api_key_here
```

### 4. Run the Smoke Test
Verify your credentials and connectivity across Chat, TTS, and STT:
```bash
python backend/smoke_test.py --all
```
Expected output:
```text
[*] Chat (sarvam-105b) ...
  [OK] The transformer architecture is a deep learning model...
[*] TTS (bulbul:v4-flash, REST) ...
  [OK] 130,580 bytes of WAV audio
[*] STT (saaras:v4) on that audio ...
  [OK] transcript: 'Hello, the Sarvam research assistant is working.' (en-IN)
```

---

## Running the Application

From the project root, start the backend server (which also serves the frontend from the same origin):
```bash
python backend/main.py
```
Or with Uvicorn:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check and configuration status |
| `/api/upload` | POST | Upload PDF; splits into ≤10-page chunks and begins async OCR |
| `/api/papers/{paper_id}` | GET | Polls OCR progress, chunk status, and text preview |
| `/api/summarize` | POST | Generates structured Markdown summary via Sarvam-105B |
| `/api/tts` | POST | Synthesizes full summary audio using `bulbul:v4-flash` |
| `/api/tts/stream` | POST | Generates a short-lived token for low-latency streaming TTS |
| `/api/tts/stream/{token}` | GET | Audio streaming endpoint for Q&A spoken replies |
| `/api/transcribe` | POST | Transcribes audio via `saaras:v4` with paper keyterm biasing |
| `/api/ask` | POST | Grounded Q&A against the paper via Sarvam-105B |
| `/api/papers/{paper_id}/reset` | POST | Resets conversational memory for a paper |

---

## Keyboard Shortcuts

- <kbd>Space</kbd> (Hold): Push-to-talk voice question.
- <kbd>S</kbd>: Trigger summary generation once a paper is ingested.
