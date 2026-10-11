"""
End-to-end integration test for the Sarvam Research Assistant backend routes.
Verifies PDF chunking, health check, summarization, TTS streaming, and Q&A.
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import httpx
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app  # noqa: E402
import paper_store  # noqa: E402
import pdf_chunker  # noqa: E402


def make_test_pdf(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


async def run_tests() -> bool:
    all_passed = True
    print("[*] Starting integration tests...\n")

    # 1. PDF Chunker test (15 pages -> 2 chunks: 10 + 5)
    print("Test 1: PDF Chunker with 15 pages...")
    pdf_15 = make_test_pdf(15)
    chunks = pdf_chunker.split_pdf(pdf_15)
    if len(chunks) == 2 and chunks[0][2] == 10 and chunks[1][2] == 5:
        print("  [OK] PDF split into 2 chunks (10 pages, 5 pages)")
    else:
        print(f"  [FAIL] Expected 2 chunks, got {len(chunks)}")
        all_passed = False

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
        # 2. Health check
        print("\nTest 2: GET /api/health...")
        res = await client.get("/api/health")
        data = res.json()
        if res.status_code == 200 and data.get("key_configured"):
            print(f"  [OK] Health OK, key_configured=True, models={data.get('models')}")
        else:
            print(f"  [FAIL] Health check failed: {res.status_code} {data}")
            all_passed = False

        # 3. Summarization with raw text
        print("\nTest 3: POST /api/summarize (raw text)...")
        sample_text = (
            "Attention Is All You Need. We propose the Transformer, a model architecture eschewing "
            "recurrence and instead relying entirely on an attention mechanism to draw global dependencies "
            "between input and output. The Transformer allows for significantly more parallelization and can "
            "reach a new state of the art in translation quality after being trained for as little as twelve hours."
        )
        res = await client.post("/api/summarize", json={"text": sample_text})
        data = res.json()
        if res.status_code == 200 and "summary" in data:
            print("  [OK] Summary generated successfully!")
            print(f"  Preview: {data['summary'][:150]}...")
        else:
            print(f"  [FAIL] Summarization failed: {res.status_code} {data}")
            all_passed = False

        # 4. REST TTS
        print("\nTest 4: POST /api/tts (REST audio synthesis)...")
        res = await client.post("/api/tts", json={"text": "The Transformer uses self-attention mechanisms."})
        if res.status_code == 200 and len(res.content) > 1000 and res.headers.get("content-type") == "audio/mpeg":
            print(f"  [OK] TTS generated {len(res.content):,} bytes of audio/mpeg")
        else:
            print(f"  [FAIL] TTS failed: {res.status_code}")
            all_passed = False

        # 5. Streaming TTS
        print("\nTest 5: POST /api/tts/stream and GET stream chunking...")
        res = await client.post("/api/tts/stream", json={"text": "Self-attention enables parallel sequence processing."})
        data = res.json()
        if res.status_code == 200 and "stream_url" in data:
            stream_url = data["stream_url"]
            print(f"  [OK] Got stream URL: {stream_url}")
            stream_res = await client.get(stream_url)
            if stream_res.status_code == 200 and len(stream_res.content) > 1000:
                print(f"  [OK] Audio stream fetched {len(stream_res.content):,} bytes")
            else:
                print(f"  [FAIL] Stream fetch failed: {stream_res.status_code}")
                all_passed = False
        else:
            print(f"  [FAIL] TTS stream creation failed: {res.status_code} {data}")
            all_passed = False

        # 6. Grounded Q&A against a test paper
        print("\nTest 6: POST /api/ask (Grounded Q&A)...")
        test_paper = paper_store.create("transformer_paper.pdf")
        test_paper.state = "ready"
        test_paper.stage = "Ready"
        test_paper.text = sample_text
        paper_store.save(test_paper)

        res = await client.post("/api/ask", json={
            "paper_id": test_paper.paper_id,
            "question": "Does the Transformer use recurrence or recurrent layers?"
        })
        data = res.json()
        if res.status_code == 200 and "answer" in data:
            print(f"  [OK] Q&A Answer: {data['answer']}")
        else:
            print(f"  [FAIL] Q&A failed: {res.status_code} {data}")
            all_passed = False

    print("\n" + ("=" * 40))
    if all_passed:
        print("[ALL INTEGRATION TESTS PASSED]")
    else:
        print("[SOME INTEGRATION TESTS FAILED]")
    print("=" * 40)
    return all_passed


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
