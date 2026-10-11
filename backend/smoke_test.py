"""
Phase 0 smoke test: proves the Sarvam API key works before anything else.

    python backend/smoke_test.py            # chat only
    python backend/smoke_test.py --all      # chat + TTS + STT round-trip
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sarvam_client as sarvam  # noqa: E402


async def main(run_all: bool) -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ok = True
    try:
        print("[*] Chat (sarvam-105b) ...")
        res = await sarvam.chat(
            [{"role": "user", "content": "In one sentence, what is the transformer architecture?"}],
            max_tokens=1024, reasoning_effort=None,
        )
        print(f"  [OK] {res['content']}\n  usage: {res['usage']}")

        if run_all:
            print("[*] TTS (bulbul:v4-flash, REST) ...")
            b64 = await sarvam.tts_rest("Hello! The Sarvam research assistant is working.", codec="wav")
            audio = base64.b64decode(b64)
            print(f"  [OK] {len(audio):,} bytes of WAV audio")

            print("[*] STT (saaras:v4) on that audio ...")
            stt = await sarvam.transcribe(audio, "hello.wav", "audio/wav", keyterms=["Sarvam"])
            print(f"  [OK] transcript: {stt['transcript']!r} ({stt['language_code']})")
    except sarvam.SarvamError as exc:
        ok = False
        print(f"  [FAIL] [{exc.status} {exc.code}] {exc.message}")
    finally:
        await sarvam.aclose()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--all" in sys.argv)))
