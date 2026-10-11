"""
Split a PDF into <=10-page chunks (Sarvam Doc AI rejects PDFs over 10 pages).

Works fully in memory: returns a list of (chunk_bytes, start_page_index, page_count)
in ascending page order so OCR output can be re-assembled in order.
"""

from __future__ import annotations

import io

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

PAGES_PER_CHUNK = 10


class PdfChunkError(ValueError):
    pass


def count_pages(pdf_bytes: bytes) -> int:
    return len(_reader(pdf_bytes).pages)


def split_pdf(pdf_bytes: bytes, pages_per_chunk: int = PAGES_PER_CHUNK) -> list[tuple[bytes, int, int]]:
    reader = _reader(pdf_bytes)
    total = len(reader.pages)
    if total == 0:
        raise PdfChunkError("The PDF has no pages.")

    chunks: list[tuple[bytes, int, int]] = []
    for start in range(0, total, pages_per_chunk):
        end = min(start + pages_per_chunk, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append((buf.getvalue(), start, end - start))
    return chunks


def _reader(pdf_bytes: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise PdfChunkError("The PDF is password-protected.") from exc
        return reader
    except PdfReadError as exc:
        raise PdfChunkError(f"Could not read PDF: {exc}") from exc
