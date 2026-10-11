"""
Text helpers: token estimation, chunking for map-reduce and TTS, markdown
stripping for speech, keyterm extraction for STT biasing, and a lightweight
lexical retriever for papers that exceed the LLM context window.
"""

from __future__ import annotations

import re
from collections import Counter

# Conservative: English academic text averages ~4 chars/token; use 3.2 to leave headroom.
CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


# --------------------------------------------------------------------------
# Generic chunking (paragraph-aware) for map-reduce summarisation
# --------------------------------------------------------------------------
def split_text(text: str, max_chars: int) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        # Hard-split any single paragraph that is itself too large.
        while len(para) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if size + len(para) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# --------------------------------------------------------------------------
# Speech helpers
# --------------------------------------------------------------------------
def strip_markdown(md: str) -> str:
    """Turn Markdown into clean prose suitable for TTS."""
    text = md
    text = re.sub(r"```.*?```", " ", text, flags=re.S)           # code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)                     # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links
    text = re.sub(r"<[^>]+>", " ", text)                         # html tags
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)         # display math
    text = re.sub(r"^\s*\|?[\s:\-|]+\|?\s*$", " ", text, flags=re.M)  # table rules
    text = text.replace("|", ", ")
    # Headings become sentences.
    text = re.sub(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", r"\1.", text, flags=re.M)
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.M)        # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)       # numbered lists
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)              # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                     # italics *x*
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)           # italics _x_
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)             # quotes
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Join lines that don't already end a sentence.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = []
    for ln in lines:
        if out and not re.search(r"[.!?:;]$", out[-1]):
            out[-1] += "."
        out.append(ln)
    return " ".join(out).strip()


def split_for_tts(text: str, max_chars: int = 2400) -> list[str]:
    """Split prose into <= max_chars pieces at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        while len(sent) > max_chars:  # pathological long sentence: split on commas/space
            cut = sent.rfind(",", 0, max_chars)
            cut = cut if cut > max_chars // 2 else sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        if len(buf) + len(sent) + 1 > max_chars:
            chunks.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------
# Keyterms for STT biasing (saaras:v4 supports up to 50 terms)
# --------------------------------------------------------------------------
_STOP = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its we our us you your they their them he she his her i me my not no nor
can could may might will would shall should do does did done have has had having than too very
into over under about above below between through during before after again further once here
there when where why how all any both each few more most other some such only own same so just
also however therefore thus using use used based paper section figure table results result method
methods approach model models data et al fig eq equation proposed show shows shown new one two
""".split())


def extract_keyterms(text: str, limit: int = 50) -> list[str]:
    """Pick acronyms and frequent capitalised/technical terms from the paper."""
    sample = text[:200_000]
    counts: Counter[str] = Counter()

    for m in re.finditer(r"\b[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9\-]*\b", sample):  # acronyms / CamelCase
        tok = m.group(0)
        if 2 <= len(tok) <= 30 and not tok.isdigit():
            counts[tok] += 3
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:[ \-][A-Z][a-z]{2,}){1,2}\b", sample):  # Multi-Word Names
        counts[m.group(0)] += 2
    for m in re.finditer(r"\b[a-z][a-z\-]{5,}\b", sample.lower()):  # frequent long words
        w = m.group(0)
        if w not in _STOP:
            counts[w] += 1

    terms: list[str] = []
    seen: set[str] = set()
    for term, score in counts.most_common(300):
        key = term.lower()
        if key in _STOP or key in seen or score < 3:
            continue
        seen.add(key)
        terms.append(term[:64])
        if len(terms) >= limit:
            break
    return terms


# --------------------------------------------------------------------------
# Lightweight retrieval for very long papers
# --------------------------------------------------------------------------
def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2]


def select_relevant(text: str, query: str, budget_chars: int, chunk_chars: int = 4000) -> str:
    """Return the most query-relevant chunks of `text`, kept in original order, within budget."""
    if len(text) <= budget_chars:
        return text
    chunks = split_text(text, chunk_chars)
    q = Counter(_words(query))
    scored = []
    for idx, ch in enumerate(chunks):
        wc = Counter(_words(ch))
        score = sum(min(wc[w], 5) * qn for w, qn in q.items())
        scored.append((score, idx))
    # Always keep the opening chunk (abstract/intro) for orientation.
    chosen = {0}
    used = len(chunks[0])
    for score, idx in sorted(scored, reverse=True):
        if idx in chosen:
            continue
        if used + len(chunks[idx]) > budget_chars:
            continue
        chosen.add(idx)
        used += len(chunks[idx])
    return "\n\n[…]\n\n".join(chunks[i] for i in sorted(chosen))
