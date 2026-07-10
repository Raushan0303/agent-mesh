"""Text chunking — split documents into overlapping chunks before embedding.

Chunking is the ingestion step between "user uploads a document" and
"chunks are embedded and stored in pgvector." Without it, a 10-page
document becomes a single giant vector that matches nothing useful.

Two strategies:
  1. Fixed-size chunking with overlap (default) — simple, predictable,
     works for any text. Splits on token count with N-token overlap
     between adjacent chunks so context isn't lost at boundaries.
  2. Sentence-aware chunking — splits on sentence boundaries first,
     then packs sentences into chunks up to the target size. Better
     for prose because chunks don't break mid-sentence.

Both are agent-agnostic — they operate on raw text and return chunks.
"""

import logging
import re

logger = logging.getLogger("agentmesh.memory.chunking")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    strategy: str = "fixed",
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Target number of tokens (word-level approximation) per chunk.
        overlap: Number of tokens to overlap between adjacent chunks.
        strategy: "fixed" for token-count splitting, "sentence" for
                  sentence-boundary-aware splitting.

    Returns:
        List of chunk strings. Empty input returns an empty list.
    """
    if not text or not text.strip():
        return []

    if strategy == "sentence":
        return _chunk_by_sentences(text, chunk_size, overlap)
    return _chunk_fixed(text, chunk_size, overlap)


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size chunking with word-level overlap.

    Splits text into words, then slides a window of chunk_size words
    with overlap step. Each window becomes a chunk.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    logger.info(
        "CHUNK_FIXED words=%d chunks=%d chunk_size=%d overlap=%d",
        len(words), len(chunks), chunk_size, overlap,
    )
    return chunks


def _chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentence-aware chunking.

    Splits text into sentences, then packs sentences into chunks up to
    chunk_size words. Overlap is achieved by carrying the last N words
    of the previous chunk into the next.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        sent_words = sentence.split()
        # If a single sentence exceeds chunk_size, fall back to fixed split
        if len(sent_words) > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words).strip())
                current_words = []
                current_count = 0
            chunks.extend(_chunk_fixed(sentence, chunk_size, overlap))
            continue

        if current_count + len(sent_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words).strip())
            # Carry overlap words from the end of the previous chunk
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = list(overlap_words)
            current_count = len(overlap_words)

        current_words.extend(sent_words)
        current_count += len(sent_words)

    if current_words:
        chunks.append(" ".join(current_words).strip())

    logger.info(
        "CHUNK_SENTENCE sentences=%d chunks=%d chunk_size=%d overlap=%d",
        len(sentences), len(chunks), chunk_size, overlap,
    )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex that handles common cases.

    Handles periods, exclamation marks, and question marks followed by
    whitespace or end of string. Not perfect (abbreviations like "Dr."
    will split), but good enough for chunking purposes.
    """
    # Split on sentence-ending punctuation followed by space or end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]
