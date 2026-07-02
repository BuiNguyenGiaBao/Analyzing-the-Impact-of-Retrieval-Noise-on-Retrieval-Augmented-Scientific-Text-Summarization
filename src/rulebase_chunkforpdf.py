# -*- coding: utf-8 -*-
"""Rule-based section parsing and robust word-window chunking for papers."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    import nltk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nltk = None  # type: ignore


class ChunkConfig:
    MAX_HEADING_LENGTH = 120
    MAX_HEADING_WORDS = 12
    TITLE_CASE_THRESHOLD = 0.6

    MIN_SECTION_WORDS = 10

    MAX_CHUNK_WORDS = 150
    OVERLAP_WORDS = 30
    MIN_CHUNK_WORDS = 40

    # Hard safety bound: no emitted chunk should be larger than this.
    HARD_MAX_CHUNK_WORDS = 180


COMMON_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
    "appendix",
    "acknowledgements",
    "acknowledgments",
}

RE_NUMBERED = re.compile(
    r"""^
    (
        (\d+(\.\d+){0,3})
        |
        ([IVXLCDM]+)
    )
    [\.\)\:]?
    \s+
    ([A-Za-z].+)
    $""",
    re.VERBOSE,
)

RE_SPACE = re.compile(r"\s+")
RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?")


def normalize_space(s: str) -> str:
    if not s:
        return ""
    s = str(s).replace("\u00a0", " ").replace("\\n", "\n").replace("/n", "\n")
    return RE_SPACE.sub(" ", s).strip()


def word_count(text: str) -> int:
    text = normalize_space(text)
    return len(text.split()) if text else 0


def _truncate_words(text: str, max_words: int) -> str:
    words = normalize_space(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def looks_like_heading(line: str) -> bool:
    line = normalize_space(line)
    if not line:
        return False
    if len(line) > ChunkConfig.MAX_HEADING_LENGTH:
        return False
    if line.endswith(".") and len(line.split()) > 3:
        return False
    if len(line.split()) > ChunkConfig.MAX_HEADING_WORDS + 2:
        return False

    m = RE_NUMBERED.match(line)
    if m:
        title_words = m.group(5).split()
        return 1 <= len(title_words) <= ChunkConfig.MAX_HEADING_WORDS

    words = line.split()
    if line.isupper() and 2 <= len(words) <= 10:
        return True
    if line.lower().strip(" .:") in COMMON_SECTIONS:
        return True
    if 1 <= len(words) <= 10:
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
        if not alpha_words:
            return False
        ratio = sum(w[0].isupper() for w in alpha_words if w) / max(len(alpha_words), 1)
        if ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True
    return False


def clean_heading(line: str) -> str:
    line = normalize_space(line)
    m = RE_NUMBERED.match(line)
    if m:
        return normalize_space(m.group(5)).strip(" .:")
    if line.isupper():
        return line.title()
    return line.strip(" .:")


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    text = normalize_space(text)
    if nltk is not None:
        try:
            sents = nltk.sent_tokenize(text)
            cleaned = [normalize_space(s) for s in sents if normalize_space(s)]
            if cleaned:
                return cleaned
        except Exception:
            pass

    # Fallback: split after sentence punctuation, but also tolerate LaTeX-heavy text.
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])", text)
    cleaned = [normalize_space(s) for s in sents if normalize_space(s)]
    return cleaned if cleaned else [text]


def rule_based_section_parse(text: str) -> List[Tuple[str, str]]:
    text = "" if text is None else str(text)
    lines = [normalize_space(x) for x in text.splitlines()]

    sections: List[Tuple[str, str]] = []
    current_title = "Body"
    current_buf: List[str] = []

    for line in lines:
        if not line:
            continue
        if looks_like_heading(line):
            if current_buf:
                sections.append((current_title, " ".join(current_buf)))
            current_title = clean_heading(line) or "Body"
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, " ".join(current_buf)))

    cleaned_sections = []
    for title, content in sections:
        content = normalize_space(content)
        if word_count(content) >= ChunkConfig.MIN_SECTION_WORDS:
            cleaned_sections.append((title, content))
    return cleaned_sections


def _split_overlong_sentence(sentence: str, max_words: int) -> List[str]:
    words = normalize_space(sentence).split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[i : i + max_words]).strip()
        for i in range(0, len(words), max_words)
        if words[i : i + max_words]
    ]


def _make_chunk(
    chunk_id: int,
    section_title: str,
    text: str,
    sec_index: int,
    chunk_index_in_section: int,
    source_doc_id: Optional[str],
    chunking_method: str = "rule_window",
) -> Dict:
    safe_text = _truncate_words(text, ChunkConfig.HARD_MAX_CHUNK_WORDS)
    wc = word_count(safe_text)
    return {
        "chunk_id": int(chunk_id),
        "section": section_title or "Body",
        "text": safe_text,
        "word_count": int(wc),
        "section_index": int(sec_index),
        "chunk_index_in_section": int(chunk_index_in_section),
        "source_doc_id": source_doc_id,
        "chunking_method": chunking_method,
    }


def chunk_sections(
    sections: List[Tuple[str, str]],
    source_doc_id: Optional[str] = None,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> List[Dict]:
    if max_words is None:
        max_words = ChunkConfig.MAX_CHUNK_WORDS
    if overlap_words is None:
        overlap_words = ChunkConfig.OVERLAP_WORDS

    max_words = max(20, min(int(max_words), ChunkConfig.HARD_MAX_CHUNK_WORDS))
    overlap_words = max(0, min(int(overlap_words), max_words - 1))

    chunks: List[Dict] = []
    chunk_id = 0

    for sec_index, (section_title, content) in enumerate(sections):
        raw_sents = split_sentences(content)
        sents: List[str] = []
        for sent in raw_sents:
            sents.extend(_split_overlong_sentence(sent, max_words))

        chunk_index_in_section = 0
        window: List[str] = []
        window_words = 0

        def emit_window(force: bool = False) -> None:
            nonlocal chunk_id, chunk_index_in_section, window, window_words
            text = normalize_space(" ".join(window))
            if not text:
                window = []
                window_words = 0
                return
            wc = word_count(text)
            if force or wc >= ChunkConfig.MIN_CHUNK_WORDS or len(chunks) == 0:
                chunks.append(
                    _make_chunk(
                        chunk_id=chunk_id,
                        section_title=section_title,
                        text=text,
                        sec_index=sec_index,
                        chunk_index_in_section=chunk_index_in_section,
                        source_doc_id=source_doc_id,
                    )
                )
                chunk_id += 1
                chunk_index_in_section += 1

            # Rebuild overlap by words, not whole sentences only. This avoids losing
            # overlap when a single sentence/block is longer than overlap_words.
            all_words = text.split()
            overlap = all_words[-overlap_words:] if overlap_words > 0 else []
            window = [" ".join(overlap)] if overlap else []
            window_words = len(overlap)

        for sent in sents:
            s_words = len(sent.split())
            if s_words == 0:
                continue
            if window_words + s_words > max_words and window:
                emit_window(force=False)
            # If the overlap alone plus sent is still too large, drop overlap.
            if window_words + s_words > max_words:
                window = []
                window_words = 0
            window.append(sent)
            window_words += s_words

        if window:
            emit_window(force=True)

    # Final safety: remove empty chunks and renumber continuously.
    safe_chunks: List[Dict] = []
    for new_id, ch in enumerate(chunks):
        text = normalize_space(ch.get("text", ""))
        if not text:
            continue
        item = dict(ch)
        item["chunk_id"] = new_id
        item["word_count"] = word_count(text)
        item["text"] = text
        safe_chunks.append(item)
    return safe_chunks


def process_document(text: str, source_doc_id: Optional[str] = None) -> Dict:
    text = "" if text is None else str(text)
    sections = rule_based_section_parse(text)
    if len(sections) == 0 and normalize_space(text):
        sections = [("Body", normalize_space(text))]
    chunks = chunk_sections(sections, source_doc_id=source_doc_id)
    return {
        "sections": sections,
        "chunks": chunks,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
    }
