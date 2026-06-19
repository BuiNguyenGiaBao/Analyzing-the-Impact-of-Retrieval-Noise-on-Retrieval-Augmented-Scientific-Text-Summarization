import re
from typing import List, Dict, Optional, Tuple

import nltk


class ChunkConfig:
    MAX_HEADING_LENGTH   = 120   # chars; longer lines are never headings
    MAX_HEADING_WORDS    = 12    # word ceiling for a heading
    TITLE_CASE_THRESHOLD = 0.6   # fraction of words that must start uppercase

    MIN_SECTION_WORDS    = 10    # drop sections shorter than this

    MAX_CHUNK_WORDS      = 150   
    OVERLAP_WORDS        = 30    
    MIN_CHUNK_WORDS      = 40   

COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "literature review", "method", "methods", "methodology",
    "experiments", "results", "discussion", "conclusion",
    "limitations", "future work", "references", "appendix",
}

# Compiled once at import time.
# Matches:  "1.", "2.3.1:", "IV." or "III)" followed by a heading title.
RE_NUMBERED = re.compile(
    r"""^
    (
        (\d+(\.\d+){0,3})   # Arabic:  1  /  1.2  /  1.2.3  /  1.2.3.4
        |
        ([IVXLCDM]+)        # Roman:   I  /  IV  /  XLII
    )
    [\.\)\:]?               # optional trailing  .  )  :
    \s+
    ([A-Za-z].+)            # heading text — must start with a letter
    $""",
    re.VERBOSE,
)


# Text utilities

def normalize_space(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def word_count(text: str) -> int:
    text = normalize_space(text)
    return len(text.split()) if text else 0

# Heading detection

def looks_like_heading(line: str) -> bool:
    line = normalize_space(line)
    if not line:
        return False

    # Hard length/punctuation rules
    if len(line) > ChunkConfig.MAX_HEADING_LENGTH:
        return False
    if line.endswith(".") and len(line.split()) > 3:
        return False

    # Numbered heading (Arabic or Roman)
    m = RE_NUMBERED.match(line)
    if m:
        title_words = m.group(5).split()
        return 1 <= len(title_words) <= ChunkConfig.MAX_HEADING_WORDS

    words = line.split()

    # All-caps heading: require ≥ 2 words to avoid matching abbreviations / symbols
    if line.isupper() and 2 <= len(words) <= 10:
        return True

    # Known section keyword (exact, lowercase)
    if line.lower() in COMMON_SECTIONS:
        return True

    # Title Case heuristic
    if 1 <= len(words) <= 10:
        ratio = sum(w[0].isupper() for w in words if w) / max(len(words), 1)
        if ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True

    return False


def clean_heading(line: str) -> str:
    """Strip numbering prefix and normalise whitespace."""
    line = normalize_space(line)
    m = RE_NUMBERED.match(line)
    if m:
        return normalize_space(m.group(5))
    if line.isupper():
        return line.title()
    return line


# Sentence splitting

def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    try:
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sents if s.strip()]


# Section parser

def rule_based_section_parse(text: str) -> List[Tuple[str, str]]:
    """
    Split *text* into (section_title, content) tuples.
    Sections shorter than MIN_SECTION_WORDS are discarded.
    """
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
            current_title = clean_heading(line)
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, " ".join(current_buf)))

    return [
        (title, content)
        for title, content in sections
        if word_count(content) >= ChunkConfig.MIN_SECTION_WORDS
    ]


# Sliding-window chunker with real overlap

def chunk_sections(
    sections: List[Tuple[str, str]],
    source_doc_id: Optional[str] = None,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> List[Dict]:
    """
    Chunk each section using a sliding window.

    The *overlap_words* tail of each completed chunk is carried forward as
    the opening sentences of the next chunk (true overlap — was a no-op before).

    Returns a list of chunk dicts with keys:
        chunk_id, section, text, word_count,
        section_index, chunk_index_in_section, source_doc_id
    """
    if max_words is None:
        max_words = ChunkConfig.MAX_CHUNK_WORDS
    if overlap_words is None:
        overlap_words = ChunkConfig.OVERLAP_WORDS

    chunks: List[Dict] = []
    chunk_id = 0

    for sec_index, (section_title, content) in enumerate(sections):
        sents = split_sentences(content)
        chunk_index_in_section = 0

        # `window` holds sentences for the current chunk being built.
        window: List[str] = []
        window_words = 0

        for sent in sents:
            s_words = len(sent.split())

            if window_words + s_words > max_words and window:
                # --- Emit current chunk ---
                text = normalize_space(" ".join(window))
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "section": section_title,
                        "text": text,
                        "word_count": word_count(text),
                        "section_index": sec_index,
                        "chunk_index_in_section": chunk_index_in_section,
                        "source_doc_id": source_doc_id,
                    }
                )
                chunk_id += 1
                chunk_index_in_section += 1

                # --- Build overlap carry-over ---
                # Walk backwards through `window` collecting sentences until
                # their total word count reaches overlap_words.
                overlap_sents: List[str] = []
                overlap_total = 0
                for s in reversed(window):
                    w = len(s.split())
                    if overlap_total + w > overlap_words:
                        break
                    overlap_sents.insert(0, s)
                    overlap_total += w

                window = overlap_sents
                window_words = overlap_total

            window.append(sent)
            window_words += s_words

        # Emit any remaining sentences in the window
        if window:
            text = normalize_space(" ".join(window))
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": text,
                    "word_count": word_count(text),
                    "section_index": sec_index,
                    "chunk_index_in_section": chunk_index_in_section,
                    "source_doc_id": source_doc_id,
                }
            )
            chunk_id += 1

    return chunks


def process_document(text: str, source_doc_id: Optional[str] = None) -> Dict:

    sections = rule_based_section_parse(text)

    if len(sections) == 0:
        sections = [("Body", text)]

    chunks = chunk_sections(sections, source_doc_id=source_doc_id)

    return {
        "sections": sections,
        "chunks": chunks,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
    }