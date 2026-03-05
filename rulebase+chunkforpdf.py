import re
from typing import List, Tuple, Dict, Optional
import nltk


class ChunkConfig:
    # Heading detection
    MAX_HEADING_LENGTH   = 120   # characters
    MAX_HEADING_WORDS    = 12
    TITLE_CASE_THRESHOLD = 0.6

    # Section filtering
    MIN_SECTION_WORDS    = 40    # drop tiny sections

    # Chunking budgets (choose one primary budget)
    MAX_CHUNK_WORDS      = 320   # sentence-aware word budget per chunk
    OVERLAP_WORDS        = 40    # overlap in words (approx; implemented via sentences)
    MIN_CHUNK_WORDS      = 80    # drop tiny chunks

    # Optional character budget (secondary safety net)
    MAX_CHUNK_CHARS      = 2200  # rough cap to avoid extreme long sentences/lines


COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "method", "methods", "methodology", "materials and methods",
    "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "limitations", "future work",
    "acknowledgments", "acknowledgements", "references", "appendix",
    "supplementary material", "supplementary materials", "bibliography"
}

RE_NUMBERED = re.compile(
    r"""^(
        (\d+(\.\d+){0,3})
        |([IVXLCDM]+)
    )[\.\)\:]?\s+([A-Za-z].+)$""",
    re.VERBOSE,
)

def normalize_space(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def word_count(text: str) -> int:
    text = normalize_space(text)
    return 0 if not text else len(text.split())

def looks_like_heading(line: str) -> bool:
    line = normalize_space(line)
    if not line:
        return False
    if len(line) > ChunkConfig.MAX_HEADING_LENGTH:
        return False
    # Avoid treating normal sentences as headings
    if line.endswith(".") and len(line.split()) > 3:
        return False

    # Numbered headings like "1. Introduction", "2.3 Experiments", "IV. Results"
    m = RE_NUMBERED.match(line)
    if m:
        title = m.group(5).strip()
        return 1 <= len(title.split()) <= ChunkConfig.MAX_HEADING_WORDS

    # ALL CAPS headings
    if line.isupper() and 1 <= len(line.split()) <= 10:
        return True

    # Common section names
    low = line.lower()
    if low in COMMON_SECTIONS:
        return True

    # Title Case heuristic
    words = line.split()
    if 1 <= len(words) <= 10:
        ratio = sum(w[0].isupper() for w in words if w) / max(len(words), 1)
        if ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True

    return False

def clean_heading(line: str) -> str:
    line = normalize_space(line)
    m = RE_NUMBERED.match(line)
    if m:
        return normalize_space(m.group(5))
    return line.title() if line.isupper() else line

def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    try:
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s and s.strip()]
    except Exception:
        # fallback: simple regex split
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sents if s and s.strip()]



def rule_based_section_parse(text: str) -> List[Tuple[str, str]]:
    """
    Parse raw extracted text into (section_title, section_text).
    Removes very short sections using word count.
    """
    if not text or not text.strip():
        return []

    lines = [normalize_space(x) for x in text.splitlines()]
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Body"
    current_buf: List[str] = []

    for line in lines:
        if not line:
            continue
        if looks_like_heading(line):
            if current_buf:
                sections.append((current_title, " ".join(current_buf).strip()))
                current_buf = []
            current_title = clean_heading(line)
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, " ".join(current_buf).strip()))

    # Filter tiny sections
    filtered = []
    for t, c in sections:
        if word_count(c) >= ChunkConfig.MIN_SECTION_WORDS:
            filtered.append((t, c))
    return filtered


def chunk_sections(
    sections: List[Tuple[str, str]],
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
    min_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    drop_sections: Optional[set] = None,
) -> List[Dict]:
    """
    Sentence-aware chunking with word/char budgets and overlap.
    This intentionally avoids model tokenizers.
    """
    if max_words is None:
        max_words = ChunkConfig.MAX_CHUNK_WORDS
    if overlap_words is None:
        overlap_words = ChunkConfig.OVERLAP_WORDS
    if min_words is None:
        min_words = ChunkConfig.MIN_CHUNK_WORDS
    if max_chars is None:
        max_chars = ChunkConfig.MAX_CHUNK_CHARS

    if drop_sections is None:
        drop_sections = {"references", "bibliography"}

    chunks: List[Dict] = []
    chunk_id = 0

    for section_title, content in sections:
        if section_title.lower() in drop_sections:
            continue

        sents = split_sentences(content)
        if not sents:
            continue

        current: List[str] = []
        current_words = 0
        current_chars = 0

        def flush(buf: List[str]) -> Optional[Dict]:
            nonlocal chunk_id
            txt = normalize_space(" ".join(buf))
            wc = word_count(txt)
            if wc >= min_words:
                out = {
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": txt,
                    "word_count": wc,
                    "char_count": len(txt),
                }
                chunk_id += 1
                return out
            return None

        def build_overlap(buf: List[str]) -> List[str]:
            if overlap_words <= 0 or not buf:
                return []
            # collect sentences from the end until reaching overlap_words (approx by words)
            ov: List[str] = []
            acc = 0
            for s in reversed(buf):
                w = len(s.split())
                if acc + w > overlap_words and ov:
                    break
                ov.insert(0, s)
                acc += w
            return ov

        for s in sents:
            s = s.strip()
            if not s:
                continue
            s_words = len(s.split())
            s_chars = len(s)

            # If a single sentence is too long, emit it as a standalone chunk (flag oversized)
            if s_words >= max_words or s_chars >= max_chars:
                if current:
                    c = flush(current)
                    if c:
                        chunks.append(c)
                    current, current_words, current_chars = [], 0, 0

                txt = normalize_space(s)
                chunks.append({
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": txt,
                    "word_count": word_count(txt),
                    "char_count": len(txt),
                    "oversized": True
                })
                chunk_id += 1
                continue

            # Check if adding this sentence exceeds budgets
            if current and (current_words + s_words > max_words or current_chars + s_chars > max_chars):
                c = flush(current)
                if c:
                    chunks.append(c)
                ov = build_overlap(current)
                current = ov + [s]
                current_words = sum(len(x.split()) for x in current)
                current_chars = sum(len(x) for x in current)
            else:
                current.append(s)
                current_words += s_words
                current_chars += s_chars

        if current:
            c = flush(current)
            if c:
                chunks.append(c)

    return chunks

def process_document(
    text: str,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> Dict:
    """
    Full pipeline: section parse -> chunk.
    """
    sections = rule_based_section_parse(text)
    chunks = chunk_sections(sections, max_words=max_words, overlap_words=overlap_words)
    return {
        "chunking": "sentence-aware word/char budget ",
        "sections": sections,
        "chunks": chunks,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
    }
