import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageChunk:
    page: int  # 1-based
    idx: int
    content: str


def extract_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text("text", sort=True) for page in doc]


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph, then sentence boundaries, packing pieces up to `size` chars."""
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:  # a single enormous "sentence"
                pieces.append(sent[:size])
                sent = sent[size - overlap :]
            if sent:
                pieces.append(sent)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:]
            # Start the overlap on a word boundary.
            space = tail.find(" ")
            current = (tail[space + 1 :] if space != -1 else tail) + " " + piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[PageChunk]:
    out: list[PageChunk] = []
    for page_no, raw in enumerate(pages, start=1):
        text = _clean(raw)
        if len(text) < 20:
            continue
        for part in _split(text, size, overlap):
            out.append(PageChunk(page=page_no, idx=len(out), content=part))
    return out
