import asyncio
import logging
import re
from pathlib import Path

import httpx
from sqlalchemy import delete, select, text

from . import openrouter
from .config import get_settings
from .db import SessionLocal
from .models import Chunk, Document
from .pdf import chunk_pages, extract_pages
from .sqltypes import vector_literal

log = logging.getLogger("marginalia")
settings = get_settings()

SYSTEM_PROMPT = """You are Marginalia, a careful reading assistant. Answer the user's question using only the numbered source passages from their PDFs.

Rules:
- Cite every factual claim with its source number in square brackets right after the claim, e.g. "Revenue grew 12% [3]." Combine as [1][4] when several apply.
- Source numbers are renumbered for every question, so only cite the sources given with the current question. Never comment on citations in earlier answers.
- Only cite source numbers that exist. Never invent facts, quotes or citations.
- If the sources don't answer the question, say so plainly, then mention what the documents do cover that's closest.
- Be concise and direct. Use Markdown (short paragraphs, lists, tables) when it helps readability. Quote sparingly."""


def pdf_path(doc_id: str) -> Path:
    return settings.upload_dir / f"{doc_id}.pdf"


async def _set(doc_id: str, **fields) -> None:
    async with SessionLocal() as s:
        doc = await s.get(Document, doc_id)
        if doc is None:
            return
        for k, v in fields.items():
            setattr(doc, k, v)
        await s.commit()


async def ingest(doc_id: str) -> None:
    """Parse, chunk, embed and store a PDF. Runs as a background task."""
    try:
        await _set(doc_id, status="processing", stage="reading", progress=0.05, error=None)
        pages = await asyncio.to_thread(extract_pages, pdf_path(doc_id))

        await _set(doc_id, stage="chunking", progress=0.15, pages=len(pages))
        chunks = chunk_pages(pages, settings.chunk_chars, settings.chunk_overlap)
        if not chunks:
            raise ValueError(
                "No selectable text found. This PDF looks scanned or image-only, and OCR isn't supported."
            )

        async with SessionLocal() as s:
            await s.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            await s.commit()

        await _set(doc_id, stage="embedding", progress=0.2)
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120) as client:
            step = openrouter.EMBED_BATCH
            for i in range(0, len(chunks), step):
                batch = [c.content for c in chunks[i : i + step]]
                vectors.extend(await openrouter.embed(batch, client))
                done = min(i + step, len(chunks)) / len(chunks)
                await _set(doc_id, progress=0.2 + 0.75 * done)

        async with SessionLocal() as s:
            s.add_all(
                Chunk(document_id=doc_id, page=c.page, idx=c.idx, content=c.content, embedding=v)
                for c, v in zip(chunks, vectors)
            )
            doc = await s.get(Document, doc_id)
            if doc is None:  # deleted while processing
                await s.rollback()
                return
            doc.status, doc.stage, doc.progress = "ready", None, 1.0
            doc.chunk_count = len(chunks)
            doc.embed_model = settings.embedding_model
            await s.commit()
        log.info("Indexed %s: %d pages, %d chunks", doc_id, len(pages), len(chunks))
    except Exception as e:  # surfaced to the UI via the document's status
        log.exception("Ingest failed for %s", doc_id)
        await _set(doc_id, status="failed", stage=None, error=str(e) or e.__class__.__name__)


async def retrieve(s, query: str, document_id: str | None) -> list[dict]:
    [qvec] = await openrouter.embed([query])
    return await search_by_vector(s, qvec, document_id)


async def search_by_vector(s, qvec: list[float], document_id: str | None) -> list[dict]:
    """Exact cosine search over ready chunks.

    This vsql_vector build rejects bound parameters inside ``COSINE_DISTANCE``
    ("argument must be a custom type or string constant") but accepts a SQL
    string constant, including at 1536 dimensions. The literal is validated
    as numbers only and inlined. The embedding column is not selected: copying
    an ``SVECTOR`` into an internal temporary table fails on this build, and
    ``TEMPORARY`` tables with ``SVECTOR`` columns fail the same way.
    """
    if len(qvec) != settings.embedding_dim:
        raise ValueError(
            f"query vector has {len(qvec)} dimensions, EMBEDDING_DIM is {settings.embedding_dim}"
        )
    literal = vector_literal(qvec)
    sql = (
        "SELECT c.id AS id, c.document_id AS document_id, c.page AS page, "
        "c.content AS content, d.name AS document_name, "
        f"COSINE_DISTANCE(c.embedding, '{literal}') AS distance "
        "FROM chunks c "
        "JOIN documents d ON d.id = c.document_id "
        "WHERE d.status = 'ready'"
    )
    params: dict = {"k": int(settings.top_k)}
    if document_id:
        sql += " AND c.document_id = :document_id"
        params["document_id"] = document_id
    sql += " ORDER BY distance LIMIT :k"
    rows = (await s.execute(text(sql), params)).all()
    return [
        {
            "n": i,
            "chunk_id": row.id,
            "document_id": row.document_id,
            "document_name": row.document_name,
            "page": row.page,
            "content": row.content,
            "score": round(1 - float(row.distance), 4),
        }
        for i, row in enumerate(rows, start=1)
    ]


CITATION = re.compile(r"\s?\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]")


def build_messages(history: list[dict], question: str, sources: list[dict]) -> list[dict]:
    # Earlier answers cite sources from their own turn; strip those markers so the
    # model doesn't confuse them with this turn's numbering.
    history = [
        {**m, "content": CITATION.sub("", m["content"])} if m["role"] == "assistant" else m
        for m in history
    ]
    blocks = "\n\n".join(
        f'<source n="{src["n"]}" document="{src["document_name"]}" page="{src["page"]}">\n'
        f'{src["content"]}\n</source>'
        for src in sources
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Sources:\n\n{blocks}\n\nQuestion: {question}"},
    ]
