#!/usr/bin/env python3
"""Check VillageSQL schema creation and top-k retrieval without OpenRouter.

Drops the Marginalia tables in DATABASE_URL (the dev database from
docker-compose, expected name ``rag``), recreates them, inserts synthetic
1536-dimension vectors, and checks ordering through ``rag.retrieve``. Then
uploads a tiny PDF through the API with the embedder stubbed out.

    cd backend && uv run python scripts/check_retrieval.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app import openrouter, rag
from app.config import get_settings
from app.db import SessionLocal, engine, init_db
from app.main import app
from app.models import Chunk, Document
from app.sqltypes import vector_literal

settings = get_settings()
DIM = settings.embedding_dim


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok: {message}")


def unit(angle: float) -> list[float]:
    values = [0.0] * DIM
    values[0] = math.cos(angle)
    values[1] = math.sin(angle)
    return values


async def reset_and_init() -> str:
    async with engine.connect() as conn:
        autocommit = await conn.execution_options(isolation_level="AUTOCOMMIT")
        version = (await autocommit.execute(text("SELECT VERSION()"))).scalar()
        print(f"VERSION() = {version}")
        database = (await autocommit.execute(text("SELECT DATABASE()"))).scalar()
        if database != "rag":
            raise SystemExit(f"refusing to reset database {database!r}; this script expects 'rag'")
        await autocommit.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in ("messages", "conversations", "chunks", "documents"):
            await autocommit.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
        await autocommit.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    await init_db()
    return str(version)


async def transaction_check() -> None:
    async with SessionLocal() as session:
        doc = Document(name="rollback.pdf", status="ready", embed_model=settings.embedding_model)
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        await session.rollback()
    async with SessionLocal() as session:
        expect(await session.get(Document, doc_id) is None, "a rolled-back insert is not committed")


async def schema_checks() -> None:
    async with engine.connect() as conn:
        ext = (
            await conn.execute(
                text(
                    "SELECT EXTENSION_VERSION FROM INFORMATION_SCHEMA.EXTENSIONS "
                    "WHERE EXTENSION_NAME = 'vsql_vector'"
                )
            )
        ).scalar()
        expect(ext is not None and str(ext) != "", f"vsql_vector is installed (version {ext})")
        max_dim = int((await conn.execute(text("SELECT VECTOR_MAX_DIMENSION()"))).scalar())
        expect(max_dim >= DIM, f"VECTOR_MAX_DIMENSION() {max_dim} fits EMBEDDING_DIM {DIM}")

        ddl = (await conn.execute(text("SHOW CREATE TABLE chunks"))).one()[1]
        expect(f"SVECTOR({DIM})" in ddl, f"chunks.embedding is SVECTOR({DIM})")
        expect("ON DELETE CASCADE" in ddl, "chunks.document_id cascades on delete")
        messages = (await conn.execute(text("SHOW CREATE TABLE messages"))).one()[1]
        expect("json" in messages.lower(), "messages.citations is JSON")
        expect("datetime(6)" in messages.lower(), "timestamps keep microsecond precision")
        conversations = (await conn.execute(text("SHOW CREATE TABLE conversations"))).one()[1]
        expect("ON DELETE CASCADE" in conversations, "conversations cascade when a document is deleted")


async def seed_library() -> tuple[str, str, list[str], str]:
    """Two documents. Doc A has top_k+2 chunks at increasing angles; doc B sits between A0 and A1."""
    query = unit(0.0)
    doc_a_angles = [i * 0.05 for i in range(settings.top_k + 2)]
    doc_b_angle = 0.03

    async with SessionLocal() as session:
        doc_a = Document(name="alpha.pdf", status="ready", embed_model=settings.embedding_model, chunk_count=len(doc_a_angles))
        doc_b = Document(name="beta.pdf", status="ready", embed_model=settings.embedding_model, chunk_count=1)
        session.add_all([doc_a, doc_b])
        await session.flush()

        contents: list[str] = []
        for i, angle in enumerate(doc_a_angles):
            content = f"alpha-{i} café — 東京"
            contents.append(content)
            session.add(
                Chunk(document_id=doc_a.id, page=i + 1, idx=i, content=content, embedding=unit(angle))
            )
        beta = "beta-passage"
        session.add(Chunk(document_id=doc_b.id, page=1, idx=0, content=beta, embedding=unit(doc_b_angle)))
        await session.commit()
        doc_a_id, doc_b_id = doc_a.id, doc_b.id

        loaded = await session.get(Chunk, 1)
        expect(loaded is not None and loaded.content.startswith("alpha-0"), "ORM reloads chunk text")
        expect(loaded is not None and "東京" in loaded.content, "utf8mb4 round-trips non-ASCII chunk text")
        expect(loaded is not None and len(loaded.embedding) == DIM, f"ORM reloads a {DIM}-d embedding")
        expect(loaded is not None and abs(loaded.embedding[0] - 1.0) < 1e-5, "stored vector keeps its first component")

    return doc_a_id, doc_b_id, contents, beta


async def retrieval_checks(doc_a_id: str, doc_b_id: str, contents: list[str], beta: str) -> None:
    async def fake_embed(texts, client=None):
        expect(texts == ["synthetic-query"], f"retrieve embeds the question text (got {texts!r})")
        return [unit(0.0)]

    original = openrouter.embed
    openrouter.embed = fake_embed
    try:
        async with SessionLocal() as session:
            all_hits = await rag.retrieve(session, "synthetic-query", None)
            alpha_hits = await rag.retrieve(session, "synthetic-query", doc_a_id)
            beta_hits = await rag.retrieve(session, "synthetic-query", doc_b_id)
    finally:
        openrouter.embed = original

    expect(len(all_hits) == settings.top_k, f"unfiltered search returns top_k={settings.top_k}")
    expect(all_hits[0]["content"] == contents[0], "nearest passage is alpha-0")
    expect(all_hits[0]["score"] == 1.0, f"identical vector scores 1 (got {all_hits[0]['score']})")
    expect(all_hits[1]["content"] == beta, "second hit is the closer beta passage, ahead of alpha-1")
    expect([hit["content"] for hit in all_hits[2:]] == contents[1 : settings.top_k - 1], "remaining hits follow angle order")
    expect(all(all_hits[i]["score"] >= all_hits[i + 1]["score"] for i in range(len(all_hits) - 1)), "scores are sorted")
    expect(contents[-1] not in {hit["content"] for hit in all_hits}, "farthest alpha passage is outside top-k")

    expect([hit["content"] for hit in alpha_hits] == contents[: settings.top_k], "document filter keeps that document's nearest chunks")
    expect(all(hit["document_id"] == doc_a_id for hit in alpha_hits), "filtered hits all belong to alpha")
    expect(len(beta_hits) == 1 and beta_hits[0]["document_id"] == doc_b_id, "single-document search returns only beta")
    expect(beta_hits[0]["page"] == 1 and beta_hits[0]["n"] == 1, "citation numbers restart at 1")


async def lifecycle_checks() -> None:
    async with SessionLocal() as session:
        other = Document(name="other-model.pdf", status="ready", embed_model="other/embed", chunk_count=4)
        busy = Document(name="interrupted.pdf", status="processing", stage="embedding", progress=0.4)
        session.add_all([other, busy])
        await session.commit()
        other_id, busy_id = other.id, busy.id

    await init_db()

    async with SessionLocal() as session:
        other = await session.get(Document, other_id)
        busy = await session.get(Document, busy_id)
        expect(other is not None and other.status == "stale", "a different embed_model marks a ready document stale")
        expect(other is not None and other.chunk_count == 4, "model mismatch keeps stored chunks")
        expect(
            busy is not None and busy.status == "failed" and busy.error and "interrupted" in busy.error,
            "a document left processing is marked failed",
        )

    # MODIFY cannot change SVECTOR(N) to SVECTOR(M); init_db drops and re-adds the column.
    async with engine.connect() as conn:
        autocommit = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await autocommit.execute(text("DELETE FROM chunks"))
        await autocommit.execute(text("ALTER TABLE chunks DROP COLUMN embedding"))
        await autocommit.execute(text("ALTER TABLE chunks ADD COLUMN embedding SVECTOR(32) NOT NULL"))
        ddl = (await autocommit.execute(text("SHOW CREATE TABLE chunks"))).one()[1]
        expect("SVECTOR(32)" in ddl, "test setup resized the column to SVECTOR(32)")

    async with SessionLocal() as session:
        ready = Document(name="needs-resize.pdf", status="ready", embed_model=settings.embedding_model, chunk_count=9)
        session.add(ready)
        await session.commit()
        ready_id = ready.id

    await init_db()

    async with engine.connect() as conn:
        ddl = (await conn.execute(text("SHOW CREATE TABLE chunks"))).one()[1]
        expect(f"SVECTOR({DIM})" in ddl, f"startup resized the column back to SVECTOR({DIM})")
    async with SessionLocal() as session:
        ready = await session.get(Document, ready_id)
        chunks = (await session.scalars(select(Chunk))).all()
        expect(ready is not None and ready.status == "stale" and ready.chunk_count == 0, "dimension change marks documents stale and clears chunk_count")
        expect(chunks == [], "dimension change deletes stored chunks")


async def placeholder_probe() -> None:
    """Server-side placeholders inside COSINE_DISTANCE are rejected; record which one we hit."""
    async with engine.connect() as conn:
        try:
            await conn.execute(
                text("PREPARE marg_cos FROM 'SELECT COSINE_DISTANCE(embedding, ?) AS distance FROM chunks LIMIT 1'")
            )
            await conn.execute(text("SET @qv = :literal"), {"literal": vector_literal(unit(0.0))})
            await conn.execute(text("EXECUTE marg_cos USING @qv"))
            print("NOTE: server-side placeholder inside COSINE_DISTANCE was accepted")
        except Exception as exc:
            message = str(exc)
            expect(
                "string constant" in message or "3219" in message,
                f"server-side placeholder rejected as expected ({message.splitlines()[0]})",
            )
            print(f"NOTE: server-side placeholder rejected: {message.splitlines()[0]}")
        finally:
            try:
                await conn.execute(text("DEALLOCATE PREPARE marg_cos"))
            except Exception:
                pass
            await conn.rollback()


def _pdf() -> bytes:
    doc = pymupdf.open()
    first = doc.new_page()
    first.insert_text((72, 72), "Paris is the capital of France.\n" * 12)
    second = doc.new_page()
    second.insert_text((72, 72), "Tokyo is the capital of Japan.\n" * 12)
    data = doc.tobytes()
    doc.close()
    return data


def _events(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event = "message"
    data: list[str] = []
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].strip())
        elif line == "" and data:
            events.append((event, "\n".join(data)))
            event, data = "message", []
    if data:
        events.append((event, "\n".join(data)))
    return events


def api_checks() -> None:
    async def fake_embed(texts, client=None):
        vectors = []
        for text_value in texts:
            vector = [0.0] * DIM
            lowered = text_value.lower()
            if "paris" in lowered:
                vector[0] = 1.0
            elif "tokyo" in lowered:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            vectors.append(vector)
        return vectors

    async def fake_chat(messages):
        yield "Paris is the capital of France [1]."

    openrouter.embed = fake_embed
    openrouter.stream_chat = fake_chat

    with TestClient(app) as client:
        health = client.get("/api/health")
        expect(health.status_code == 200, f"health endpoint responds ({health.status_code})")
        expect(health.json()["embedding_model"] == settings.embedding_model, "health reports the embedding model")

        uploaded = client.post(
            "/api/documents",
            files={"file": ("notes.pdf", _pdf(), "application/pdf")},
        )
        expect(uploaded.status_code == 201, f"PDF upload accepted ({uploaded.status_code} {uploaded.text})")
        doc_id = uploaded.json()["id"]
        expect("T" in uploaded.json()["created_at"] and uploaded.json()["created_at"].endswith("+00:00"), "created_at is UTC")

        listing = client.get("/api/documents").json()
        doc = next(item for item in listing if item["id"] == doc_id)
        expect(doc["status"] == "ready", f"ingest stored embeddings (status={doc['status']} error={doc['error']})")
        expect(doc["pages"] == 2 and doc["chunk_count"] >= 2, f"PDF was chunked (pages={doc['pages']} chunks={doc['chunk_count']})")

        reindexed = client.post(f"/api/documents/{doc_id}/reindex")
        expect(reindexed.status_code == 200, "re-index accepted")
        doc = next(item for item in client.get("/api/documents").json() if item["id"] == doc_id)
        expect(doc["status"] == "ready" and doc["chunk_count"] >= 2, "re-index embedded the PDF again")

        created = client.post("/api/conversations", json={"document_id": doc_id})
        expect(created.status_code == 201, "conversation created against one document")
        conv_id = created.json()["id"]

        with client.stream(
            "POST",
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Where is Paris?"},
        ) as response:
            expect(response.status_code == 200, f"message stream opened ({response.status_code})")
            body = response.read().decode()

        import json

        parsed = [(name, json.loads(data)) for name, data in _events(body)]
        sources = next(payload for name, payload in parsed if name == "sources")
        tokens = "".join(payload for name, payload in parsed if name == "token")
        expect(sources and "Paris" in sources[0]["content"], "top passage for the question is the Paris page")
        expect(sources[0]["score"] == 1.0 and sources[0]["page"] == 1, "Paris passage is page 1 with score 1")
        expect(all("Tokyo" not in hit["content"] or hit["score"] < sources[0]["score"] for hit in sources), "Tokyo ranks below Paris")
        expect("Paris is the capital of France [1]." in tokens, "answer tokens stream")
        expect(any(name == "done" for name, _payload in parsed), "stream finishes with done")

        saved = client.get(f"/api/conversations/{conv_id}").json()
        expect(saved["title"].startswith("Where is Paris"), "first question becomes the conversation title")
        roles = [message["role"] for message in saved["messages"]]
        expect(roles == ["user", "assistant"], f"user and assistant messages were stored ({roles})")
        assistant = saved["messages"][1]
        expect(assistant["citations"][0]["content"] == sources[0]["content"], "citation snapshot was persisted")
        expect(assistant["citations"][0]["document_name"] == "notes.pdf", "citation keeps the document name")

        everywhere = client.post("/api/conversations", json={})
        expect(everywhere.status_code == 201 and everywhere.json()["document_id"] is None, "all-documents conversation created")

        deleted = client.delete(f"/api/documents/{doc_id}")
        expect(deleted.status_code == 204, "document delete succeeds")
        expect(client.get("/api/documents").json() == [], "document list is empty")
        missing = client.get(f"/api/conversations/{conv_id}")
        expect(missing.status_code == 404, "deleting a document cascades to its conversations")
        removed = client.delete(f"/api/conversations/{everywhere.json()['id']}")
        expect(removed.status_code == 204, "all-documents conversation can be deleted")


async def main_async() -> None:
    _ = vector_literal(unit(0.0))  # fail fast if the literal helper rejects a real vector
    try:
        vector_literal([math.nan])
    except ValueError:
        print("ok: non-finite embeddings are rejected")
    else:
        raise SystemExit("FAIL: non-finite embeddings were accepted")

    version = await reset_and_init()
    expect("villagesql" in version.lower(), f"server is VillageSQL (got {version})")
    await schema_checks()
    await transaction_check()
    ids = await seed_library()
    await retrieval_checks(*ids)
    await placeholder_probe()
    await lifecycle_checks()

    async with engine.connect() as conn:
        autocommit = await conn.execution_options(isolation_level="AUTOCOMMIT")
        for table in ("messages", "conversations", "chunks", "documents"):
            await autocommit.execute(text(f"DELETE FROM `{table}`"))
    await engine.dispose()


def main() -> None:
    asyncio.run(main_async())
    api_checks()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
