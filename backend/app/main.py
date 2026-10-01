import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import openrouter, rag
from .config import get_settings
from .db import SessionLocal, get_session, init_db
from .models import Chunk, Conversation, Document, Message

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield


app = FastAPI(title="Marginalia", lifespan=lifespan)


# ---------- serializers ----------


def doc_out(d: Document) -> dict:
    return {
        "id": d.id,
        "name": d.name,
        "size_bytes": d.size_bytes,
        "pages": d.pages,
        "chunk_count": d.chunk_count,
        "status": d.status,
        "stage": d.stage,
        "progress": d.progress,
        "error": d.error,
        "created_at": d.created_at.isoformat(),
    }


def conv_out(c: Conversation) -> dict:
    return {
        "id": c.id,
        "title": c.title,
        "document_id": c.document_id,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def msg_out(m: Message) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "citations": m.citations or [],
        "created_at": m.created_at.isoformat(),
    }


# ---------- meta ----------


@app.get("/api/health")
async def health():
    return {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "api_key_set": bool(settings.openrouter_api_key),
    }


# ---------- documents ----------


@app.get("/api/documents")
async def list_documents(s: AsyncSession = Depends(get_session)):
    docs = (await s.scalars(select(Document).order_by(Document.created_at.desc()))).all()
    return [doc_out(d) for d in docs]


@app.post("/api/documents", status_code=201)
async def upload_document(
    file: UploadFile, tasks: BackgroundTasks, s: AsyncSession = Depends(get_session)
):
    name = file.filename or "document.pdf"
    if not name.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"PDFs must be under {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, f"{name} isn't a valid PDF file.")

    doc = Document(name=name, size_bytes=len(data), stage="queued")
    s.add(doc)
    await s.commit()
    rag.pdf_path(doc.id).write_bytes(data)
    tasks.add_task(rag.ingest, doc.id)
    return doc_out(doc)


async def _get_doc(s: AsyncSession, doc_id: str) -> Document:
    doc = await s.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


@app.post("/api/documents/{doc_id}/reindex")
async def reindex_document(doc_id: str, tasks: BackgroundTasks, s: AsyncSession = Depends(get_session)):
    doc = await _get_doc(s, doc_id)
    if doc.status == "processing":
        raise HTTPException(409, "This document is already being processed.")
    if not rag.pdf_path(doc_id).exists():
        raise HTTPException(410, "The original PDF is missing. Delete this document and upload it again.")
    doc.status, doc.stage, doc.progress, doc.error = "processing", "queued", 0.0, None
    await s.commit()
    tasks.add_task(rag.ingest, doc_id)
    return doc_out(doc)


@app.delete("/api/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, s: AsyncSession = Depends(get_session)):
    doc = await _get_doc(s, doc_id)
    await s.execute(delete(Chunk).where(Chunk.document_id == doc_id))
    await s.delete(doc)
    await s.commit()
    rag.pdf_path(doc_id).unlink(missing_ok=True)


@app.get("/api/documents/{doc_id}/file")
async def document_file(doc_id: str, s: AsyncSession = Depends(get_session)):
    doc = await _get_doc(s, doc_id)
    path = rag.pdf_path(doc_id)
    if not path.exists():
        raise HTTPException(410, "The original PDF is missing.")
    return FileResponse(path, media_type="application/pdf", filename=doc.name, content_disposition_type="inline")


# ---------- conversations ----------


class NewConversation(BaseModel):
    document_id: str | None = None


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


@app.get("/api/conversations")
async def list_conversations(s: AsyncSession = Depends(get_session)):
    convs = (await s.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))).all()
    return [conv_out(c) for c in convs]


@app.post("/api/conversations", status_code=201)
async def create_conversation(body: NewConversation, s: AsyncSession = Depends(get_session)):
    if body.document_id:
        await _get_doc(s, body.document_id)
    conv = Conversation(document_id=body.document_id)
    s.add(conv)
    await s.commit()
    return conv_out(conv)


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, s: AsyncSession = Depends(get_session)):
    conv = await s.get(Conversation, conv_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found.")
    msgs = (
        await s.scalars(select(Message).where(Message.conversation_id == conv_id).order_by(Message.id))
    ).all()
    return {**conv_out(conv), "messages": [msg_out(m) for m in msgs]}


@app.delete("/api/conversations/{conv_id}", status_code=204)
async def delete_conversation(conv_id: str, s: AsyncSession = Depends(get_session)):
    conv = await s.get(Conversation, conv_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found.")
    await s.delete(conv)
    await s.commit()


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


NOTHING_INDEXED = (
    "There's nothing to search yet. Upload a PDF, wait for it to finish indexing, then ask again."
)


@app.post("/api/conversations/{conv_id}/messages")
async def send_message(conv_id: str, body: NewMessage):
    question = body.content.strip()

    async def stream():
        async with SessionLocal() as s:
            conv = await s.get(Conversation, conv_id)
            if conv is None:
                yield sse("error", {"message": "Conversation not found."})
                return

            prior = (
                await s.scalars(
                    select(Message)
                    .where(Message.conversation_id == conv_id)
                    .order_by(Message.id.desc())
                    .limit(settings.history_turns * 2)
                )
            ).all()[::-1]
            history = [{"role": m.role, "content": m.content} for m in prior]

            user_msg = Message(conversation_id=conv_id, role="user", content=question)
            s.add(user_msg)
            if not prior:
                conv.title = question[:80] + ("…" if len(question) > 80 else "")
            conv.updated_at = datetime.now(timezone.utc)
            await s.commit()
            yield sse("conversation", conv_out(conv))

            answer = ""
            sources: list[dict] = []
            try:
                # Fold the previous question in so follow-ups ("and why?") still retrieve well.
                last_q = next((m.content for m in reversed(prior) if m.role == "user"), "")
                query = f"{last_q}\n{question}" if last_q else question
                sources = await rag.retrieve(s, query, conv.document_id)
                yield sse("sources", sources)

                if not sources:
                    answer = NOTHING_INDEXED
                    yield sse("token", answer)
                else:
                    async for token in openrouter.stream_chat(rag.build_messages(history, question, sources)):
                        answer += token
                        yield sse("token", token)
                yield sse("done", {})
            except openrouter.OpenRouterError as e:
                yield sse("error", {"message": str(e)})
            except Exception as e:
                logging.getLogger("marginalia").exception("Answer failed")
                yield sse("error", {"message": f"Something went wrong while answering: {e}"})
            finally:
                if answer:
                    s.add(Message(conversation_id=conv_id, role="assistant", content=answer, citations=sources))
                    await asyncio.shield(s.commit())

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
