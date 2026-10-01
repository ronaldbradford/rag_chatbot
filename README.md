# Marginalia

Upload PDFs, ask questions, and read the exact passage behind every answer in the margin.

- **Backend:** FastAPI · PyMuPDF · Postgres 17 + pgvector · OpenRouter (chat + embeddings)
- **Frontend:** React + Vite + TypeScript

## Run it

Requires Docker, [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
# 1. Database (Postgres + pgvector on port 5433)
docker compose up -d

# 2. Backend on :8000
cd backend
cp .env.example .env        # then add your OPENROUTER_API_KEY
uv run uvicorn app.main:app --port 8000 --reload

# 3. Frontend on :5173 (proxies /api to the backend)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and drop a PDF anywhere on the page.

## Configuration (`backend/.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | https://openrouter.ai/keys |
| `CHAT_MODEL` | `anthropic/claude-sonnet-5.5` | Any OpenRouter chat model ID |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Any OpenRouter embedding model ID |
| `EMBEDDING_DIM` | `1536` | Must match the embedding model's vector size |
| `DATABASE_URL` | `postgresql+asyncpg://rag:rag@localhost:5433/rag` | |

Restart the backend after editing `.env`.

**Switching the embedding model:** existing documents are marked *Needs re-index* on the shelf. Click **Re-index** on each one. If the vector size changes too, update `EMBEDDING_DIM`; stored vectors are cleared on startup and the column is resized. Sizes above 2000 dims work but skip the HNSW index.

## How it works

1. **Upload** stores the PDF in `backend/data/uploads/`. A background task extracts text page by page, splits it into ~1800-character passages tagged with their page, embeds them in batches, and writes them to `chunks` (pgvector, HNSW cosine index).
2. **Ask** embeds the question (folded together with the previous one so follow-ups retrieve well), pulls the top 8 passages from one document or all of them, and streams the model's answer over SSE. The model cites passages as `[n]`.
3. **Cite:** each `[n]` becomes a footnote mark. Clicking it opens the passage in the margin with a highlighter stroke, plus a link that opens the PDF at that page.
4. **Look:** Graphite Glass, frosted macOS-style panels over a stone wallpaper with a red accent. All colors are tokens at the top of `frontend/src/index.css`.
5. **History:** conversations, messages and the passages each answer used are stored in Postgres, so citations still work after a reload.

## Limits

- Scanned or image-only PDFs have no extractable text and are rejected (no OCR).
- Single user, no authentication.
