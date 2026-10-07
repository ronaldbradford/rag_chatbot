# Marginalia

Upload PDFs, ask questions, and read the exact passage behind every answer in the margin.

- **Backend:** FastAPI · PyMuPDF · VillageSQL + vsql_vector · OpenRouter (chat + embeddings)
- **Frontend:** React + Vite + TypeScript

## Run it

Requires Docker, [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
# 1. Database (VillageSQL + vsql_vector on port 3306)
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

## Database

`docker compose` runs [`ghcr.io/ronaldbradford/villagesql-server:nightly.2026-09-14-vector`](https://ghcr.io/ronaldbradford/villagesql-server) (also tagged `vector-local`). VillageSQL is a MySQL 8.4 fork. On the first start of an empty data directory the image enables preview extensions and runs `INSTALL EXTENSION vsql_vector`.

Vectors are `SVECTOR(N)` columns, up to 3072 dimensions, so the default 1536-dimension embedding fits. This build of `vsql_vector` has no HNSW index; each question is an exact cosine scan, which is enough for a single-user library.

If that registry image can't be pulled, build [villagesql-server](https://github.com/villagesql/villagesql-server) and [vsql-vector](https://github.com/villagesql/vsql-vector) yourself and tag the result `ghcr.io/ronaldbradford/villagesql-server:nightly.2026-09-14-vector`. This compose file stays pointed at that name.

Without an OpenRouter key, `cd backend && uv run python scripts/check_retrieval.py` inserts synthetic 1536-dimension vectors and checks top-k ordering through the app's retrieval code. It drops the Marginalia tables in `DATABASE_URL` first.

## Configuration (`backend/.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | https://openrouter.ai/keys |
| `CHAT_MODEL` | `anthropic/claude-sonnet-5.5` | Any OpenRouter chat model ID |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Any OpenRouter embedding model ID |
| `EMBEDDING_DIM` | `1536` | Must match the embedding model's vector size. Max 3072. |
| `DATABASE_URL` | `mysql+aiomysql://rag:rag@127.0.0.1:3306/rag?charset=utf8mb4` | VillageSQL |

Restart the backend after editing `.env`.

**Switching the embedding model:** existing documents are marked *Needs re-index* on the shelf. Click **Re-index** on each one. If the vector size changes too, update `EMBEDDING_DIM`; stored chunks are cleared on startup and the `SVECTOR` column is recreated at the new size.

## How it works

1. **Upload** stores the PDF in `backend/data/uploads/`. A background task extracts text page by page, splits it into ~1800-character passages tagged with their page, embeds them in batches, and writes them to `chunks` (`SVECTOR`, cosine distance, exact scan).
2. **Ask** embeds the question (folded together with the previous one so follow-ups retrieve well), pulls the top 8 passages from one document or all of them, and streams the model's answer over SSE. The model cites passages as `[n]`.
3. **Cite:** each `[n]` becomes a footnote mark. Clicking it opens the passage in the margin with a highlighter stroke, plus a link that opens the PDF at that page.
4. **Look:** Graphite Glass, frosted macOS-style panels over a stone wallpaper with a red accent. All colors are tokens at the top of `frontend/src/index.css`.
5. **History:** conversations, messages and the passages each answer used are stored in VillageSQL, so citations still work after a reload.

## Limits

- Scanned or image-only PDFs have no extractable text and are rejected (no OCR).
- Single user, no authentication.
- `vsql_vector` on this image cannot change an `SVECTOR` column's dimension with `MODIFY` (the column is dropped and re-added), and it has no approximate-search index.
