import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings
from .models import Base

log = logging.getLogger("marginalia")

settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# pgvector HNSW indexes support at most 2000 dimensions for `vector`.
HNSW_MAX_DIM = 2000


async def get_session():
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    dim = settings.embedding_dim
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        current_dim = (
            await conn.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
                )
            )
        ).scalar()

        if current_dim is not None and current_dim != dim:
            log.warning(
                "EMBEDDING_DIM changed from %s to %s: clearing stored vectors. "
                "Re-index documents from the UI.",
                current_dim,
                dim,
            )
            await conn.execute(text("DROP INDEX IF EXISTS chunks_embedding_hnsw"))
            await conn.execute(text("DELETE FROM chunks"))
            await conn.execute(text(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dim})"))
            await conn.execute(
                text("UPDATE documents SET status = 'stale', chunk_count = 0 WHERE status = 'ready'")
            )

        # Documents embedded with another model can't be compared with new queries.
        await conn.execute(
            text(
                "UPDATE documents SET status = 'stale' "
                "WHERE status = 'ready' AND embed_model IS DISTINCT FROM :m"
            ),
            {"m": settings.embedding_model},
        )
        # Anything left mid-processing by a crash or restart can't resume.
        await conn.execute(
            text(
                "UPDATE documents SET status = 'failed', error = 'Processing was interrupted. Re-index to try again.' "
                "WHERE status = 'processing'"
            )
        )

        if dim <= HNSW_MAX_DIM:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
        else:
            log.warning("EMBEDDING_DIM %s > %s: skipping HNSW index (exact search).", dim, HNSW_MAX_DIM)
