import logging
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings
from .models import Base

log = logging.getLogger("marginalia")

settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"init_command": "SET time_zone = '+00:00'"},
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_SVECTOR_DIM = re.compile(r"`(?P<column>[^`]+)`\s+(?:[\w.]+\.)?SVECTOR\((?P<dim>\d+)\)", re.I)


async def get_session():
    async with SessionLocal() as session:
        yield session


def _dim_from_ddl(ddl: str, column: str) -> int | None:
    for match in _SVECTOR_DIM.finditer(ddl):
        if match.group("column") == column:
            return int(match.group("dim"))
    return None


async def _autocommit():
    """DDL on VillageSQL implicitly commits; don't wrap startup in one transaction."""
    conn = await engine.connect()
    await conn.execution_options(isolation_level="AUTOCOMMIT")
    return conn


async def init_db() -> None:
    dim = settings.embedding_dim
    conn = await _autocommit()
    try:
        version = (await conn.execute(text("SELECT VERSION()"))).scalar()
        try:
            max_dim = int((await conn.execute(text("SELECT VECTOR_MAX_DIMENSION()"))).scalar())
        except Exception as exc:
            raise RuntimeError(
                "vsql_vector is not installed. Use the VillageSQL image in docker-compose "
                "(first start runs INSTALL EXTENSION vsql_vector) and restart the backend."
            ) from exc
        if not 1 <= dim <= max_dim:
            raise RuntimeError(
                f"EMBEDDING_DIM={dim} is outside the range vsql_vector supports (1..{max_dim})."
            )

        ext = (
            await conn.execute(
                text(
                    "SELECT EXTENSION_VERSION FROM INFORMATION_SCHEMA.EXTENSIONS "
                    "WHERE EXTENSION_NAME = 'vsql_vector'"
                )
            )
        ).scalar()
        log.info("Connected to %s (vsql_vector %s, max dim %s)", version, ext, max_dim)

        await conn.run_sync(Base.metadata.create_all)
        await _align_embedding_dim(conn, dim)

        # Documents embedded with another model can't be compared with new queries.
        # utf8mb4_0900_ai_ci is case-insensitive; model ids are case-sensitive.
        await conn.execute(
            text(
                "UPDATE documents SET status = 'stale' "
                "WHERE status = 'ready' AND NOT (embed_model COLLATE utf8mb4_bin <=> :m)"
            ),
            {"m": settings.embedding_model},
        )
        # Anything left mid-processing by a crash or restart can't resume.
        await conn.execute(
            text(
                "UPDATE documents SET status = 'failed', "
                "error = 'Processing was interrupted. Re-index to try again.' "
                "WHERE status = 'processing'"
            )
        )
    finally:
        await conn.close()


async def _align_embedding_dim(conn, dim: int) -> None:
    """Recreate ``chunks.embedding`` when EMBEDDING_DIM no longer matches.

    ``MODIFY COLUMN`` cannot convert ``SVECTOR(N)`` to ``SVECTOR(M)`` on this
    build, even after the table is empty. Drop and add the column instead.
    ``INFORMATION_SCHEMA.COLUMNS.COLUMN_TYPE`` also omits the dimension, so the
    declared width is read from ``SHOW CREATE TABLE``.
    """
    ddl = (
        await conn.execute(text("SHOW CREATE TABLE chunks"))
    ).one()[1]
    current = _dim_from_ddl(ddl, "embedding")
    if current is None:
        raise RuntimeError(f"chunks.embedding is not an SVECTOR column:\n{ddl}")
    if current == dim:
        return

    log.warning(
        "EMBEDDING_DIM changed from %s to %s: clearing stored vectors. "
        "Re-index documents from the UI.",
        current,
        dim,
    )
    await conn.execute(text("DELETE FROM chunks"))
    await conn.execute(text("ALTER TABLE chunks DROP COLUMN embedding"))
    await conn.execute(text(f"ALTER TABLE chunks ADD COLUMN embedding SVECTOR({int(dim)}) NOT NULL"))
    await conn.execute(
        text("UPDATE documents SET status = 'stale', chunk_count = 0 WHERE status = 'ready'")
    )
