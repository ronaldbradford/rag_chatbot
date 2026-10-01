from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Change these in backend/.env and restart the server.
    chat_model: str = "anthropic/claude-sonnet-5.5"
    embedding_model: str = "openai/text-embedding-3-small"
    # Must match the embedding model's output size. Changing it wipes stored
    # vectors on next startup; re-index documents from the UI afterwards.
    embedding_dim: int = 1536

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    max_upload_mb: int = 50

    chunk_chars: int = 1800
    chunk_overlap: int = 250
    top_k: int = 8
    history_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
