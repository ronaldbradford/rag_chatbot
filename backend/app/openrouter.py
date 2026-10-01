import json
from collections.abc import AsyncIterator

import httpx

from .config import get_settings

settings = get_settings()

EMBED_BATCH = 64


class OpenRouterError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not settings.openrouter_api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set. Add it to backend/.env and restart.")
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "X-Title": "Marginalia",
        "HTTP-Referer": "http://localhost:5173",
    }


def _error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return err.get("message") or resp.text
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


async def embed(texts: list[str], client: httpx.AsyncClient | None = None) -> list[list[float]]:
    own = client is None
    client = client or httpx.AsyncClient(timeout=120)
    try:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            resp = await client.post(
                f"{settings.openrouter_base_url}/embeddings",
                headers=_headers(),
                json={"model": settings.embedding_model, "input": batch},
            )
            if resp.status_code != 200:
                raise OpenRouterError(f"Embedding request failed: {_error_message(resp)}")
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors = [d["embedding"] for d in data]
            if vectors and len(vectors[0]) != settings.embedding_dim:
                raise OpenRouterError(
                    f"{settings.embedding_model} returns {len(vectors[0])}-dim vectors but "
                    f"EMBEDDING_DIM is {settings.embedding_dim}. Update backend/.env."
                )
            out.extend(vectors)
        return out
    finally:
        if own:
            await client.aclose()


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    payload = {"model": settings.chat_model, "messages": messages, "stream": True}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, read=300)) as client:
        async with client.stream(
            "POST",
            f"{settings.openrouter_base_url}/chat/completions",
            headers=_headers(),
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise OpenRouterError(f"Chat request failed: {_error_message(resp)}")
            async for line in resp.aiter_lines():
                # OpenRouter sends ": OPENROUTER PROCESSING" keep-alive comments.
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    raise OpenRouterError(event["error"].get("message", "Model stream failed"))
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
