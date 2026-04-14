from __future__ import annotations

import unicodedata
from typing import List, Optional

from openai import OpenAI

from mcp_router.config import RouterConfig, openai_api_key


class OpenAIEmbedder:
    def __init__(self, cfg: RouterConfig, api_key: Optional[str] = None):
        key = api_key or openai_api_key(cfg)
        if not key:
            raise ValueError(
                f"Missing API key: set {cfg.openai_api_key_env} in the environment."
            )
        self._client = OpenAI(api_key=key)
        self._model = cfg.embedding_model

    @staticmethod
    def _sanitize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    def embed(self, text: str) -> List[float]:
        text = self._sanitize_text(text).strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = [self._sanitize_text(t).strip() or " " for t in texts[i : i + batch_size]]
            resp = self._client.embeddings.create(model=self._model, input=chunk)
            # Preserve order
            by_idx = {d.index: d.embedding for d in resp.data}
            for j in range(len(chunk)):
                out.append(list(by_idx[j]))
        return out
