from __future__ import annotations

import unicodedata
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class GemmaEmbedder:
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self._model = SentenceTransformer(model_name)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    def embed(self, text: str) -> List[float]:
        text = self._sanitize_text(text).strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        chunk = [(self._sanitize_text(t).strip() or " ") for t in texts]
        vecs = self._model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        return [row.tolist() for row in arr]
