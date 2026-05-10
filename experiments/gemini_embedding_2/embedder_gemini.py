from __future__ import annotations

import os
import time
import unicodedata
from typing import List

import numpy as np
import requests


class GeminiEmbedder:
    def __init__(
        self,
        model_name: str = "google/gemini-embedding-2-preview",
        api_key_env: str = "OPENROUTER_API_KEY",
        min_request_interval_s: float = 0.70,
        endpoint: str = "https://openrouter.ai/api/v1/embeddings",
        dimensions: int = 3072,
    ):
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"{api_key_env} is not set")
        self._model_name = model_name
        self._api_key = api_key
        self._endpoint = endpoint
        self._dimensions = int(dimensions)
        self._min_request_interval_s = float(min_request_interval_s)
        self._last_request_ts = 0.0

    @staticmethod
    def _sanitize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

    def _embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        # Rate limit: keep API requests under ~100/min. Note that /route does 2
        # embeddings (server_intent + tool_intent), so we throttle per request.
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_request_interval_s:
            time.sleep(self._min_request_interval_s - elapsed)

        payload = {
            "model": self._model_name,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self._dimensions,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response_data = None
        for attempt in range(8):
            try:
                resp = requests.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code >= 400:
                    msg = f"{resp.status_code} {resp.text}"
                    if "429" not in msg:
                        raise RuntimeError(f"OpenRouter embed failed: {msg}")
                    if attempt == 7:
                        raise RuntimeError(f"OpenRouter embed failed after retries: {msg}")
                    time.sleep(12)
                    continue
                response_data = resp.json()
                data = response_data.get("data") or []
                if data:
                    self._last_request_ts = time.monotonic()
                    break
                if attempt == 7:
                    raise RuntimeError(f"OpenRouter embedding response had no data: {response_data}")
                time.sleep(6)
                continue
                self._last_request_ts = time.monotonic()
            except requests.RequestException as exc:
                if attempt == 7:
                    raise
                time.sleep(12)
        if response_data is None:
            raise ValueError("OpenRouter embedding response was empty")

        data = response_data.get("data") or []
        if not data:
            raise ValueError(f"OpenRouter embedding response had no data: {response_data}")
        return [np.asarray(item["embedding"], dtype=np.float32).tolist() for item in data]

    def embed(self, text: str) -> List[float]:
        text = self._sanitize_text(text).strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        return self._embed_texts([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        chunk = [(self._sanitize_text(t).strip() or " ") for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(chunk), batch_size):
            out.extend(self._embed_texts(chunk[i : i + batch_size]))
        return out
