from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class SpladeLiteRetriever:
    """
    Lightweight SPLADE-style sparse retriever approximation.

    Uses sparse TF-IDF with uni/bi-grams + log-scaled tf as a low-latency
    lexical-expansion stage. This is not full SPLADE training; it's a practical
    sparse stage that scales well and complements dense retrieval.
    """

    def __init__(self, documents: list[str]):
        self._docs = documents
        if not documents:
            self._vectorizer = None
            self._doc_mat = None
            return
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            norm="l2",
            lowercase=True,
        )
        self._doc_mat = self._vectorizer.fit_transform(documents)
        self._doc_mat = normalize(self._doc_mat, norm="l2")

    def rank(self, query: str, top_n: int) -> List[Tuple[int, float]]:
        if not self._vectorizer or self._doc_mat is None or not query.strip():
            return []
        q = self._vectorizer.transform([query])
        q = normalize(q, norm="l2")
        sims = (self._doc_mat @ q.T).toarray().ravel()
        if sims.size == 0:
            return []
        top_n = min(top_n, sims.shape[0])
        order = np.argsort(-sims)[:top_n]
        return [(int(i), float(sims[i])) for i in order]
