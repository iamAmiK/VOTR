from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """
    Tokenizes text with subword splitting and lightweight plural stripping.
    Used consistently across BM25, field re-ranking, and query decomposition.
    """
    original_tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    
    split_text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    split_text = re.sub(r"[_\-./:#]+", " ", split_text)
    split_tokens = re.findall(r"[a-zA-Z0-9]+", split_text.lower())
    
    stripped = []
    for t in split_tokens:
        if t.endswith("ies") and len(t) > 4:
            stripped.append(t[:-3] + "y")
        elif t.endswith("s") and len(t) > 3 and not t.endswith("ss") and not t.endswith("us"):
            stripped.append(t[:-1])
            
    return original_tokens + split_tokens + stripped
