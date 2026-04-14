"""Evaluation helpers (needle-in-a-haystack, APIBank) — stubs for paper experiments."""

from __future__ import annotations


def retrieval_accuracy_at_k(predicted: list[str], gold: str, k: int) -> bool:
    return gold in predicted[:k]


def estimate_tokens_for_compressed_tools(
    compressed_lines: list[str],
    *,
    encoding_name: str = "cl100k_base",
) -> int:
    """
    Token count for a list of compressed schema blocks (one block per list element).

    Uses ``tiktoken`` when installed; otherwise falls back to a whitespace token proxy.
    """
    try:
        from mcp_router.evaluation.tokens import count_tokens

        return sum(count_tokens(line, encoding_name=encoding_name) for line in compressed_lines)
    except ImportError:
        return sum(len(line.split()) for line in compressed_lines)
