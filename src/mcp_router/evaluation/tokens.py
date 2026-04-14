"""
Tokenizer-accurate token counts for injected tool schemas.

Uses ``tiktoken`` (OpenAI public encodings). Counts depend on the chosen
encoding; defaults to ``cl100k_base`` (GPT-4 / text-embedding-3 family).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from mcp_router.evaluation.mcp_zero_format import mcp_zero_function_block
from mcp_router.registry.schema import RegisteredTool, RegisteredServer
from mcp_router.schema_compress.compressor import compress_tool_line

if TYPE_CHECKING:
    import tiktoken


class TiktokenNotInstalledError(ImportError):
    pass


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    try:
        import tiktoken
    except ImportError as e:
        raise TiktokenNotInstalledError(
            "Install tiktoken for accurate counts: pip install tiktoken"
        ) from e
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, *, encoding_name: str = "cl100k_base") -> int:
    """Return token count for ``text`` under the given tiktoken encoding."""
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text))


def compressed_schema_tokens(
    server: RegisteredServer,
    tool: RegisteredTool,
    *,
    encoding_name: str = "cl100k_base",
) -> int:
    """Tokens for the string ``RouterEngine`` returns as ``RoutedTool.compressed``."""
    block = compress_tool_line(server.name, tool)
    return count_tokens(block, encoding_name=encoding_name)


def mcp_zero_schema_tokens(
    server: RegisteredServer,
    tool: RegisteredTool,
    *,
    encoding_name: str = "cl100k_base",
) -> int:
    """Tokens for one MCP-Zero-style ``<function>...</function>`` block."""
    block = mcp_zero_function_block(server.name, tool)
    return count_tokens(block, encoding_name=encoding_name)
