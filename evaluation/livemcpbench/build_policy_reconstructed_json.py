#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_full95_extensions import reconstruct_steps  # noqa: E402


DEFAULT_PREPARED = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_prepared.json"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "router_policy_reconstructed_stepwise.json"


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "help",
    "identified",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "other",
    "please",
    "provide",
    "save",
    "show",
    "single",
    "the",
    "this",
    "to",
    "up",
    "use",
    "using",
    "with",
    "within",
}

GENERIC_SERVER_WORDS = {
    "a",
    "agent",
    "app",
    "context",
    "mcp",
    "model",
    "official",
    "protocol",
    "server",
    "service",
    "tool",
    "tools",
}

CLEANUP_REPLACEMENTS = {
    r"\bgithb\b": "github",
    r"\bgh\b": "github",
    r"\bprs\b": "pull requests",
}

LEADING_VERB_REPLACEMENTS = {
    "getting": "get",
    "retrieves": "retrieve",
    "retrieving": "retrieve",
    "calculates": "calculate",
    "calculating": "calculate",
    "converts": "convert",
    "converting": "convert",
    "creates": "create",
    "creating": "create",
    "generates": "generate",
    "generating": "generate",
    "lists": "list",
    "listing": "list",
    "reads": "read",
    "reading": "read",
    "searches": "search",
    "searching": "search",
    "writes": "write",
    "writing": "write",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def apply_allowed_cleanup(text: str) -> str:
    out = normalize_spaces(text)
    for pattern, repl in CLEANUP_REPLACEMENTS.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return normalize_spaces(out)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def compact_phrase(text: str, min_words: int, max_words: int) -> str:
    toks = words(text)
    if not toks:
        return ""
    toks = toks[:max_words]
    if len(toks) < min_words:
        return " ".join(toks)
    return " ".join(toks)


def infer_server_phrase_from_name(server_names: list[str]) -> str:
    if not server_names:
        return ""
    raw = server_names[0]
    toks = [tok for tok in words(raw) if tok not in GENERIC_SERVER_WORDS]
    if not toks:
        toks = words(raw)
    phrase = " ".join(toks[:6]).strip()
    if not phrase:
        return ""
    if "operations" not in phrase.split():
        phrase = f"{phrase} operations"
    return compact_phrase(phrase, min_words=2, max_words=6)


def infer_server_phrase_from_step(step_query: str) -> str:
    toks = [tok for tok in words(step_query) if tok not in STOPWORDS]
    if not toks:
        toks = words(step_query)
    phrase = " ".join(toks[:5]).strip()
    if not phrase:
        phrase = "general operations"
    if len(phrase.split()) < 2:
        phrase = f"{phrase} operations"
    return compact_phrase(phrase, min_words=2, max_words=6)


def build_server_intent(step: dict[str, Any]) -> str:
    gold_server_names = step.get("gold_server_names") or []
    phrase = infer_server_phrase_from_name(gold_server_names)
    if not phrase:
        assigned = step.get("assigned_resolved_servers") or []
        phrase = infer_server_phrase_from_name(assigned)
    if not phrase:
        phrase = infer_server_phrase_from_step(step.get("step_query", ""))
    return phrase


def build_tool_intent(step_query: str) -> str:
    text = apply_allowed_cleanup(step_query).strip().rstrip(".")
    toks = text.split()
    if toks:
        first = toks[0].lower()
        if first in LEADING_VERB_REPLACEMENTS:
            toks[0] = LEADING_VERB_REPLACEMENTS[first]
    text = " ".join(toks)
    text = normalize_spaces(text)
    tool_words = text.split()
    if len(tool_words) > 16:
        text = " ".join(tool_words[:16])
    return text


def build_payload(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_intent": build_server_intent(step),
        "tool_intent": build_tool_intent(step["step_query"]),
        "session_id": f"livemcpbench-{step['task_id']}",
        "record_session": True,
    }


def build_output(prepared_path: Path) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    rows = []
    for task in prepared["tasks"]:
        for step in reconstruct_steps(task):
            rows.append(
                {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "step_index": step["step_index"],
                    "step_query": step["step_query"],
                    "gold_server_names": step["gold_server_names"],
                    "reconstruction_label": step["reconstruction_label"],
                    "assigned_tool_names": step["assigned_tool_names"],
                    "payload": build_payload(step),
                }
            )
    return {
        "metadata": {
            "source_prepared_path": str(prepared_path),
            "policy_path": str(ROOT / "Policy.md"),
            "note": (
                "Policy-style reconstructed router payloads for LiveMCPBench stepwise evaluation. "
                "Payloads follow Policy.md heuristics without an extra LLM rewrite step."
            ),
            "num_payloads": len(rows),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Policy.md-style reconstructed router payload JSON.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    payload = build_output(args.prepared)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {payload['metadata']['num_payloads']} policy-style payloads to {args.out}")


if __name__ == "__main__":
    main()
