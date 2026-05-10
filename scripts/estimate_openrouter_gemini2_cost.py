import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    meta = json.loads((root / "data/index/meta.json").read_text(encoding="utf-8"))
    servers = meta.get("servers", [])

    def n(s: str | None) -> int:
        return len(s or "")

    # Index-time texts embedded by the project's build scripts:
    # - server description embedding
    # - server summary embedding
    # - tool description embedding (one per tool)
    desc_len = 0
    sum_len = 0
    tool_len = 0
    num_tools = 0

    for s in servers:
        desc = s.get("description") or s.get("name", "") or ""
        summary = s.get("summary") or desc
        desc_len += n(desc)
        sum_len += n(summary)
        tools = s.get("tools") or []
        for t in tools:
            tool_desc = t.get("description") or t.get("name", "") or ""
            tool_len += n(tool_desc)
            num_tools += 1

    chars_index = desc_len + sum_len + tool_len
    est_tokens_index = chars_index / 4.0

    # Query-time texts embedded per evaluated row:
    # engine.py embeds server_intent and tool_intent for each /route call.
    fc_dir = root / "benchmarks" / "functional_correctness"
    suites = [
        "ambiguity_collision.priority.json",
        "multi_hop.large.cross_app.json",
        "multi_hop.medium_250.cross_app.json",
        "multi_hop.small_100.cross_app.json",
        "multi_tool.large.single_turn.json",
        "multi_tool.medium_250.single_turn.json",
        "multi_tool.small_100.single_turn.json",
        "robustness_safety.priority.json",
        "single_tool.bloomberg.clean.json",
        "single_tool.clean.json",
        "single_tool.github.clean.json",
        "single_tool.medium_250.clean.json",
        "single_tool.telegram.clean.json",
    ]

    chars_query = 0
    rows = 0
    for name in suites:
        data = json.loads((fc_dir / name).read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                chars_query += n(item.get("server_intent")) + n(item.get("tool_intent"))
                rows += 1
        elif isinstance(data, dict):
            cases = data.get("cases")
            if cases is None:
                items = data.get("items") or []
                for item in items:
                    chars_query += n(item.get("server_intent")) + n(item.get("tool_intent"))
                    rows += 1
            else:
                for c in cases:
                    hops = c.get("hops") or []
                    for h in hops:
                        chars_query += n(h.get("server_intent")) + n(h.get("tool_intent"))
                        rows += 1
                    subs = c.get("subtasks") or []
                    for sub in subs:
                        chars_query += n(sub.get("server_intent")) + n(sub.get("tool_intent"))
                        rows += 1
        else:
            raise RuntimeError(f"Unexpected suite format for {name}")

    est_tokens_query = chars_query / 4.0

    # Pricing provided by user:
    cost_per_million = 0.208
    index_cost = est_tokens_index / 1e6 * cost_per_million
    query_cost = est_tokens_query / 1e6 * cost_per_million
    total_cost = index_cost + query_cost

    print("EST_COST_OPENROUTER_GEMINI_EMBEDDING_2_PREVIEW")
    print(f"Index: servers={len(servers)} tools={num_tools}")
    print(f"Index chars={chars_index} est_tokens={est_tokens_index:.0f}")
    print(f"Index cost=${index_cost:.4f}")
    print(f"Query rows (/route calls)={rows}")
    print(f"Query chars={chars_query} est_tokens={est_tokens_query:.0f}")
    print(f"Query cost=${query_cost:.4f}")
    print(f"Total est=${total_cost:.4f}")


if __name__ == "__main__":
    main()

