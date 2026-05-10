"""Gemini embedding large-suite runner (resumable).

Runs only the "large" functional correctness suites:
- multi_hop.large.cross_app
- multi_tool.large.single_turn

This runner is RESUMABLE: it writes a checkpoint file after each hop/subtask
so you can restart after rate limits/quotas or API key rotation.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "benchmarks" / "functional_correctness"
OUT = ROOT / "benchmarks" / "results" / "functional_correctness_gemini_embedding_2"
EQUIV = FC / "equivalence_map.json"
PORT = 8777
BASE_URL = f"http://127.0.0.1:{PORT}"

SUITES = [
    ("multi_hop.large.cross_app", FC / "multi_hop.large.cross_app.json"),
    ("multi_tool.large.single_turn", FC / "multi_tool.large.single_turn.json"),
]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def call_route(payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/route",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<failed to read error body>"
        raise RuntimeError(f"/route failed: HTTP {exc.code}. Body:\n{body}") from exc


def hit_at(predicted: list[str], expected: str, k: int) -> bool:
    return expected in predicted[:k]


def load_equivalence_lookup(path: Path | None) -> dict[str, set[str]]:
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = raw.get("equivalent_tool_groups", raw)
    if not isinstance(groups, list):
        return {}
    lookup: dict[str, set[str]] = {}
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        members = {str(item) for item in group}
        for member in members:
            lookup[member] = set(members)
    return lookup


def equivalent_keys(expected: str, equivalence_lookup: dict[str, set[str]]) -> set[str]:
    return set(equivalence_lookup.get(expected, {expected}))


def equivalent_hit_at(
    predicted: list[str],
    expected: str,
    k: int,
    equivalence_lookup: dict[str, set[str]],
) -> bool:
    valid = equivalent_keys(expected, equivalence_lookup)
    return any(tool_key in valid for tool_key in predicted[:k])


def evaluate_item(
    item: dict[str, Any],
    *,
    session_id: str,
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    payload = {
        "server_intent": item["server_intent"],
        "tool_intent": item["tool_intent"],
        "session_id": session_id,
        "record_session": keep_memory,
    }
    out = call_route(payload)
    predicted = [t["tool_key"] for t in out.get("tools", [])]
    expected = item["expected_tool_key"]
    recommended_k = int(out.get("recommended_handoff_k", 1))
    recommended_k = max(1, min(recommended_k, len(predicted))) if predicted else 1

    row = {
        "server_intent": item["server_intent"],
        "tool_intent": item["tool_intent"],
        "expected_tool_key": expected,
        "predicted_topk": predicted[:5],
        "hit_at_1": hit_at(predicted, expected, 1),
        "hit_at_3": hit_at(predicted, expected, 3),
        "hit_at_5": hit_at(predicted, expected, 5),
        "confidence": out.get("confidence", "unknown"),
        "recommended_handoff_k": recommended_k,
        "hit_at_recommended_k": hit_at(predicted, expected, recommended_k),
        "score_gap": out.get("score_gap", 0.0),
        "top1_score": out.get("top1_score", 0.0),
        "top2_score": out.get("top2_score", 0.0),
    }
    if equivalence_lookup:
        row["equivalent_tool_keys"] = sorted(equivalent_keys(expected, equivalence_lookup))
        row["equivalence_hit_at_1"] = equivalent_hit_at(predicted, expected, 1, equivalence_lookup)
        row["equivalence_hit_at_3"] = equivalent_hit_at(predicted, expected, 3, equivalence_lookup)
        row["equivalence_hit_at_5"] = equivalent_hit_at(predicted, expected, 5, equivalence_lookup)
        row["equivalence_hit_at_recommended_k"] = equivalent_hit_at(
            predicted,
            expected,
            recommended_k,
            equivalence_lookup,
        )
    return row


def summarize_metric_family(
    rows: list[dict[str, Any]],
    hit1_key: str,
    hit3_key: str,
    hit5_key: str,
    handoff_key: str,
) -> dict[str, Any]:
    n = len(rows)
    per_conf_counts: dict[str, int] = defaultdict(int)
    per_conf_top1: dict[str, int] = defaultdict(int)
    per_conf_handoff: dict[str, int] = defaultdict(int)

    for row in rows:
        conf = row["confidence"]
        per_conf_counts[conf] += 1
        per_conf_top1[conf] += int(row[hit1_key])
        per_conf_handoff[conf] += int(row[handoff_key])

    confidence_buckets = {}
    for conf in sorted(per_conf_counts.keys()):
        c = per_conf_counts[conf]
        confidence_buckets[conf] = {
            "count": c,
            "top1_accuracy": (per_conf_top1[conf] / c) if c else 0.0,
            "handoff_accuracy_at_recommended_k": (per_conf_handoff[conf] / c) if c else 0.0,
        }

    return {
        "num_items": n,
        "top1_accuracy": (sum(int(r[hit1_key]) for r in rows) / n) if n else 0.0,
        "top3_accuracy": (sum(int(r[hit3_key]) for r in rows) / n) if n else 0.0,
        "top5_accuracy": (sum(int(r[hit5_key]) for r in rows) / n) if n else 0.0,
        "handoff_accuracy_at_recommended_k": (sum(int(r[handoff_key]) for r in rows) / n) if n else 0.0,
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


def summarize_rows(rows: list[dict[str, Any]], equivalence_lookup: dict[str, set[str]]) -> dict[str, Any]:
    summary = summarize_metric_family(
        rows,
        hit1_key="hit_at_1",
        hit3_key="hit_at_3",
        hit5_key="hit_at_5",
        handoff_key="hit_at_recommended_k",
    )
    if equivalence_lookup:
        summary["equivalence_aware"] = summarize_metric_family(
            rows,
            hit1_key="equivalence_hit_at_1",
            hit3_key="equivalence_hit_at_3",
            hit5_key="equivalence_hit_at_5",
            handoff_key="equivalence_hit_at_recommended_k",
        )
    return summary


def run_multi_hop(
    suite_name: str,
    suite_path: Path,
    *,
    equivalence_lookup: dict[str, set[str]],
) -> None:
    outp = OUT / f"{suite_path.stem}.report.json"
    ckpt = OUT / f"{suite_path.stem}.checkpoint.json"

    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = raw.get("cases", [])
    if not cases:
        raise SystemExit(f"No cases in {suite_path}")

    case_results: list[dict[str, Any]] = []
    case_idx = 0
    hop_idx = 0
    current_case: dict[str, Any] | None = None

    if ckpt.is_file():
        c = json.loads(ckpt.read_text(encoding="utf-8"))
        if c.get("suite") == suite_name:
            case_results = c.get("case_results", []) or []
            case_idx = int(c.get("case_idx", 0))
            hop_idx = int(c.get("hop_idx", 0))
            current_case = c.get("current_case")
            print(f"[resume] {suite_name}: case {case_idx+1}/{len(cases)} hop {hop_idx+1}", flush=True)

    for ci in range(case_idx, len(cases)):
        case = cases[ci]
        session_id = case.get("session_id") or f"fc-{case['id']}"

        if current_case is None or current_case.get("case_id") != case["id"]:
            current_case = {
                "case_id": case["id"],
                "description": case.get("description", ""),
                "hops": [],
            }
            hop_idx = 0

        hops = case.get("hops") or []
        for hi in range(hop_idx, len(hops)):
            row = evaluate_item(
                hops[hi],
                session_id=session_id,
                keep_memory=False,
                equivalence_lookup=equivalence_lookup,
            )
            current_case["hops"].append(row)

            _atomic_write(
                ckpt,
                json.dumps(
                    {
                        "suite": suite_name,
                        "case_results": case_results,
                        "case_idx": ci,
                        "hop_idx": hi + 1,
                        "current_case": current_case,
                    },
                    ensure_ascii=False,
                ),
            )

        # finalize case
        rows = current_case["hops"]
        case_result = {
            "case_id": current_case["case_id"],
            "description": current_case.get("description", ""),
            "chain_success_at_1": all(r["hit_at_1"] for r in rows),
            "chain_success_at_3": all(r["hit_at_3"] for r in rows),
            "chain_success_at_5": all(r["hit_at_5"] for r in rows),
            "hops": rows,
        }
        if equivalence_lookup:
            case_result["equivalence_chain_success_at_1"] = all(r["equivalence_hit_at_1"] for r in rows)
            case_result["equivalence_chain_success_at_3"] = all(r["equivalence_hit_at_3"] for r in rows)
            case_result["equivalence_chain_success_at_5"] = all(r["equivalence_hit_at_5"] for r in rows)

        case_results.append(case_result)
        current_case = None
        hop_idx = 0

        if len(case_results) % 10 == 0:
            print(f"{suite_name}: completed {len(case_results)}/{len(cases)} cases", flush=True)

    # finalize report
    all_rows = [hop for c in case_results for hop in c.get("hops", [])]
    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["chain_success_rate_at_1"] = sum(int(c["chain_success_at_1"]) for c in case_results) / len(case_results)
    summary["chain_success_rate_at_3"] = sum(int(c["chain_success_at_3"]) for c in case_results) / len(case_results)
    summary["chain_success_rate_at_5"] = sum(int(c["chain_success_at_5"]) for c in case_results) / len(case_results)
    if equivalence_lookup:
        ea = summary["equivalence_aware"]
        ea["num_cases"] = len(case_results)
        ea["chain_success_rate_at_1"] = sum(int(c["equivalence_chain_success_at_1"]) for c in case_results) / len(
            case_results
        )
        ea["chain_success_rate_at_3"] = sum(int(c["equivalence_chain_success_at_3"]) for c in case_results) / len(
            case_results
        )
        ea["chain_success_rate_at_5"] = sum(int(c["equivalence_chain_success_at_5"]) for c in case_results) / len(
            case_results
        )

    report = {
        "summary": summary,
        "results": case_results,
    }
    _atomic_write(outp, json.dumps(report, ensure_ascii=False))
    ckpt.unlink(missing_ok=True)
    print(f"Wrote {outp}", flush=True)


def run_multi_tool(
    suite_name: str,
    suite_path: Path,
    *,
    equivalence_lookup: dict[str, set[str]],
) -> None:
    outp = OUT / f"{suite_path.stem}.report.json"
    ckpt = OUT / f"{suite_path.stem}.checkpoint.json"

    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = raw.get("cases", [])
    if not cases:
        raise SystemExit(f"No cases in {suite_path}")

    case_results: list[dict[str, Any]] = []
    case_idx = 0
    sub_idx = 0
    current_case: dict[str, Any] | None = None

    if ckpt.is_file():
        c = json.loads(ckpt.read_text(encoding="utf-8"))
        if c.get("suite") == suite_name:
            case_results = c.get("case_results", []) or []
            case_idx = int(c.get("case_idx", 0))
            sub_idx = int(c.get("sub_idx", 0))
            current_case = c.get("current_case")
            print(f"[resume] {suite_name}: case {case_idx+1}/{len(cases)} subtask {sub_idx+1}", flush=True)

    for ci in range(case_idx, len(cases)):
        case = cases[ci]
        session_id = case.get("session_id") or f"fc-{case['id']}"

        if current_case is None or current_case.get("case_id") != case["id"]:
            current_case = {
                "case_id": case["id"],
                "description": case.get("description", ""),
                "subtasks": [],
            }
            sub_idx = 0

        subtasks = case.get("subtasks") or []
        for si in range(sub_idx, len(subtasks)):
            row = evaluate_item(
                subtasks[si],
                session_id=session_id,
                keep_memory=False,
                equivalence_lookup=equivalence_lookup,
            )
            current_case["subtasks"].append(row)

            _atomic_write(
                ckpt,
                json.dumps(
                    {
                        "suite": suite_name,
                        "case_results": case_results,
                        "case_idx": ci,
                        "sub_idx": si + 1,
                        "current_case": current_case,
                    },
                    ensure_ascii=False,
                ),
            )

        rows = current_case["subtasks"]
        case_result = {
            "case_id": current_case["case_id"],
            "description": current_case.get("description", ""),
            "subtasks": rows,
            "chain_success_at_1": all(r["hit_at_1"] for r in rows),
            "chain_success_at_3": all(r["hit_at_3"] for r in rows),
            "chain_success_at_5": all(r["hit_at_5"] for r in rows),
        }
        if equivalence_lookup:
            case_result["equivalence_chain_success_at_1"] = all(r["equivalence_hit_at_1"] for r in rows)
            case_result["equivalence_chain_success_at_3"] = all(r["equivalence_hit_at_3"] for r in rows)
            case_result["equivalence_chain_success_at_5"] = all(r["equivalence_hit_at_5"] for r in rows)

        case_results.append(case_result)
        current_case = None
        sub_idx = 0

        if len(case_results) % 10 == 0:
            print(f"{suite_name}: completed {len(case_results)}/{len(cases)} cases", flush=True)

    all_rows = [sub for c in case_results for sub in c.get("subtasks", [])]
    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["chain_success_rate_at_1"] = sum(int(c["chain_success_at_1"]) for c in case_results) / len(case_results)
    summary["chain_success_rate_at_3"] = sum(int(c["chain_success_at_3"]) for c in case_results) / len(case_results)
    summary["chain_success_rate_at_5"] = sum(int(c["chain_success_at_5"]) for c in case_results) / len(case_results)
    if equivalence_lookup:
        ea = summary["equivalence_aware"]
        ea["num_cases"] = len(case_results)
        ea["chain_success_rate_at_1"] = sum(int(c["equivalence_chain_success_at_1"]) for c in case_results) / len(
            case_results
        )
        ea["chain_success_rate_at_3"] = sum(int(c["equivalence_chain_success_at_3"]) for c in case_results) / len(
            case_results
        )
        ea["chain_success_rate_at_5"] = sum(int(c["equivalence_chain_success_at_5"]) for c in case_results) / len(
            case_results
        )

    report = {
        "summary": summary,
        "results": case_results,
    }
    _atomic_write(outp, json.dumps(report, ensure_ascii=False))
    ckpt.unlink(missing_ok=True)
    print(f"Wrote {outp}", flush=True)


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if not env.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set in environment.")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "experiments.gemini_embedding_2.router_gemini:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"Gemini router did not become healthy in time.\n{err}")

        OUT.mkdir(parents=True, exist_ok=True)
        equivalence_lookup = load_equivalence_lookup(EQUIV if EQUIV.is_file() else None)

        for suite_name, suite_path in SUITES:
            print(f"==> {suite_path.name}", flush=True)
            if suite_name.startswith("multi_hop."):
                run_multi_hop(suite_name, suite_path, equivalence_lookup=equivalence_lookup)
            else:
                run_multi_tool(suite_name, suite_path, equivalence_lookup=equivalence_lookup)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()

