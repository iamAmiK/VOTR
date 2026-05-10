"""Qwen3 Embedding 8B main benchmark runner.

Starts Qwen router on port 8778, runs the main comparison suites, and writes
reports under benchmarks/results/functional_correctness_qwen_embedding_8b.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "benchmarks" / "functional_correctness"
OUT = ROOT / "benchmarks" / "results" / "functional_correctness_qwen_embedding_8b"
EVAL = FC / "evaluate.py"
EQUIV = FC / "equivalence_map.json"
PORT = 8778
BASE_URL = f"http://127.0.0.1:{PORT}"

MAIN_SUITES = [
    "single_tool.clean.json",
    "multi_hop.large.cross_app.json",
    "multi_hop.medium_250.cross_app.json",
    "multi_tool.large.single_turn.json",
    "multi_tool.medium_250.single_turn.json",
    "ambiguity_collision.priority.json",
    "robustness_safety.priority.json",
]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "experiments.qwen_embedding_8b.router_qwen:app",
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
        for _ in range(90):
            try:
                urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"Qwen router did not become healthy in time.\n{err}")

        OUT.mkdir(parents=True, exist_ok=True)
        equiv_args: list[str] = []
        if EQUIV.is_file():
            equiv_args = ["--equivalence-map", str(EQUIV)]

        for suite_name in MAIN_SUITES:
            path = FC / suite_name
            if not path.is_file():
                raise FileNotFoundError(path)
            outp = OUT / f"{path.stem}.report.json"
            print(f"==> {path.name}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(EVAL),
                    "--base-url",
                    BASE_URL,
                    "--cases",
                    str(path),
                    "--out",
                    str(outp),
                    "--progress",
                    *equiv_args,
                ],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
        print(f"\nWrote Qwen reports under {OUT}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
