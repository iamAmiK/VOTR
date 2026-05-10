"""Gemma one-shot benchmark runner.

Starts Gemma router on port 8775, runs all functional correctness suites,
and writes reports under benchmarks/results/functional_correctness_gemma.
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
OUT = ROOT / "benchmarks" / "results" / "functional_correctness_gemma"
EVAL = FC / "evaluate.py"
EQUIV = FC / "equivalence_map.json"
PORT = 8775
BASE_URL = f"http://127.0.0.1:{PORT}"


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "experiments.gemma.router_gemma:app",
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
            raise RuntimeError(f"Gemma router did not become healthy in time.\n{err}")

        OUT.mkdir(parents=True, exist_ok=True)
        deprecated = frozenset(
            {
                "multi_hop.small_3.cross_app.json",
                "multi_hop.medium_50.cross_app.json",
                "multi_tool.small_3.single_turn.json",
                "multi_tool.medium_50.single_turn.json",
                "single_tool.medium_50.clean.json",
                "multi_hop.cross_app.json",
                "multi_tool.single_turn.json",
            }
        )
        suites = sorted(
            p
            for p in FC.glob("*.json")
            if p.name != "equivalence_map.json" and p.name not in deprecated
        )
        if not suites:
            raise SystemExit("No suite JSON files found.")

        equiv_args: list[str] = []
        if EQUIV.is_file():
            equiv_args = ["--equivalence-map", str(EQUIV)]

        for path in suites:
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
                    *equiv_args,
                ],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
        print(f"\nWrote Gemma reports under {OUT}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
