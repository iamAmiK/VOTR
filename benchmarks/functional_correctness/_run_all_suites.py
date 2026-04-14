"""One-shot: start uvicorn, run every suite JSON, stop server. Not part of pytest."""
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
OUT = ROOT / "benchmarks" / "results" / "functional_correctness"
EVAL = FC / "evaluate.py"
EQUIV = FC / "equivalence_map.json"


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Install python-dotenv or set OPENAI_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    load_dotenv(ROOT.parent / "LiveMCPBench" / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (e.g. load LiveMCPBench/.env).", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mcp_router.router:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for i in range(90):
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"Router did not become healthy in time.\n{err}")

        OUT.mkdir(parents=True, exist_ok=True)
        deprecated = frozenset(
            {
                "multi_hop.small_3.cross_app.json",
                "multi_hop.medium_50.cross_app.json",
                "multi_tool.small_3.single_turn.json",
                "multi_tool.medium_50.single_turn.json",
                "single_tool.medium_50.clean.json",
                # Same scale as multi_hop.large / multi_tool.large (avoids duplicate 500-step runs).
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
                    "http://127.0.0.1:8765",
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
        print(f"\nWrote reports under {OUT}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
