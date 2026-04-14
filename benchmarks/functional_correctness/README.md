# Functional Correctness

Router-only benchmark suite.

## Scope

- Input: `server_intent` + `tool_intent`
- Label: `expected_tool_key`
- Primary metrics: top-1, top-3, top-5
- Secondary metrics: `confidence`, `recommended_handoff_k`, handoff success@k

## Notes

- Regenerate tier sizes (100 / 250 / 500 evaluation rows) with `python benchmarks/functional_correctness/build_scaled_suites.py` from repo root.
- No orchestrator/policy assumptions in this suite.
- The same cases can later be reused for end-to-end comparison.
- Store reports in `benchmarks/results/functional_correctness/`.
- Use `--equivalence-map benchmarks/functional_correctness/equivalence_map.json` to report an additional `equivalence_aware` summary for known alias/duplicate-server tool groups while keeping the strict metrics unchanged.
