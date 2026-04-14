## Baselines And Ablations

Automated benchmark runner for retrieval-profile comparisons over the existing
functional-correctness suites.

### Profiles

- `dense_only`
- `bm25_only`
- `dense_bm25`
- `full_stack`
- `no_handoff_policy`
- `no_session_memory`

### Output

Reports are written to `benchmarks/results/baselines_ablations/`:

- one JSON file per `suite x profile`
- one aggregate `summary.json`
- one aggregate `summary.md`

### Run

```bash
python benchmarks/baselines_ablations/run_profiles.py
```
