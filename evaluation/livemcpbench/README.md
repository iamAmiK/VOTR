## LiveMCPBench Retrieval Eval

Utilities for preparing a paper-aligned retrieval evaluation dataset from
`LiveMCPBench`.

The initial preparation step does three things:

- parses task questions, ordered steps, and ordered tool annotations
- maps annotated tools to parent MCP servers using `tools/LiveMCPTool/tools.json`
- emits a strict `exact_step_subset` where each step aligns 1:1 with an annotated
  tool and the tool maps to a unique server

### Run

```bash
python evaluation/livemcpbench/prepare_paper_aligned_eval.py \
  --annotations "C:\path\to\LiveMCPBench\annotated_data\all_annotations.json" \
  --tool-catalog "C:\path\to\LiveMCPBench\tools\LiveMCPTool\tools.json" \
  --server-catalog "C:\path\to\LiveMCPBench\tools\LiveMCPTool\all_config.json"
```

Outputs are written to `evaluation/results/livemcpbench/`.

### Paper-Faithful Variant

This repo also includes an eval-only implementation of the paper-style unified
tool-plus-agent retrieval procedure. It retrieves from a mixed `agent ∪ tool`
catalog, then collapses ranked tool hits to their owner servers using Algorithm 1.

```bash
python evaluation/livemcpbench/eval_tool_to_agent_paper_variant.py \
  --prepared evaluation/results/livemcpbench/paper_aligned_exact_step_subset.json \
  --catalog data/catalog_subsets/livemcpbench.embedding.json
```

This currently runs on the locally derivable exact-step subset, not the full
95-task benchmark.

### Full-95 Extensions

The repo also provides two clearly labeled extensions over the full 95-task
annotation file:

- `full95-task-level-server-retrieval`: uses the original task question and
  derives the gold server set from the task's annotated tools
- `full95-reconstructed-stepwise-server-retrieval`: uses step queries for all
  95 tasks, but reconstructs step-level gold server sets heuristically from the
  ordered step and tool annotations

```bash
python evaluation/livemcpbench/eval_full95_extensions.py \
  --prepared evaluation/results/livemcpbench/paper_aligned_prepared.json \
  --catalog data/catalog_subsets/livemcpbench.embedding.json
```
