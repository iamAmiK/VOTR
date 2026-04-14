# Paper Comparison Summary

## MCP-Zero vs VOTR

| Metric | MCP-Zero | VOTR |
| --- | ---: | ---: |
| Servers | 308 | 309 |
| Tools | 2797 | 2806 |
| Top-1 accuracy | 95.2% | 96.4% |
| Accuracy gain | --- | +1.2 pp |
| Avg prompt / route tokens | 111.0* | 230.3 |
| Latency p50 (ms) | n/r | 312.37 |
| Latency p95 (ms) | n/r | 392.69 |
| Hybrid sparse+dense retrieval | No | Yes |
| Dynamic registry | No | Yes |
| Confidence-gated adaptive k | No | Yes |
| Compressed schema format | No | Yes |

* MCP-Zero token figure is the published full-collection single-turn APIBank result, not a rerun on the VOTR benchmark harness.

## VOTR Full-Catalogue Savings

| Quantity | Value |
| --- | ---: |
| Tool count | 2806 |
| Full-catalogue MCP-Zero-style tokens (same index) | 262487.0 |
| Full-catalogue VOTR compact tokens (same index) | 73650.0 |
| VOTR average route tokens | 230.27 |
| Reduction vs full-catalogue MCP-Zero-style injection | 99.912% |
| Reduction vs full-catalogue VOTR compact injection | 99.687% |

These numbers are derived from local benchmark artifacts and are ready to cite in the paper with an explicit note about which baseline metrics are published versus locally re-measured.
