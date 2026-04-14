# Router Results Tables

## Functional Correctness
### multi_hop.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 0.800 | 1.000 | 1.000 | 1.000 | chain@1=0.333 |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| dense_only | 0.933 | 1.000 | 1.000 | 1.000 | chain@1=0.667 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
### multi_hop.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| dense_bm25 | 0.933 | 1.000 | 1.000 | 1.000 | chain@1=0.667 |
| dense_only | 0.933 | 1.000 | 1.000 | 1.000 | chain@1=0.667 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
### multi_hop.small_3
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
### multi_tool.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 0.857 | 1.000 | 1.000 | 1.000 | all_targets@1=0.667 |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| dense_only | 0.857 | 1.000 | 1.000 | 1.000 | all_targets@1=0.667 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
### multi_tool.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| dense_bm25 | 0.889 | 1.000 | 1.000 | 1.000 | all_targets@1=0.667 |
| dense_only | 0.889 | 1.000 | 1.000 | 1.000 | all_targets@1=0.667 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
### multi_tool.small_3
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
### single_tool.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 0.990 | 0.998 | 0.998 | 0.998 |  |
| dense_bm25 | 0.942 | 0.974 | 0.986 | 0.978 |  |
| dense_only | 0.938 | 0.950 | 0.952 | 0.952 |  |
| full_stack | 0.964 | 0.980 | 0.990 | 0.982 |  |
| no_handoff_policy | 0.964 | 0.980 | 0.990 | 0.992 |  |
| no_session_memory | 0.964 | 0.980 | 0.990 | 0.982 |  |
### single_tool.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 0.932 | 0.932 | 0.932 | 0.932 |  |
| dense_bm25 | 0.920 | 0.932 | 0.932 | 0.932 |  |
| dense_only | 0.920 | 0.932 | 0.932 | 0.932 |  |
| full_stack | 0.932 | 0.932 | 0.932 | 0.932 |  |
| no_handoff_policy | 0.932 | 0.932 | 0.932 | 0.932 |  |
| no_session_memory | 0.932 | 0.932 | 0.932 | 0.932 |  |
### single_tool.small_bloomberg
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 |  |
### single_tool.small_github
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 |  |
### single_tool.small_telegram
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 |  |
| full_stack | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 |  |
| no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 |  |
## Functional Groups
### single_tool
| Suite | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_tool.large | 0.964 | 0.980 | 0.990 | 0.982 |  |
| single_tool.medium_50 | 0.932 | 0.932 | 0.932 | 0.932 |  |
| single_tool.small_bloomberg | 1.000 | 1.000 | 1.000 | 1.000 |  |
| single_tool.small_github | 1.000 | 1.000 | 1.000 | 1.000 |  |
| single_tool.small_telegram | 1.000 | 1.000 | 1.000 | 1.000 |  |
### multi_hop
| Suite | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_hop.large | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.medium_50 | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.small_3 | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
### multi_tool
| Suite | Top-1 | Top-3 | Top-5 | Handoff@k | Case Metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_tool.large | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| multi_tool.medium_50 | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |
| multi_tool.small_3 | 1.000 | 1.000 | 1.000 | 1.000 | all_targets@1=1.000 |

## Confidence Calibration
### multi_hop.large
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 15 | 1.000 | 1.000 | 2.867 |
### multi_hop.medium_50
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 15 | 1.000 | 1.000 | 4.067 |
### multi_hop.small_3
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 6 | 1.000 | 1.000 | 2.667 |
### multi_tool.large
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 7 | 1.000 | 1.000 | 2.429 |
### multi_tool.medium_50
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 9 | 1.000 | 1.000 | 3.889 |
### multi_tool.small_3
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 6 | 1.000 | 1.000 | 2.667 |
### single_tool.large
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 500 | 0.964 | 0.982 | 3.362 |
### single_tool.medium_50
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 88 | 0.932 | 0.932 | 3.136 |
### single_tool.small_bloomberg
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 9 | 1.000 | 1.000 | 2.556 |
### single_tool.small_github
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 26 | 1.000 | 1.000 | 3 |
### single_tool.small_telegram
| Items | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: |
| 64 | 1.000 | 1.000 | 3.281 |
## Confidence Groups
| Scale | Suite | Top-1 | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: | ---: |
| large | multi_hop.large | 1.000 | 1.000 | 2.867 |
| large | multi_tool.large | 1.000 | 1.000 | 2.429 |
| large | single_tool.large | 0.964 | 0.982 | 3.362 |
| medium | multi_hop.medium_50 | 1.000 | 1.000 | 4.067 |
| medium | multi_tool.medium_50 | 1.000 | 1.000 | 3.889 |
| medium | single_tool.medium_50 | 0.932 | 0.932 | 3.136 |
| small | multi_hop.small_3 | 1.000 | 1.000 | 2.667 |
| small | multi_tool.small_3 | 1.000 | 1.000 | 2.667 |
| small | single_tool.small_bloomberg | 1.000 | 1.000 | 2.556 |
| small | single_tool.small_github | 1.000 | 1.000 | 3 |
| small | single_tool.small_telegram | 1.000 | 1.000 | 3.281 |

## Efficiency
### multi_hop.large
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 271.748 | 284.851 | 273.008 | 8 | 2.867 | 231.417 |
### multi_hop.medium_50
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 252.372 | 310.888 | 266.841 | 8 | 4.067 | 206.683 |
### multi_hop.small_3
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 260.412 | 273.211 | 260.589 | 8 | 2.667 | 234.250 |
### multi_tool.large
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 254.486 | 285.284 | 260.381 | 8 | 2.429 | 233.500 |
### multi_tool.medium_50
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 310.578 | 549.537 | 333.576 | 8 | 3.889 | 215.778 |
### multi_tool.small_3
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 255.674 | 278.820 | 256.347 | 8 | 2.667 | 234.250 |
### single_tool.large
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 312.374 | 392.691 | 325.635 | 8 | 3.362 | 230.273 |
### single_tool.medium_50
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 88 | 344.062 | 447.638 | 355.959 | 8 | 3.136 | 226.273 |
### single_tool.small_bloomberg
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 251.303 | 566.472 | 283.746 | 8 | 2.556 | 256.889 |
### single_tool.small_github
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26 | 256.774 | 288.413 | 259.133 | 8 | 3 | 292.731 |
### single_tool.small_telegram
| Items | P50 ms | P95 ms | Mean ms | Returned Tools | Avg Recommended k | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 275.440 | 431.640 | 295.651 | 8 | 3.281 | 171.191 |
## Efficiency Groups
| Scale | Suite | P50 ms | P95 ms | Mean Est. Tokens |
| --- | ---: | ---: | ---: | ---: |
| large | multi_hop.large | 271.748 | 284.851 | 231.417 |
| large | multi_tool.large | 254.486 | 285.284 | 233.500 |
| large | single_tool.large | 312.374 | 392.691 | 230.273 |
| medium | multi_hop.medium_50 | 252.372 | 310.888 | 206.683 |
| medium | multi_tool.medium_50 | 310.578 | 549.537 | 215.778 |
| medium | single_tool.medium_50 | 344.062 | 447.638 | 226.273 |
| small | multi_hop.small_3 | 260.412 | 273.211 | 234.250 |
| small | multi_tool.small_3 | 255.674 | 278.820 | 234.250 |
| small | single_tool.small_bloomberg | 251.303 | 566.472 | 256.889 |
| small | single_tool.small_github | 256.774 | 288.413 | 292.731 |
| small | single_tool.small_telegram | 275.440 | 431.640 | 171.191 |

## Overlap-Aware Evaluation
### multi_hop.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 |
### multi_hop.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### multi_hop.small_3
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### multi_tool.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### multi_tool.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### multi_tool.small_3
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### single_tool.large
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.964 | 0.980 | 0.990 | 0.982 | 0.000 | 0.000 | 0.000 | 0.972 |
| overlap_aware | 0.964 | 0.980 | 0.990 | 0.982 | 0.012 | 1.000 | 1.000 | 0.972 |
### single_tool.medium_50
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.932 | 0.932 | 0.932 | 0.932 | 0.000 | 0.000 | 0.000 | 0.932 |
| overlap_aware | 0.932 | 0.932 | 0.932 | 0.932 | 0.000 | 0.000 | 0.000 | 0.932 |
### single_tool.small_bloomberg
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### single_tool.small_github
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
### single_tool.small_telegram
| Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous | Equiv Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| overlap_aware | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
## Overlap Groups
| Suite | Profile | Top-1 | Ambiguous Rate | Exact@1 On Ambiguous | Equiv@1 On Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_hop.large | baseline | 1.000 | 0.000 | 0.000 | 0.000 |
| multi_hop.large | overlap_aware | 1.000 | 0.200 | 1.000 | 1.000 |
| multi_tool.large | baseline | 1.000 | 0.000 | 0.000 | 0.000 |
| multi_tool.large | overlap_aware | 1.000 | 0.000 | 0.000 | 0.000 |
| single_tool.large | baseline | 0.964 | 0.000 | 0.000 | 0.000 |
| single_tool.large | overlap_aware | 0.964 | 0.012 | 1.000 | 1.000 |

## Abstention And Registration
| Suite | Cases | Pass Rate | Avg Recommended k | Confidence Distribution |
| --- | ---: | ---: | ---: | ---: |
| Policy-cleaned unsupported intents | 8 | 1.000 | 5 | low:8 |
| Raw unsupported stress | 8 | 1.000 | 5 | low:8 |
### Threshold Sensitivity
| Config | High Gap | Medium Gap | Handoff@k | Avg Recommended k |
| --- | ---: | ---: | ---: | ---: |
| Current | 0.001 | 0.001 | 0.984 | 3.066 |
| Dev-recommended | 0.001 | 0.001 | 0.975 | 2.082 |
### Dynamic Registration
| Control Hit@1 | Pre-register Miss@5 Rate | Post-register Hit@1 Rate | Controls Preserved@1 Rate |
| --- | ---: | ---: | ---: |
| 1.000 | 1.000 | 1.000 | 1.000 |

## Router Eval Sweep
### Multi-hop Eval
| Dataset | k | Mean Hop@1 | Mean Hop@k | Chain Success@k |
| --- | ---: | ---: | ---: | ---: |
| multihop_cases.10hop | 5 | 0.900 | 1.000 | 1.000 |
| multihop_cases.20hop | 5 | 0.950 | 1.000 | 1.000 |
| multihop_cases.20hop.strict | 5 | 0.900 | 1.000 | 1.000 |
| multihop_cases.25hop.strict | 5 | 0.920 | 1.000 | 1.000 |
| multihop_cases.50hop.adversarial.unique_servers | 5 | 0.000 | 0.000 | 0.000 |
| multihop_cases.50hop.adversarial_valid.unique_servers | 5 | 0.580 | 0.900 | 0.000 |
| multihop_cases.50hop.realistic_hard.unique_servers | 5 | 1.000 | 1.000 | 1.000 |
| multihop_cases.50hop.unique_servers | 5 | 1.000 | 1.000 | 1.000 |
| multihop_cases.50hop_handmade_hard.unique_servers | 5 | 1.000 | 1.000 | 1.000 |
| multihop_cases.sample | 5 | 1.000 | 1.000 | 1.000 |
### Confidence Eval
| Dataset | Hops | Top-1 | Handoff@k | Confidence Counts |
| --- | ---: | ---: | ---: | ---: |
| multihop_cases.10hop | 10 | 0.900 | 1.000 | high:5, low:5 |
| multihop_cases.20hop | 20 | 0.950 | 1.000 | high:8, low:11, medium:1 |
| multihop_cases.20hop.strict | 20 | 0.900 | 1.000 | high:8, low:11, medium:1 |
| multihop_cases.25hop.strict | 25 | 0.920 | 1.000 | high:9, low:14, medium:2 |
| multihop_cases.50hop.adversarial.unique_servers | 50 | 0.000 | 0.000 | high:33, low:15, medium:2 |
| multihop_cases.50hop.adversarial_valid.unique_servers | 50 | 0.580 | 0.700 | high:33, low:11, medium:6 |
| multihop_cases.50hop.realistic_hard.unique_servers | 50 | 1.000 | 1.000 | high:22, low:21, medium:7 |
| multihop_cases.50hop.unique_servers | 50 | 1.000 | 1.000 | high:17, low:25, medium:8 |
| multihop_cases.50hop_handmade_hard.unique_servers | 50 | 1.000 | 1.000 | high:26, low:19, medium:5 |
| multihop_cases.sample | 5 | 1.000 | 1.000 | high:2, low:3 |
### Handoff Calibration
| Dataset | High Gap | Medium Gap | Handoff Accuracy | Avg Recommended k | Constraints OK |
| --- | ---: | ---: | ---: | ---: | ---: |
| multihop_cases.10hop | 0.002 | 0.001 | 1.000 | 3.400 | True |
| multihop_cases.20hop | 0.001 | 0.001 | 1.000 | 1.800 | True |
| multihop_cases.20hop.strict | 0.001 | 0.001 | 1.000 | 3.400 | True |
| multihop_cases.25hop.strict | 0.001 | 0.001 | 1.000 | 3.400 | True |
| multihop_cases.50hop.adversarial.unique_servers | 0.000 | 0.000 | 0.000 | 1.120 | False |
| multihop_cases.50hop.adversarial_valid.unique_servers | 0.011 | 0.003 | 0.880 | 4.720 | False |
| multihop_cases.50hop.realistic_hard.unique_servers | 0.001 | 0.001 | 1.000 | 1.720 | True |
| multihop_cases.50hop.unique_servers | 0.001 | 0.001 | 1.000 | 1.760 | True |
| multihop_cases.50hop_handmade_hard.unique_servers | 0.001 | 0.000 | 1.000 | 1.080 | True |
| multihop_cases.sample | 0.001 | -0.000 | 1.000 | 2.200 | True |

## LiveMCPBench
### Individual LiveMCPBench Results
| Eval | Queries | Recall@1 | Recall@3 | Recall@5 | mAP@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact subset server eval | 82 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Paper-faithful tool-to-agent | 82 | 0.707 | 0.829 | 0.878 | 0.773 | 0.799 |
| Full95 task-level | 95 | 0.507 | 0.688 | 0.754 | 0.628 | 0.678 |
| Full95 reconstructed stepwise | 268 | 0.524 | 0.616 | 0.660 | 0.577 | 0.599 |
| Router format exact subset | 82 | 0.463 | 0.671 | 0.756 | 0.578 | 0.622 |
| Router format full95 task-level | 95 | 0.423 | 0.604 | 0.646 | 0.529 | 0.575 |
| Router format full95 reconstructed | 268 | 0.483 | 0.623 | 0.683 | 0.562 | 0.593 |
| Router policy payloads | 268 | 0.877 | 0.924 | 0.927 | 0.901 | 0.909 |
### Grouped LiveMCPBench Results
| Group | Eval | Recall@1 | Recall@5 | mAP@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper-aligned | Exact subset server eval | 0.000 | 0.000 | 0.000 | 0.000 |
| Paper-aligned | Paper-faithful tool-to-agent | 0.707 | 0.878 | 0.773 | 0.799 |
| Full95 extensions | Task-level | 0.507 | 0.754 | 0.628 | 0.678 |
| Full95 extensions | Reconstructed stepwise | 0.524 | 0.660 | 0.577 | 0.599 |
| Router-native | Router format exact subset | 0.463 | 0.756 | 0.578 | 0.622 |
| Router-native | Router format full95 task-level | 0.423 | 0.646 | 0.529 | 0.575 |
| Router-native | Router format full95 reconstructed | 0.483 | 0.683 | 0.562 | 0.593 |
| Router-native | Router policy payloads | 0.877 | 0.927 | 0.901 | 0.909 |
