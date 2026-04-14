# Baselines And Ablations

| Suite | Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Extra |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| multi_hop.small_100 | dense_only | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.small_100 | bm25_only | 0.990 | 0.990 | 1.000 | 1.000 | chain@1=0.950 |
| multi_hop.small_100 | dense_bm25 | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.small_100 | full_stack | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.small_100 | no_handoff_policy | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.small_100 | no_session_memory | 1.000 | 1.000 | 1.000 | 1.000 | chain@1=1.000 |
| multi_hop.medium_250 | dense_only | 0.928 | 0.936 | 0.936 | 0.936 | chain@1=0.720 |
| multi_hop.medium_250 | bm25_only | 0.936 | 0.936 | 0.936 | 0.936 | chain@1=0.740 |
| multi_hop.medium_250 | dense_bm25 | 0.928 | 0.936 | 0.936 | 0.936 | chain@1=0.720 |
| multi_hop.medium_250 | full_stack | 0.936 | 0.936 | 0.936 | 0.936 | chain@1=0.740 |
| multi_hop.medium_250 | no_handoff_policy | 0.936 | 0.936 | 0.936 | 0.936 | chain@1=0.740 |
| multi_hop.medium_250 | no_session_memory | 0.936 | 0.936 | 0.936 | 0.936 | chain@1=0.740 |
| multi_hop.large | dense_only | 0.938 | 0.950 | 0.952 | 0.952 | chain@1=0.790 |
| multi_hop.large | bm25_only | 0.990 | 0.998 | 0.998 | 0.998 | chain@1=0.960 |
| multi_hop.large | dense_bm25 | 0.942 | 0.974 | 0.986 | 0.980 | chain@1=0.810 |
| multi_hop.large | full_stack | 0.964 | 0.980 | 0.990 | 0.982 | chain@1=0.870 |
| multi_hop.large | no_handoff_policy | 0.964 | 0.980 | 0.990 | 0.992 | chain@1=0.870 |
| multi_hop.large | no_session_memory | 0.964 | 0.980 | 0.990 | 0.982 | chain@1=0.870 |
