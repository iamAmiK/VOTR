# Token-Budget Pareto Analysis

Accuracy = Recall@injected_k  (fraction of queries where correct tool is in the injected set).
Token proxy = avg_k × 50 tokens/tool (compressed schema format).

| Policy | Accuracy | Avg k | Tokens (proxy) | Null-route % |
| --- | ---: | ---: | ---: | ---: |
| fixed_k=1 | 0.964 | 1.00 | 50 | 0.0% |
| fixed_k=3 | 0.980 | 3.00 | 150 | 0.0% |
| fixed_k=5 | 0.990 | 5.00 | 250 | 0.0% |
| gap_adaptive | 0.982 | 2.91 | 145 | 0.0% |
| conformal_adaptive | 0.982 | 2.91 | 145 | 0.0% |
