"""
Verify all headline numbers from actual evaluation files.
"""
import json, os, math

# ── 1. single_tool.large full_stack: what is the actual Top-1? ───────────────
print("=== SINGLE_TOOL.LARGE FULL_STACK (OpenAI) ===")
with open('benchmarks/results/baselines_ablations/single_tool.large.full_stack.json') as f:
    st = json.load(f)
s = st['summary']
print(f"  top1={s['top1_accuracy']}, top3={s['top3_accuracy']}, top5={s['top5_accuracy']}")
print(f"  handoff={s['handoff_accuracy_at_recommended_k']}, avg_k={s['avg_recommended_handoff_k']}")
print(f"  n={s['num_items']}")
misses = sum(1 for r in st['results'] if not r.get('hit_at_1', False))
print(f"  misses={misses} ({misses/s['num_items']*100:.1f}% error rate)")

# ── 2. single_tool.clean.report.json (the main benchmark report) ──────────────
print("\n=== SINGLE_TOOL.CLEAN.REPORT.JSON ===")
with open('benchmarks/results/functional_correctness/single_tool.clean.report.json') as f:
    st_report = json.load(f)
s2 = st_report['summary']
print(f"  top1={s2['top1_accuracy']}, top3={s2['top3_accuracy']}, top5={s2['top5_accuracy']}")
print(f"  handoff={s2['handoff_accuracy_at_recommended_k']}, avg_k={s2['avg_recommended_handoff_k']}")
print(f"  n={s2['num_items']}")

# ── 3. multi_hop.large full_stack ─────────────────────────────────────────────
print("\n=== MULTI_HOP.LARGE FULL_STACK (OpenAI) ===")
with open('benchmarks/results/baselines_ablations/multi_hop.large.full_stack.json') as f:
    mh = json.load(f)
s3 = mh['summary']
print(f"  top1={s3['top1_accuracy']}, top5={s3['top5_accuracy']}")
print(f"  handoff={s3['handoff_accuracy_at_recommended_k']}, avg_k={s3['avg_recommended_handoff_k']}")
print(f"  chain@1={s3['chain_success_rate_at_1']}, chain@5={s3.get('chain_success_rate_at_5','?')}")

# ── 4. Confidence tier data ───────────────────────────────────────────────────
print("\n=== CONFIDENCE TIER DATA ===")
with open('benchmarks/results/confidence/single_tool.large.json') as f:
    conf = json.load(f)
print("Keys:", list(conf.keys()))
if 'rows' in conf:
    for row in conf['rows']:
        print(f"  {row}")

# ── 5. Check the main single_tool.clean.live_server.report.json ──────────────
print("\n=== SINGLE_TOOL.CLEAN.LIVE_SERVER.REPORT.JSON ===")
with open('benchmarks/results/functional_correctness/single_tool.clean.live_server.report.json') as f:
    ls = json.load(f)
s4 = ls['summary']
print(f"  top1={s4['top1_accuracy']}, top3={s4['top3_accuracy']}, top5={s4['top5_accuracy']}")
print(f"  handoff={s4['handoff_accuracy_at_recommended_k']}, avg_k={s4['avg_recommended_handoff_k']}")
print(f"  n={s4['num_items']}")
if 'confidence_buckets' in s4:
    for tier, data in s4['confidence_buckets'].items():
        print(f"    [{tier}] count={data['count']}, top1={data['top1_accuracy']:.4f}, handoff={data['handoff_accuracy_at_recommended_k']:.4f}")

# ── 6. Check index file sizes ─────────────────────────────────────────────────
print("\n=== INDEX FILE SIZES ===")
# Look for the index directory
for idx_dir in ['data/index', 'index', 'data/index_full', 'data/index_full_fa']:
    if os.path.exists(idx_dir):
        total = 0
        for root, dirs, files in os.walk(idx_dir):
            for f in files:
                sz = os.path.getsize(os.path.join(root, f))
                total += sz
                print(f"  {os.path.join(root, f)}: {sz/1024/1024:.1f} MB")
        print(f"  TOTAL: {total/1024/1024:.1f} MB")
        break
else:
    print("  Index directory not found in common locations")
    # Try to find .npy files
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.npy') or f == 'meta.json':
                path = os.path.join(root, f)
                sz = os.path.getsize(path)
                if sz > 1000:
                    print(f"  {path}: {sz/1024/1024:.2f} MB")

# ── 7. Bootstrap CI for top-1 ─────────────────────────────────────────────────
print("\n=== BOOTSTRAP CI FOR TOP-1 ===")
with open('benchmarks/results/baselines_ablations/single_tool.large.full_stack.json') as f:
    st = json.load(f)
results = st['results']
n = len(results)
hits = [1 if r.get('hit_at_1', False) else 0 for r in results]
top1 = sum(hits) / n

import random
random.seed(42)
boot_means = []
for _ in range(10000):
    sample = random.choices(hits, k=n)
    boot_means.append(sum(sample)/n)
boot_means.sort()
ci_lo = boot_means[int(0.025*10000)]
ci_hi = boot_means[int(0.975*10000)]
print(f"  Top-1: {top1:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (95% bootstrap CI)")
print(f"  As percentages: {top1*100:.1f}% [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

# ── 8. Verify McNemar p-values ────────────────────────────────────────────────
print("\n=== VERIFY MCNEMAR P-VALUES ===")
def exact_mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    def log_binom(n, k):
        if k < 0 or k > n: return float('-inf')
        r = 0.0
        for i in range(k): r += math.log(n-i) - math.log(i+1)
        return r
    def pmf(n, k): return math.exp(log_binom(n,k) + k*math.log(0.5) + (n-k)*math.log(0.5)) if 0<=k<=n else 0
    p_b = pmf(n, b)
    return min(sum(pmf(n,k) for k in range(n+1) if pmf(n,k) <= p_b+1e-10), 1.0)

# Load full_stack hits
with open('benchmarks/results/baselines_ablations/multi_hop.large.full_stack.json') as f:
    fs_data = json.load(f)
fs_hits = []
for case in fs_data['results']:
    for hop in case.get('hops', []):
        fs_hits.append(1 if hop.get('hit_at_1', False) else 0)
print(f"  full_stack: n={len(fs_hits)}, top1={sum(fs_hits)/len(fs_hits):.4f}")

for profile in ['dense_only', 'bm25_only', 'dense_bm25', 'no_handoff_policy', 'no_session_memory']:
    with open(f'benchmarks/results/baselines_ablations/multi_hop.large.{profile}.json') as f:
        p_data = json.load(f)
    p_hits = []
    for case in p_data['results']:
        for hop in case.get('hops', []):
            p_hits.append(1 if hop.get('hit_at_1', False) else 0)
    b = sum(1 for i in range(len(fs_hits)) if fs_hits[i]==1 and p_hits[i]==0)
    c = sum(1 for i in range(len(fs_hits)) if fs_hits[i]==0 and p_hits[i]==1)
    pval = exact_mcnemar(b, c)
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
    print(f"  {profile}: b={b}, c={c}, p={pval:.6f} -> {sig}")

# ── 9. Check avg_k from confidence data ──────────────────────────────────────
print("\n=== AVG_K FROM CONFIDENCE SINGLE_TOOL.LARGE ===")
with open('benchmarks/results/confidence/single_tool.large.json') as f:
    conf_data = json.load(f)
print("Rows:", conf_data.get('rows', []))
