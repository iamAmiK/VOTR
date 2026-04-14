import json
from pathlib import Path

recs = json.loads(Path("benchmarks/results/conformal_calibration.json").read_text())["all_records"]
ncs = sorted(r["nonconformity_score"] for r in recs)
gaps = sorted(r["score_gap"] for r in recs)
ratios = sorted(
    r["top2_score"] / r["top1_score"] if r["top1_score"] > 1e-9 else 1.0
    for r in recs
)
top1s = sorted(r["top1_score"] for r in recs)

print("GAP distribution:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    idx = int(len(gaps) * p / 100)
    print(f"  p{p}: {round(gaps[idx], 7)}")

print("RATIO (top2/top1) distribution:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    idx = int(len(ratios) * p / 100)
    print(f"  p{p}: {round(ratios[idx], 5)}")

print("TOP1 ABS distribution:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    idx = int(len(top1s) * p / 100)
    print(f"  p{p}: {round(top1s[idx], 5)}")

print("NC distribution:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    idx = int(len(ncs) * p / 100)
    print(f"  p{p}: {round(ncs[idx], 5)}")

hard = [r for r in recs if r["correct_rank"] != 1]
print(f"\nHard cases (rank != 1): {len(hard)}")
for r in hard[:15]:
    ratio = r["top2_score"] / r["top1_score"] if r["top1_score"] > 1e-9 else 1.0
    print(
        f"  rank={r['correct_rank']} gap={round(r['score_gap'],6)}"
        f" ratio={round(ratio,4)} nc={round(r['nonconformity_score'],4)}"
        f" null={r['null_route']}"
    )
