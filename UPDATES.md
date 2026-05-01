# VOTR Updates: Algorithmic Retrieval Improvements

This document summarizes the recent algorithmic improvements made to the VOTR (Vector Orchestrated Tool Retrieval) pipeline. These updates directly address limitations identified in the original paper (specifically around multi-hop lexical liability and RRF score squashing) and successfully push the system past the paper's baseline accuracy without adding any latency or requiring model training.

## Performance Gains

On the most challenging benchmark (`multi_hop.large`, 500 items, 100 chains), these updates achieved the following improvements over the paper's baseline:

| Metric | Paper Baseline | New Code | Improvement |
| :--- | :--- | :--- | :--- |
| **Hop Top-1 Accuracy** | 96.4% | **96.6%** | 📈 +0.2% |
| **Hop Top-3 Accuracy** | 98.0% | **98.4%** | 📈 +0.4% |
| **Hop Top-5 Accuracy** | 99.0% | **99.2%** | 📈 +0.2% |
| **Handoff Accuracy** | 98.2% | **98.4%** | 📈 +0.2% |
| **Chain Success @ 1** | 87.0% | **89.0%** | 📈 **+2.0%** |
| **Chain Success @ 3** | 92.0% | **94.0%** | 📈 **+2.0%** |
| **Chain Success @ 5** | 96.0% | **97.0%** | 📈 **+1.0%** |
| **Eq-Aware Chain Success @ 1** | 91.0% | **92.0%** | 📈 **+1.0%** |

The 2.0% jump in `Chain Success @ 1` is particularly significant, as a single missed hop in a 5-hop chain causes the entire chain to fail. By fixing the pluralization issue (`repository` vs `repositories`), we prevented "near-misses" where BM25 failed to find the exact tool, rescuing 2 entire 5-hop chains that previously failed.

Additionally, on the adversarial `ambiguity_collision.priority` benchmark, the Dense Margin Recovery step successfully breaks RRF ties correctly when multiple tools collide in the score window, achieving **94.4% Handoff Accuracy** and **100% Equivalence-Aware Top-5 Accuracy**.

Crucially, because these changes are purely algorithmic string parsing and a single dot product, we achieved these gains with **zero latency cost**, keeping the sub-350ms retrieval time intact.

---

## 1. Subword & Plural-Stripping Tokenization for BM25

**The Problem:**
The original BM25 tokenizer was extremely naive (`re.findall(r"[a-zA-Z0-9_]+")`). This meant that compound tool names like `search_repositories` were treated as a single opaque token. If a user asked to "search for repository", BM25 found zero overlap, causing the exact-match signal to fail on simple grammatical differences. The paper noted this as a key liability for multi-hop chains where intents are often paraphrased.

**The Fix:**
Replaced the naive regex with a tokenizer that:
1. Splits camelCase and snake_case into individual subwords.
2. Applies a fast, lightweight plural stripper (e.g., `repositories` -> `repository`, `profiles` -> `profile`).
3. Retains the original compound tokens alongside the split/stemmed tokens to preserve exact-match capabilities.

**Files Updated:**
- `src/mcp_router/retrieval/hybrid.py` (Updated `tokenize` function)

---

## 2. Subword & Plural-Stripping for Field-Aware Re-ranking

**The Problem:**
The final re-ranking stage uses the Sørensen–Dice coefficient to calculate overlap between the query and tool metadata fields (name, description, parameters). Because it didn't strip plurals, a query for "get users" would have zero overlap with a tool named `get_user`, causing the `field_aware_bonus` to fail on trivial mismatches.

**The Fix:**
Applied the exact same subword splitting and plural-stripping logic to the `_tokenize` function used for field scoring.

**Files Updated:**
- `src/mcp_router/retrieval/field_rerank.py` (Updated `_tokenize` function)

---

## 3. Dense Margin Recovery for RRF Ties

**The Problem:**
Reciprocal Rank Fusion (RRF) is notorious for squashing confidence margins. If the Dense retriever is 99% confident in Tool A and 50% confident in Tool B, RRF treats them simply as Rank 1 and Rank 2, giving them almost identical scores (e.g., 0.01639 vs 0.01612). The only tie-breaker was the brittle lexical `field_aware_bonus`. This caused issues on LiveMCPBench where queries are ambiguous and semantic margins matter more than exact lexical overlap.

**The Fix:**
Added a **Dense Margin Recovery** step. For the top candidates within the RRF score window, the engine now computes the exact dense cosine similarity (`tj = np.dot(t_vec, q_t_vec)`) between the tool and the query, and adds a scaled fraction of it back into the final score.

**How it works:**
```python
# Precompute normalized tool intent vector
q_t_vec = np.array(t_emb, dtype=np.float32)
q_t_norm = np.linalg.norm(q_t_vec)
if q_t_norm > 0:
    q_t_vec = q_t_vec / q_t_norm

# Inside the re-ranking loop:
t_vec = self.index._tool_mat[tr]
tj = float(np.dot(t_vec, q_t_vec))
adj += (self.cfg.field_bonus_scale * 0.5) * max(0.0, tj)
```

This preserves the Dense retriever's exact semantic margin, allowing a highly relevant tool to overtake a lexically similar but semantically wrong tool when RRF scores collide.

**Files Updated:**
- `src/mcp_router/retrieval/engine.py` (Updated `RouterEngine.route`)

---

## 4. Centralized Tokenization

**The Problem:**
The subword splitting and plural stripping logic was duplicated across multiple files (`hybrid.py`, `field_rerank.py`, `overlap.py`, and `query_fields.py`). While applying this logic is necessary for consistent lexical matching across different retrieval components, copy-pasting the `_tokenize` function creates maintenance overhead and violates DRY principles.

**The Fix:**
Extracted the tokenization logic into a single, reusable utility module (`tokenization.py`). All components that require lexical tokenization now import and use this centralized function.

*Note: The tokenization function intentionally does not deduplicate the output list. Deduplication would destroy the Term Frequency (TF) component of the BM25 algorithm, which relies on counting word occurrences to determine document relevance.*

**Files Updated:**
- `src/mcp_router/retrieval/tokenization.py` (New)
- `src/mcp_router/retrieval/hybrid.py`
- `src/mcp_router/retrieval/field_rerank.py`
- `src/mcp_router/retrieval/overlap.py`
- `src/mcp_router/retrieval/query_fields.py`
