# RecursiveLLM Model3 V2 Research Notes (March 2026)

This note is intentionally short and tied to the live `model3_v2` codepath in `RecursiveLLM`.

## Objective

Improve `model3_v2` on the quality/throughput/FLOP frontier without turning it into `model3_v3`.

The current `v2` strengths are:
- cheaper train step than `v3`
- simpler recurrent path
- already competitive validation CE

The likely path to winning is not to add many mechanisms. It is to improve the efficiency and quality of the single-workspace recurrent core.

## External Research Signals

### 1. Titans: learning persistent memory updates
Source: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

Why it matters:
- argues for explicit short-term + long-term memory separation
- emphasizes learned memory updates rather than static carry heuristics

Mapping to `model3_v2`:
- `WorkspaceCell` currently uses attention over summaries plus gated proposal deltas
- a learned carry/update gate over workspace state is a plausible low-risk extension

Concrete hypothesis:
- replace fixed update mixing heuristics with a learned per-slot update gate or write-strength gate

### 2. State-space duality / Mamba-2
Source: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)

Why it matters:
- shows that recurrent/state-space style updates can be made much more hardware-efficient
- supports the idea that not all quality gains need more full attention

Mapping to `model3_v2`:
- the workspace update path may benefit more from structured recurrent updates than from a heavier token backbone

Concrete hypothesis:
- test simpler recurrent update laws inside `WorkspaceCell` instead of multi-proposal mixtures
- do not start with a full Mamba rewrite; start with a cheaper gated update ablation

### 3. Gated Delta Networks
Source: [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

Why it matters:
- combines gating with delta-style targeted memory writes
- specifically argues that gating and precise memory modification are complementary

Mapping to `model3_v2`:
- `WorkspaceCell` already has gating plus learned deltas, but not a true delta-rule-style memory write
- `workspace_num_proposals` may be a less efficient way to get expressivity than a better write rule

Concrete hypothesis:
- replace proposal mixtures with a gated delta-style workspace update
- ablate `workspace_num_proposals=1` + improved update rule versus current multi-proposal baseline

### 4. Kimi Linear
Source: [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)

Why it matters:
- shows that hybrid recurrent/linear-memory layers can outperform full attention under fair comparisons
- emphasizes chunkwise hardware-efficient training and strong decode efficiency

Mapping to `model3_v2`:
- good inspiration for future token/workspace hybrids
- not a first ablation target for this repo, because it is too invasive relative to the current code

Concrete hypothesis:
- use this as justification to prefer hybrid recurrent-memory ideas over immediately expanding attention complexity

## Evidence After Round 1

Confirmed wins:
- `workspace_num_proposals = 1`
- `workspace_collapse_weight = 0.0`
- `workspace_mlp_mult = 1`
- `workspace_slots = 4`
- `planner_teacher_mode = critic_sparse` with `planner_use_halt_head = False`
- `n_loops = 3` and `n_loops = 2`
- `n_kv_head = 2`

Confirmed weak or losing singles:
- RoPE
- SwiGLU
- richer tape-only changes
- local editor context as a standalone single

Interpretation:
- the branch still wants cheaper recurrent memory and cheaper control, not a larger or fancier trunk
- the next serious question is whether the winners compound

## Highest-ROI Hypothesis Queue for Model3 V2

1. **Winner stacking**
- Stack `workspace_slots=4`, `n_loops=3`, sparse no-halt teacher, and `n_kv_head=2`
- Goal: find the best combined quality/QPF point

2. **Chunking / frontier discretization**
- Vary `workspace_block_tokens` and `workspace_frontier_blocks` while preserving or intentionally shrinking frontier token budget
- Goal: test whether `v2` is paying for the wrong block granularity

3. **Learned workspace carry / write strength**
- Add a learned gate in `WorkspaceCell` over update magnitude or carry
- Goal: better memory retention vs overwrite without extra loops

4. **Replace proposal mixtures with a cheaper gated-delta update**
- Single-proposal mode is already the new base
- Next step is a more structured gated write instead of proposal softmax mixing
- Goal: better QPF than the current generic update MLP

5. **Selective richer summaries only if they help the winner stack**
- Retry light summary enrichment only on top of already-good profiles
- Goal: preserve more frontier signal without importing `v3`'s cost structure

## Anti-Goals

Do not start with:
- a wholesale `v2 -> v3` migration
- backbone modernization for its own sake
- more controller heads
- larger workspaces or more loops without evidence

## Decision Rule

Keep only changes that improve one of:
- final val CE
- train step time
- decode probe time

without materially regressing the others.
