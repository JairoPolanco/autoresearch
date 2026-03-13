# RecursiveLLM Model3 V2 Experiment Catalog

This is the master backlog for autonomous research on `model3_v2` in:

```text
/Users/jairopolanco/Projects/RecursiveLLM
```

Primary target:
- `ouroboros/config/train_nano_25m_model3_v2.py`

Reference line:
- `ouroboros/config/train_nano_25m_model3_v3.py`

This catalog is intentionally larger than the current queue. It contains:
- experiments that are immediately auto-runnable through config overrides
- experiments that need agent code edits between batches
- deeper research ideas that are mathematically plausible but should stay behind evidence gates

The machine-readable overnight queue lives in:
- `experiment_queue_recursivellm_v2.json`

## Objective

Maximize `model3_v2` on the joint frontier:
- lower validation CE
- lower train step time
- lower decode probe latency
- better quality-per-FLOP

The governing constraint is not "most novelty". It is:

```text
keep only what improves CE, throughput, or decode latency without unacceptable regressions
```

## Current Baseline Understanding

`model3_v2` is strong because it keeps a single-workspace recurrent core and avoids the full dual-workspace tax of `v3`.

Its likely remaining ceiling is not "add many more subsystems". It is:
- make the workspace update rule more efficient
- make planner/control supervision cheaper and cleaner
- improve token refinement only where it buys real quality
- tighten the memory/summarization path without importing `v3`'s full cost structure

### Confirmed Round-1 winners

These are not guesses. They already won in the first overnight sweep.

- Promoted into the live `v2` config:
  - `workspace_num_proposals = 1`
  - `workspace_proposal_temp = 1.0`
  - `workspace_collapse_weight = 0.0`
  - `workspace_mlp_mult = 1`
- Additional kept single experiments:
  - `workspace_slots = 4`
  - `planner_teacher_mode=critic_sparse` + `planner_use_halt_head=False`
  - `n_loops = 3`
  - `n_loops = 2`
  - `n_kv_head = 2`

### Confirmed Round-1 losers

These should not be rerun as isolated singles unless the base changes materially.

- RoPE only
- SwiGLU only
- full `RoPE + GQA + SwiGLU`
- tape-only changes (`tape_recent`, `tape_slots`, `tape_coarse`, `tape_detach_off`)
- two-proposal revival under sparse teacher
- local editor context as a standalone single
- clamp retunes and tiny collapse regularization
- most raw optimizer-only sweeps

## Primary-Source Inspirations

These are included only as idea sources, not as mandates.

- Titans: Learning to Memorize at Test Time
  - https://arxiv.org/abs/2501.00663
  - motivates learned write-strength / carry behavior rather than static memory heuristics
- Transformers are SSMs / Mamba-2 / Structured State Space Duality
  - https://arxiv.org/abs/2405.21060
  - motivates cheaper structured recurrent updates instead of heavier attention everywhere
- Gated Delta Networks
  - https://arxiv.org/abs/2412.06464
  - motivates targeted gated memory writes as a replacement for proposal-mixture complexity
- Mixture-of-Depths
  - https://arxiv.org/abs/2404.02258
  - motivates honest compute routing with real savings instead of decorative control losses
- xLSTM
  - https://arxiv.org/abs/2405.04517
  - motivates exponential-gated memory retention and stronger recurrent inductive bias
- Griffin
  - https://arxiv.org/abs/2402.19427
  - motivates hybrid local-attention + recurrent-memory mixtures when full attention is overkill
- Kimi Linear
  - https://arxiv.org/abs/2510.26692
  - motivates decode-efficient chunkwise recurrent/linear hybrids, but only as a long-range inspiration
- Adaptive Parallel Reasoning
  - https://arxiv.org/abs/2504.15466
  - motivates multiple cheap reasoning trajectories only if they actually beat single-trajectory recurrent updates on QPF
- Learning to Skip
  - https://arxiv.org/abs/2311.15436
  - motivates compute gating tied to measurable utility rather than decorative routing losses
- Long-context token mixing with recent/cache/state decomposition
  - use as a design pattern, not a mandate, for future tape-source routing and recent-vs-coarse memory balance

## Automatic Research Strategy

Run the queue in layers, not all at once:

1. `winner_interactions`
   - compounds the confirmed winners from round 1
2. `chunking`
   - tests whether the block/frontier discretization itself is suboptimal
3. `control_round2`
   - revisits planner/control ideas only on top of the new promoted base
4. `summary_editor_round2`
   - retries richer summaries and local editor context only as interactions
5. `substrate_variants`
   - focuses on KV-head compression variants because GQA was the only substrate family that won
6. `optimizer_round2`
   - only around already-good structural variants
7. `long_shots`
   - only when the rest of the queue is exhausted

Recommended overnight command:

```bash
cd /Users/jairopolanco/Projects/autoresearch
python auto_recursivellm_v2_search.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --queue-file /Users/jairopolanco/Projects/autoresearch/experiment_queue_recursivellm_v2.json \
  --stage winner_interactions \
  --stage chunking \
  --stage control_round2 \
  --steps 16 \
  --final-steps 32 \
  --seeds 1337 1338
```

For a full overnight run:

```bash
python auto_recursivellm_v2_search.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --queue-file /Users/jairopolanco/Projects/autoresearch/experiment_queue_recursivellm_v2.json \
  --steps 16 \
  --final-steps 32 \
  --seeds 1337 1338
```

For a more rigorous late-stage rerun:

```bash
python auto_recursivellm_v2_search.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --queue-file /Users/jairopolanco/Projects/autoresearch/experiment_queue_recursivellm_v2.json \
  --steps 32 \
  --final-steps 64 \
  --seeds 1337 1338 1339
```

## Auto-Runnable Experiment Families

### 1. Planner / Teacher Cost Reduction

Files touched by the mechanism:
- `ouroboros/model3/model.py`
- `ouroboros/model3/planner_v3.py`
- `ouroboros/model3/config.py`

Status:
- sparse critic supervision already proved useful
- the next question is whether it compounds with the structural winners

Hypothesis:
- the planner should remain cheap, sparse, and mostly invisible when it is not buying measurable CE

Round-2 queue items:
- sparse no-halt teacher + `workspace_slots=4`
- sparse no-halt teacher + `n_loops=3`
- sparse no-halt teacher + `n_kv_head=2`
- sparse no-halt teacher + stacked winner combinations
- stride/sample/cost variants only on top of the no-halt base

What success looks like:
- same or better CE
- lower train step time
- positive `compute_savings_frac`

### 2. Compute Budget, Frontier Breadth, and Chunking

Files touched:
- `ouroboros/model3/model.py`
- `ouroboros/model3/config.py`

Status:
- `n_loops=3` and `n_loops=2` both won as singles
- shrinking frontier alone did not

Hypothesis:
- the next high-value question is not only loop count, but whether block size and frontier discretization are mismatched

Round-2 queue items:
- `workspace_slots=4 + n_loops=3`
- `workspace_slots=4 + n_loops=2`
- `workspace_block_tokens=4 + frontier_blocks=8`
- `workspace_block_tokens=16 + frontier_blocks=2`
- `workspace_block_tokens` / `frontier_blocks` interactions with `n_loops=3`, `workspace_slots=4`, and `n_kv_head=2`

What success looks like:
- materially better step time with little or no CE loss

### 3. Workspace Capacity and Update Law

Files touched:
- `ouroboros/model3/workspace.py`
- `ouroboros/model3/config.py`

Hypothesis:
- `v2` may be overpaying for capacity or under-regularizing recurrent updates

Status:
- `workspace_mlp_mult=1` is the best pure quality winner so far
- `workspace_slots=4` also won

Round-2 queue items:
- stack `workspace_slots=4` with the other winners
- only revisit update-law knobs if stacked winners plateau

High-value code ideas after config sweeps:
- learned workspace carry / write-strength gate
- gated-delta workspace update replacing proposal mixtures
- per-slot update gating inspired by Titans and Gated Delta Networks
- factorized or low-rank proposal deltas

### 4. Tape and Summary Bank

Files touched:
- `ouroboros/model3/tape.py`
- `ouroboros/model3/summarizer.py`
- `ouroboros/model3/model.py`

Status:
- tape-only singles all lost
- summary enrichment may still matter as an interaction, not a singleton

Hypothesis:
- richer summaries may only help when the rest of the recurrent core is already cheaper or sharper

Round-2 queue items:
- richer frontier summaries + `workspace_slots=4`
- richer frontier summaries + `n_loops=3`
- richer frontier summaries + `n_kv_head=2`

High-value code ideas:
- summary outputs that expose both content and uncertainty/salience
- typed memory-source embeddings for tape vs retrieval vs live frontier
- incremental cached summary weights if a mathematically safe form is found
- summary dropout or stochastic slot masking to improve robustness

### 5. Editor Expressivity

Files touched:
- `ouroboros/model3/workspace.py`
- `ouroboros/model3/workspace_core.py`

Status:
- local editor context lost as a singleton

Hypothesis:
- it may still help as an interaction when the workspace and loop budgets are already cheaper

Round-2 queue items:
- local context + `workspace_slots=4`
- local context + `n_loops=3`
- local context + richer frontier summaries

High-value code ideas:
- dual-channel reader: separate global-workspace and frontier-workspace readers
- low-rank or grouped-query token-workspace reader
- delta-direction normalization
- editor depth-2 gating only if CE gain justifies latency

### 6. Optimizer Sweeps

Files touched:
- `run_recursivellm_bakeoff.py`

Status:
- raw optimizer-only singles mostly lost

Hypothesis:
- optimizer changes are only worth retesting around already-winning structural profiles

Round-2 queue items:
- higher LR only around `workspace_slots=4`, `n_loops=3`, or `n_kv_head=2`
- lower weight decay only around structural winners
- slower beta2 only around structural winners

Important rule:
- keep optimizer changes only if they are stable over multiple seeds

### 7. Backbone and Attention Substrate

Files touched:
- `ouroboros/model3/blocks.py`
- `ouroboros/model3/config.py`

Status:
- GQA won
- RoPE and SwiGLU lost as singles

Hypothesis:
- the only substrate family worth spending more queue budget on right now is KV-head compression

Round-2 queue items:
- `n_kv_head = 1`
- `n_kv_head = 3`
- interactions of GQA with the winner stack
- RoPE/SwiGLU only as late-stage long shots

This stage is intentionally late because early evidence on `v3` suggested these are not automatically wins in this repo.

## Code-Level Experiments For Agent-Edited Batches

These are not in the overnight JSON queue because they require edits between runs.

### A. Learned Workspace Carry / Write Gate

Target file:
- `ouroboros/model3/workspace.py`

Current issue:
- `WorkspaceCell` uses a gate on the delta itself, but not an explicit learned carry vs overwrite choice

Experiment:
- add a GRU-style update gate
- test:
  - scalar global carry
  - per-slot carry
  - per-channel carry

Best form:

```text
u = old_workspace
c = candidate_update
z = sigmoid(Wz [u ; c])
new = (1 - z) * u + z * c
```

Why it matters:
- cleaner memory retention vs overwrite tradeoff
- likely more parameter-efficient than proposal mixtures

### B. Gated Delta Update Rule

Target file:
- `ouroboros/model3/workspace.py`

Experiment:
- replace proposal-mixture deltas with a delta-rule-style write:

```text
delta = gate(x) * write(x)
workspace = workspace + clamp(delta)
```

Variants:
- simple single-write delta
- low-rank delta
- slot-keyed delta

Why it matters:
- cheaper than proposal softmax mixtures
- directly inspired by Gated Delta Networks
- especially relevant now that single-proposal mode is already the promoted base

### C. Structured Recurrent Update

Target files:
- `ouroboros/model3/workspace.py`
- `ouroboros/model3/blocks.py`

Experiment:
- replace or augment `slot_attn + src_attn` with a state-space-like or gated recurrent update

Low-risk version:
- keep attention readout
- replace part of MLP update with exponential moving-memory gate

Higher-risk version:
- local SSM-style update inside workspace slots

### D. Planner/Executor Simplification

Target files:
- `ouroboros/model3/planner_v3.py`
- `ouroboros/model3/model.py`

Experiments:
- remove halt head permanently if redundant
- predict `k` directly and rank blocks with a smaller head
- reduce planner width or share some projections
- distill exact probes into a cheaper critic target on a subset of steps only

### E. Crown/Probe Cost Reduction

Target files:
- `ouroboros/model3/model.py`
- `ouroboros/model3/blocks.py`

Experiments:
- lighter block-local probe head instead of repeated frontier CE through the full crown
- delayed probe schedule: only every N blocks or N optimizer steps
- distill full probe CE into a lightweight frontier critic

### F. Summary Enrichment Without Tape Bloat

Target files:
- `ouroboros/model3/summarizer.py`

Experiments:
- dual summary slots with one fixed "tail" query and one learned content query
- variance-aware summary with a second moment or norm statistic
- learned query bank for the current block only
- summary residual from the last token representation

### G. Hybrid Local Attention + Workspace Read

Target files:
- `ouroboros/model3/blocks.py`
- `ouroboros/model3/workspace.py`

Experiments:
- let the editor see both workspace context and causal within-block attention
- or let the backbone alternate local attention and workspace-conditioned editing

Why it matters:
- Griffin-style hybridization may give more local reasoning quality without full global-attention cost

## Long-Shot / Research Branch Ideas

These are not for the first overnight runs.

### 1. Slot Specialization Pressure
- entropy regularization or orthogonality losses across workspace slots
- slot dropout
- slot routing based on summary type

### 2. Tape Retrieval Type Embeddings
- distinct embeddings or bias terms for recent slots, coarse slots, and external retrieval slots

### 3. Chunkwise Linear Memory Reader
- a Kimi Linear / SSD-inspired read path from tape to workspace
- only after the current simpler path plateaus

### 4. Adaptive Block Size
- test `workspace_block_tokens = 4`, `8`, `16`
- couple it with frontier width to preserve or intentionally shrink frontier token budget
- this moved up into the round-2 automatic queue because it is no longer just a long shot

### 5. Distilled Control Teacher
- run exact probe calibration periodically
- train a cheap critic head to replace most exact probes
- use critic agreement as the promotion gate

## Suggested Stop Rule

Stop adding mechanism when three consecutive serious ablations each fail to deliver one of:
- at least `0.25%` CE improvement at matched FLOPs
- at least `3%` train-step improvement
- at least `3%` decode improvement

If gains are below that and within seed noise:
- freeze the architecture
- promote the best config
- move remaining ideas to a research branch

## Recommended Promotion Order

1. Promote stacked wins from `winner_interactions`
2. Promote one chunking win only if it survives confirmation
3. Promote one control-stack interaction only if it clearly improves CE or compute savings
4. Promote one summary/editor interaction only if it survives confirmation
5. Treat `optimizer` as profile-specific, not architecture-wide, until repeated
6. Treat RoPE/SwiGLU as guilty until proven useful

## Current Principle

`model3_v2` should win by being a cleaner and cheaper recurrent-memory model, not by slowly accumulating every frontier idea in the literature.
