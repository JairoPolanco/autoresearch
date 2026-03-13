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

## Automatic Research Strategy

Run the queue in layers, not all at once:

1. `control_teacher`
   - highest ROI
   - attacks planner/probe cost directly
2. `budget`
   - trims loop count and frontier width
3. `workspace`
   - tunes capacity, clamp strength, and recurrent update complexity
4. `tape_summary`
   - memory/summarization tradeoffs
5. `editor`
   - higher-expressivity token refinement
6. `optimizer`
   - only after architecture-side easy wins are harvested
7. `substrate`
   - RoPE/GQA/SwiGLU and other backbone modernizations
8. `interactions`
   - only after winning singles are known

Recommended overnight command:

```bash
cd /Users/jairopolanco/Projects/autoresearch
python auto_recursivellm_v2_search.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --queue-file /Users/jairopolanco/Projects/autoresearch/experiment_queue_recursivellm_v2.json \
  --stage control_teacher \
  --stage budget \
  --stage workspace \
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

Hypothesis:
- the planner should learn from a cheaper sparse critic signal instead of dense exact CE probing

Auto queue items:
- sparse critic teacher with first/last loops
- sparse critic with stride-2 or stride-4 block probes
- sampled sparse probes every 2 or 4 loops
- removing the halt head if it is redundant
- turning on explicit `k_head` or loop-value supervision
- changing planner compute cost

What success looks like:
- same or better CE
- lower train step time
- positive `compute_savings_frac`

### 2. Compute Budget and Frontier Breadth

Files touched:
- `ouroboros/model3/model.py`
- `ouroboros/model3/config.py`

Hypothesis:
- some of `v2`'s quality is already saturated before 4 loops and 4 frontier blocks

Auto queue items:
- `n_loops = 3`
- `n_loops = 2`
- `workspace_frontier_blocks = 3`
- `workspace_frontier_blocks = 2`
- combinations of lower loops and smaller frontier
- executor minimum active blocks = 2
- dense no-executor control

What success looks like:
- materially better step time with little or no CE loss

### 3. Workspace Capacity and Update Law

Files touched:
- `ouroboros/model3/workspace.py`
- `ouroboros/model3/config.py`

Hypothesis:
- `v2` may be overpaying for capacity or under-regularizing recurrent updates

Auto queue items:
- `workspace_slots = 4` and `8`
- `workspace_mlp_mult = 1` and `3`
- tighter and looser update clamps
- reintroducing only a tiny collapse penalty

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

Hypothesis:
- the memory bank is one of the cheapest places to preserve more useful structure

Auto queue items:
- `tape_recent_blocks = 8` and `32`
- `tape_slots_per_block = 1` and `4`
- `tape_coarse_levels = 1` and `3`
- `tape_detach_append = False`
- richer frontier summaries with 2 slots and tail-aware blends

High-value code ideas:
- summary outputs that expose both content and uncertainty/salience
- typed memory-source embeddings for tape vs retrieval vs live frontier
- incremental cached summary weights if a mathematically safe form is found
- summary dropout or stochastic slot masking to improve robustness

### 5. Editor Expressivity

Files touched:
- `ouroboros/model3/workspace.py`
- `ouroboros/model3/workspace_core.py`

Hypothesis:
- local causal context inside the editor may improve token refinement more cheaply than deeper recurrent loops

Auto queue items:
- `editor_use_local_causal_context = True`
- local context combined with richer frontier summaries

High-value code ideas:
- dual-channel reader: separate global-workspace and frontier-workspace readers
- low-rank or grouped-query token-workspace reader
- delta-direction normalization
- editor depth-2 gating only if CE gain justifies latency

### 6. Optimizer Sweeps

Files touched:
- `run_recursivellm_bakeoff.py`

Hypothesis:
- some architecture changes only show up when LR / decay / betas are not inherited from old baselines

Auto queue items:
- LR: `2e-4`, `3.3e-4`, `5e-4`
- weight decay: `0.05`, `0.2`
- beta2: `0.99`
- `(beta1, beta2) = (0.95, 0.99)`
- optimizer + sparse-teacher interaction

Important rule:
- keep optimizer changes only if they are stable over multiple seeds

### 7. Backbone Modernization

Files touched:
- `ouroboros/model3/blocks.py`
- `ouroboros/model3/config.py`

Hypothesis:
- RoPE, GQA, and SwiGLU may help quality-per-bandwidth, but only if they fit this branch

Auto queue items:
- RoPE only
- SwiGLU only
- GQA only
- RoPE + GQA + SwiGLU

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
- possibly coupled with frontier width

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

1. Promote low-risk wins from `control_teacher`
2. Promote one or two wins from `budget`
3. Promote any neutral-to-positive `tape_summary` win
4. Promote `editor` only if the CE gain is consistent
5. Treat `optimizer` as profile-specific, not architecture-wide, until repeated
6. Promote `substrate` only if it wins against the current branch, not because it is fashionable

## Current Principle

`model3_v2` should win by being a cleaner and cheaper recurrent-memory model, not by slowly accumulating every frontier idea in the literature.
