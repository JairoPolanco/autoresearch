# RecursiveLLM Autoresearch Program

This branch adapts the `autoresearch` workflow to `/Users/jairopolanco/Projects/RecursiveLLM`.

## Goal

Advance `model3_v2` until it is the best quality/throughput/FLOP point in the `model3` family.

Primary success criterion:
- lower `model3_v2` validation CE at the end of the fixed benchmark

Secondary success criteria:
- lower `model3_v2` step time
- lower `model3_v2` decode probe time
- eventually clear `model3_v3` on the same benchmark, or match it while staying meaningfully cheaper

Do not accept quality wins that come with disproportionate train-time regressions unless they materially improve the long-run architecture ceiling.

Companion files:
- `research_notes_mar2026.md` — primary-source idea notes
- `experiment_catalog_recursivellm_v2.md` — full backlog, priorities, and stop rules
- `experiment_queue_recursivellm_v2.json` — machine-readable overnight queue

Current promoted `model3_v2` base:
- `workspace_num_proposals = 1`
- `workspace_proposal_temp = 1.0`
- `workspace_collapse_weight = 0.0`
- `workspace_mlp_mult = 1`

The next automatic search should treat those as the fixed base and primarily search interactions on top.

## Scope

Work in this target repo:

```text
/Users/jairopolanco/Projects/RecursiveLLM
```

Primary in-scope files there:
- `ouroboros/model3/*.py`
- `ouroboros/config/train_nano_25m_model3_v2.py`
- `ouroboros/config/train_nano_25m_model3_v3.py`
- `tests/test_model3_basic.py`
- `tests/test_metrics_spec_helpers.py`
- `tests/test_architecture_scorecard.py`
- `tests/test_train_entrypoint_smoke.py`

Do not make unrelated `model`, `model2`, or dataset changes unless they are required for `model3`.

## Baseline Experiment

Run:

```bash
python run_recursivellm_bakeoff.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --steps 64 \
  --seeds 1337 1338
```

This compares:
- primary target: `train_nano_25m_model3_v2.py`
- reference line: `train_nano_25m_model3_v3.py`

using the local `data/stage1_fwe` corpus in the target repo.

## Keep / Discard Rule

Keep a `RecursiveLLM` change only if one of these is true:
- `model3_v2` final validation CE improves versus the previous kept run
- CE is statistically tied, but `model3_v2` step time or decode time improves materially
- `model3_v2` closes the gap to `model3_v3` while preserving its train-time efficiency edge
- the change simplifies the code while staying neutral on the benchmark

Discard if:
- `model3_v2` CE regresses
- step time grows with no compensating quality win
- the change adds mechanism without benchmark evidence

## Logging

Log every run to `results.tsv` in this repo.

Columns:

```text
timestamp	recursivellm_commit	v2_val_ce	v3_val_ce	v2_step_ms	v3_step_ms	v2_decode_ms	v3_decode_ms	status	description
```

Status:
- `keep`
- `discard`
- `crash`

## Notes

- Treat `model3_v2` as the research target.
- Treat `model3_v3` as a frontier reference, not the main branch.
- Prefer exact optimizations and mathematically clean changes over speculative mechanism growth.
- Start from the hypotheses in `research_notes_mar2026.md` before inventing new mechanisms.
- Prefer stacking confirmed winners before reviving discarded singles.
- For unattended runs, use the queue runner instead of editing the hardcoded list:

```bash
python auto_recursivellm_v2_search.py \
  --repo /Users/jairopolanco/Projects/RecursiveLLM \
  --queue-file /Users/jairopolanco/Projects/autoresearch/experiment_queue_recursivellm_v2.json \
  --steps 16 \
  --final-steps 32 \
  --seeds 1337 1338
```

- Run focused stages first:
- Run focused winner-stacking and chunking stages first:

```bash
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
