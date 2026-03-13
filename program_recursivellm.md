# RecursiveLLM Autoresearch Program

This branch adapts the `autoresearch` workflow to `/Users/jairopolanco/Projects/RecursiveLLM`.

## Goal

Advance `model3_v3` until it is the clear default over `model3_v2` on a fixed local bakeoff.

Primary success criterion:
- lower `model3_v3` validation CE at the end of the fixed benchmark

Secondary success criteria:
- lower `model3_v3` step time
- lower `model3_v3` decode probe time

Do not accept quality wins that come with disproportionate train-time regressions unless they materially improve the long-run architecture ceiling.

## Scope

Work in this target repo:

```text
/Users/jairopolanco/Projects/RecursiveLLM
```

Primary in-scope files there:
- `ouroboros/model3/*.py`
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
- `train_nano_25m_model3_v2.py`
- `train_nano_25m_model3_v3.py`

using the local `data/stage1_fwe` corpus in the target repo.

## Keep / Discard Rule

Keep a `RecursiveLLM` change only if one of these is true:
- `model3_v3` final validation CE improves versus the previous kept run
- CE is statistically tied, but `model3_v3` step time or decode time improves materially
- the change simplifies the code while staying neutral on the benchmark

Discard if:
- `model3_v3` CE regresses
- step time grows with no compensating quality win
- the change adds mechanism without benchmark evidence

## Logging

Log every run to `results.tsv` in this repo.

Columns:

```text
timestamp	recursivellm_commit	v2_val_ce	v3_val_ce	v2_step_ms	v3_step_ms	status	description
```

Status:
- `keep`
- `discard`
- `crash`

## Notes

- Treat `model3_v2` as the control.
- Treat `model3_v3` as the research target.
- Prefer exact optimizations and mathematically clean changes over speculative mechanism growth.
