#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


QUEUE = [
    {
        "name": "single_proposal",
        "description": "workspace_num_proposals=1 to reduce proposal softmax overhead",
        "v2_overrides": {
            "workspace_num_proposals": 1,
            "workspace_proposal_temp": 1.0,
        },
    },
    {
        "name": "single_proposal_no_collapse",
        "description": "single proposal plus remove collapse regularizer tax",
        "v2_overrides": {
            "workspace_num_proposals": 1,
            "workspace_proposal_temp": 1.0,
            "workspace_collapse_weight": 0.0,
        },
    },
    {
        "name": "sparse_teacher",
        "description": "critic_sparse planner teacher on v2",
        "v2_overrides": {
            "planner_teacher_mode": "critic_sparse",
            "planner_probe_mode": "first_last",
            "planner_calibration_interval": 1024,
        },
    },
    {
        "name": "summary_rich",
        "description": "richer frontier summaries for v2",
        "v2_overrides": {
            "frontier_summary_slots": 2,
            "frontier_summary_tail_blend": 0.5,
        },
    },
    {
        "name": "single_proposal_sparse_teacher",
        "description": "single proposal plus sparse planner teacher",
        "v2_overrides": {
            "workspace_num_proposals": 1,
            "workspace_proposal_temp": 1.0,
            "planner_teacher_mode": "critic_sparse",
            "planner_probe_mode": "first_last",
            "planner_calibration_interval": 1024,
        },
    },
]


def _run_bakeoff(repo: Path, *, steps: int, seeds: list[int], out_path: Path, v2_overrides: dict[str, object]) -> dict:
    cmd = [
        "python",
        str(Path(__file__).with_name("run_recursivellm_bakeoff.py")),
        "--repo",
        str(repo),
        "--steps",
        str(int(steps)),
        "--out",
        str(out_path),
        "--v2-overrides-json",
        json.dumps(v2_overrides),
    ]
    if seeds:
        cmd.extend(["--seeds", *[str(s) for s in seeds]])
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _status(candidate: dict, baseline: dict) -> str:
    cand = candidate["results"]["model3_v2"]
    base = baseline["results"]["model3_v2"]
    cand_ce = float(cand["mean_val_end"])
    base_ce = float(base["mean_val_end"])
    cand_step = float(cand["mean_step_ms"])
    base_step = float(base["mean_step_ms"])
    cand_decode = float(cand["mean_decode_probe_ms"])
    base_decode = float(base["mean_decode_probe_ms"])
    if cand_ce < base_ce - 0.01:
        return "keep"
    if abs(cand_ce - base_ce) <= 0.01 and (cand_step < base_step * 0.97 or cand_decode < base_decode * 0.97):
        return "keep"
    return "discard"


def _append_result(results_path: Path, *, commit: str, candidate: dict, status: str, description: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    v2 = candidate["results"]["model3_v2"]
    v3 = candidate["results"]["model3_v3"]
    row = (
        f"{ts}\t{commit}\t"
        f"{float(v2['mean_val_end']):.6f}\t{float(v3['mean_val_end']):.6f}\t"
        f"{float(v2['mean_step_ms']):.2f}\t{float(v3['mean_step_ms']):.2f}\t"
        f"{float(v2['mean_decode_probe_ms']):.2f}\t{float(v3['mean_decode_probe_ms']):.2f}\t"
        f"{status}\t{description}\n"
    )
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic screen for RecursiveLLM model3_v2 config hypotheses.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--steps", type=int, default=32, help="screening steps")
    ap.add_argument("--final-steps", type=int, default=64, help="confirmation steps for keep candidates")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338])
    ap.add_argument("--limit", type=int, default=0, help="limit number of queued candidates, 0 = all")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    root = Path(__file__).resolve().parent
    results_path = root / "results.tsv"
    baseline_out = root / "auto_baseline.json"
    baseline = _run_bakeoff(repo, steps=int(args.steps), seeds=[int(s) for s in args.seeds], out_path=baseline_out, v2_overrides={})
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()

    queue = QUEUE[: int(args.limit)] if int(args.limit) > 0 else QUEUE
    kept = []
    for idx, item in enumerate(queue):
        out_path = root / f"auto_candidate_{idx}_{item['name']}.json"
        candidate = _run_bakeoff(
            repo,
            steps=int(args.steps),
            seeds=[int(s) for s in args.seeds],
            out_path=out_path,
            v2_overrides=dict(item["v2_overrides"]),
        )
        status = _status(candidate, baseline)
        _append_result(results_path, commit=commit, candidate=candidate, status=status, description=f"screen {item['name']}: {item['description']}")
        if status == "keep":
            kept.append(item)

    for idx, item in enumerate(kept):
        out_path = root / f"auto_final_{idx}_{item['name']}.json"
        candidate = _run_bakeoff(
            repo,
            steps=int(args.final_steps),
            seeds=[int(s) for s in args.seeds],
            out_path=out_path,
            v2_overrides=dict(item["v2_overrides"]),
        )
        status = _status(candidate, baseline)
        _append_result(results_path, commit=commit, candidate=candidate, status=status, description=f"final {item['name']}: {item['description']}")
        print(json.dumps({"name": item["name"], "status": status, "results": candidate["results"]["model3_v2"]}, indent=2))


if __name__ == "__main__":
    main()
