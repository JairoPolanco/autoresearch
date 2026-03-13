#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUEUE_PATH = Path(__file__).with_name("experiment_queue_recursivellm_v2.json")


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_queue(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, [dict(item) for item in payload]
    if isinstance(payload, dict):
        defaults = dict(payload.get("defaults", {}))
        queue = [dict(item) for item in payload.get("queue", [])]
        return defaults, queue
    raise ValueError(f"unsupported queue payload in {path}")


def _merge_dicts(*parts: dict[str, object] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for part in parts:
        if part:
            out.update(dict(part))
    return out


def _with_base_overrides(base: dict[str, object], delta: dict[str, object] | None) -> dict[str, object]:
    out = dict(base)
    if delta:
        out.update(dict(delta))
    return out


def _resolve_scalar(
    item: dict[str, object],
    defaults: dict[str, object],
    cli_value: int | None,
    key: str,
    fallback: int,
) -> int:
    if key in item:
        return int(item[key])
    if cli_value is not None:
        return int(cli_value)
    if key in defaults:
        return int(defaults[key])
    return int(fallback)


def _run_bakeoff(
    repo: Path,
    *,
    steps: int,
    block_size: int,
    step_offset: int,
    seeds: list[int],
    out_path: Path,
    v2_overrides: dict[str, object],
    v3_overrides: dict[str, object],
    optimizer_overrides: dict[str, object],
    config_v2: str,
    config_v3: str,
) -> dict:
    cmd = [
        "python",
        str(Path(__file__).with_name("run_recursivellm_bakeoff.py")),
        "--repo",
        str(repo),
        "--steps",
        str(int(steps)),
        "--block-size",
        str(int(block_size)),
        "--step-offset",
        str(int(step_offset)),
        "--v2-overrides-json",
        _stable_json(v2_overrides),
        "--v3-overrides-json",
        _stable_json(v3_overrides),
        "--optimizer-overrides-json",
        _stable_json(optimizer_overrides),
        "--out",
        str(out_path),
    ]
    if config_v2:
        cmd.extend(["--config-v2", config_v2])
    if config_v3:
        cmd.extend(["--config-v3", config_v3])
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


def _filter_queue(
    queue: list[dict[str, object]],
    *,
    stages: set[str],
    tags: set[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for item in queue:
        if item.get("enabled", True) is False:
            continue
        item_stage = str(item.get("stage", "default"))
        item_tags = {str(tag) for tag in item.get("tags", [])}
        if stages and item_stage not in stages:
            continue
        if tags and not (item_tags & tags):
            continue
        out.append(item)
    out.sort(key=lambda item: (-int(item.get("priority", 0)), str(item.get("stage", "")), str(item.get("name", ""))))
    return out


def _baseline_context_key(
    *,
    steps: int,
    block_size: int,
    step_offset: int,
    seeds: list[int],
    base_v2_overrides: dict[str, object],
    optimizer_overrides: dict[str, object],
    v3_overrides: dict[str, object],
    config_v2: str,
    config_v3: str,
) -> str:
    payload = {
        "steps": int(steps),
        "block_size": int(block_size),
        "step_offset": int(step_offset),
        "seeds": [int(s) for s in seeds],
        "base_v2_overrides": base_v2_overrides,
        "optimizer_overrides": optimizer_overrides,
        "v3_overrides": v3_overrides,
        "config_v2": config_v2,
        "config_v3": config_v3,
    }
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:12]


def _print_queue(queue: list[dict[str, object]]) -> None:
    for item in queue:
        payload = {
            "stage": item.get("stage", "default"),
            "priority": item.get("priority", 0),
            "name": item.get("name"),
            "tags": item.get("tags", []),
            "description": item.get("description", ""),
        }
        print(json.dumps(payload, ensure_ascii=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic screen for RecursiveLLM model3_v2 hypotheses.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queue-file", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--steps", type=int, default=None, help="screening steps; queue default if omitted")
    ap.add_argument("--final-steps", type=int, default=None, help="confirmation steps; queue default if omitted")
    ap.add_argument("--block-size", type=int, default=None, help="bakeoff block size; queue default if omitted")
    ap.add_argument("--step-offset", type=int, default=None, help="global training-step offset; queue default if omitted")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338])
    ap.add_argument("--stage", action="append", default=[], help="only run queue items from this stage; repeatable")
    ap.add_argument("--tag", action="append", default=[], help="only run queue items with any of these tags; repeatable")
    ap.add_argument("--limit", type=int, default=0, help="limit number of queued candidates, 0 = all")
    ap.add_argument("--list", action="store_true", help="print queue and exit")
    ap.add_argument("--optimizer-overrides-json", default="", help="global optimizer overrides merged into all queue items")
    ap.add_argument("--config-v2", default="")
    ap.add_argument("--config-v3", default="")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    root = Path(__file__).resolve().parent
    results_path = root / "results.tsv"
    queue_path = Path(args.queue_file).resolve()
    queue_defaults, raw_queue = _load_queue(queue_path)
    queue = _filter_queue(raw_queue, stages={str(s) for s in args.stage}, tags={str(t) for t in args.tag})
    if int(args.limit) > 0:
        queue = queue[: int(args.limit)]

    if args.list:
        _print_queue(queue)
        return

    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    global_optimizer = json.loads(args.optimizer_overrides_json) if args.optimizer_overrides_json else {}
    screen_steps_default = int(queue_defaults.get("steps", 16))
    final_steps_default = int(queue_defaults.get("final_steps", 32))
    block_size_default = int(queue_defaults.get("block_size", 32))
    step_offset_default = int(queue_defaults.get("step_offset", 0))
    optimizer_defaults = dict(queue_defaults.get("optimizer_overrides", {}))
    base_v2_overrides_defaults = dict(queue_defaults.get("base_v2_overrides", {}))
    base_v3_overrides_defaults = dict(queue_defaults.get("base_v3_overrides", {}))
    baselines: dict[str, dict] = {}

    def get_baseline(
        *,
        steps: int,
        block_size: int,
        step_offset: int,
        base_v2_overrides: dict[str, object],
        optimizer_overrides: dict[str, object],
        v3_overrides: dict[str, object],
    ) -> dict:
        key = _baseline_context_key(
            steps=steps,
            block_size=block_size,
            step_offset=step_offset,
            seeds=[int(s) for s in args.seeds],
            base_v2_overrides=base_v2_overrides,
            optimizer_overrides=optimizer_overrides,
            v3_overrides=v3_overrides,
            config_v2=args.config_v2,
            config_v3=args.config_v3,
        )
        if key not in baselines:
            out_path = root / f"auto_baseline_{key}.json"
            baselines[key] = _run_bakeoff(
                repo,
                steps=steps,
                block_size=block_size,
                step_offset=step_offset,
                seeds=[int(s) for s in args.seeds],
                out_path=out_path,
                v2_overrides=base_v2_overrides,
                v3_overrides=v3_overrides,
                optimizer_overrides=optimizer_overrides,
                config_v2=args.config_v2,
                config_v3=args.config_v3,
            )
        return baselines[key]

    kept: list[dict[str, object]] = []
    for idx, item in enumerate(queue):
        screen_steps = _resolve_scalar(item, queue_defaults, args.steps, "screen_steps", screen_steps_default)
        final_steps = _resolve_scalar(item, queue_defaults, args.final_steps, "final_steps", final_steps_default)
        block_size = _resolve_scalar(item, queue_defaults, args.block_size, "block_size", block_size_default)
        step_offset = _resolve_scalar(item, queue_defaults, args.step_offset, "step_offset", step_offset_default)
        optimizer_overrides = _merge_dicts(optimizer_defaults, global_optimizer, item.get("optimizer_overrides"))
        base_v2_overrides = _with_base_overrides(base_v2_overrides_defaults, item.get("base_v2_overrides"))
        v2_overrides = _with_base_overrides(base_v2_overrides, item.get("v2_overrides"))
        v3_overrides = _with_base_overrides(base_v3_overrides_defaults, item.get("v3_overrides"))
        baseline = get_baseline(
            steps=screen_steps,
            block_size=block_size,
            step_offset=step_offset,
            base_v2_overrides=base_v2_overrides,
            optimizer_overrides=optimizer_overrides,
            v3_overrides=v3_overrides,
        )
        out_path = root / f"auto_candidate_{idx}_{item['name']}.json"
        candidate = _run_bakeoff(
            repo,
            steps=screen_steps,
            block_size=block_size,
            step_offset=step_offset,
            seeds=[int(s) for s in args.seeds],
            out_path=out_path,
            v2_overrides=v2_overrides,
            v3_overrides=v3_overrides,
            optimizer_overrides=optimizer_overrides,
            config_v2=args.config_v2,
            config_v3=args.config_v3,
        )
        status = _status(candidate, baseline)
        description = (
            f"screen [{item.get('stage', 'default')}] {item['name']}: "
            f"{item.get('description', '')}"
        )
        _append_result(results_path, commit=commit, candidate=candidate, status=status, description=description)
        if status == "keep":
            kept.append(
                {
                    "item": copy.deepcopy(item),
                    "final_steps": final_steps,
                    "block_size": block_size,
                    "step_offset": step_offset,
                    "base_v2_overrides": base_v2_overrides,
                    "optimizer_overrides": optimizer_overrides,
                    "v3_overrides": v3_overrides,
                }
            )

    for idx, entry in enumerate(kept):
        item = dict(entry["item"])
        final_steps = int(entry["final_steps"])
        block_size = int(entry["block_size"])
        step_offset = int(entry["step_offset"])
        base_v2_overrides = dict(entry["base_v2_overrides"])
        optimizer_overrides = dict(entry["optimizer_overrides"])
        v3_overrides = dict(entry["v3_overrides"])
        baseline = get_baseline(
            steps=final_steps,
            block_size=block_size,
            step_offset=step_offset,
            base_v2_overrides=base_v2_overrides,
            optimizer_overrides=optimizer_overrides,
            v3_overrides=v3_overrides,
        )
        out_path = root / f"auto_final_{idx}_{item['name']}.json"
        candidate = _run_bakeoff(
            repo,
            steps=final_steps,
            block_size=block_size,
            step_offset=step_offset,
            seeds=[int(s) for s in args.seeds],
            out_path=out_path,
            v2_overrides=_with_base_overrides(base_v2_overrides, item.get("v2_overrides")),
            v3_overrides=v3_overrides,
            optimizer_overrides=optimizer_overrides,
            config_v2=args.config_v2,
            config_v3=args.config_v3,
        )
        status = _status(candidate, baseline)
        description = (
            f"final [{item.get('stage', 'default')}] {item['name']}: "
            f"{item.get('description', '')}"
        )
        _append_result(results_path, commit=commit, candidate=candidate, status=status, description=description)
        print(
            json.dumps(
                {
                    "name": item["name"],
                    "stage": item.get("stage", "default"),
                    "status": status,
                    "results": candidate["results"]["model3_v2"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
