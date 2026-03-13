#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _load_cfg(config_cls, path: Path, *, vocab_size: int, block_size: int):
    ns = {"__file__": str(path.resolve())}
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), ns)
    keys = set(getattr(config_cls, "__dataclass_fields__", {}).keys())
    cfg_data = {k: v for k, v in ns.items() if k in keys}
    cfg_data["vocab_size"] = int(vocab_size)
    cfg_data["block_size"] = int(block_size)
    cfg = config_cls(**cfg_data)
    if hasattr(cfg, "validate"):
        cfg.validate()
    return cfg


def _apply_overrides(cfg, overrides: dict[str, object] | None):
    if not overrides:
        return cfg
    for key, value in overrides.items():
        setattr(cfg, key, value)
    if hasattr(cfg, "validate"):
        cfg.validate()
    return cfg


def _sample_batch(mem: np.memmap, *, block_size: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    start = rng.randrange(0, len(mem) - block_size - 1)
    x = torch.from_numpy(np.array(mem[start : start + block_size], dtype=np.int64)).unsqueeze(0)
    y = torch.from_numpy(np.array(mem[start + 1 : start + 1 + block_size], dtype=np.int64)).unsqueeze(0)
    return x, y


def _eval_val(model, mem: np.memmap, *, block_size: int, seed: int, step_arg: int, eval_iters: int = 8) -> float:
    rng = random.Random(seed)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = _sample_batch(mem, block_size=block_size, rng=rng)
            _, loss, _ = model(x, targets=y, step=step_arg)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def _decode_probe_ms(model, *, vocab_size: int, block_size: int) -> float:
    model.eval()
    prefix_len = min(16, max(1, block_size - 2))
    idx = torch.randint(0, int(vocab_size), (1, prefix_len))
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.generate(idx, max_new_tokens=2, top_k=1)
        return (time.perf_counter() - t0) * 1000.0


def _run_variant(
    model_cls,
    cfg,
    train_data: np.memmap,
    val_data: np.memmap,
    *,
    steps: int,
    step_offset: int,
    seeds: list[int],
    optimizer_overrides: dict[str, float] | None,
) -> dict[str, float | list]:
    opt_cfg = dict(optimizer_overrides or {})
    learning_rate = float(opt_cfg.get("learning_rate", 2.62e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.1))
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.95))
    rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_cls(cfg)
        opt = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            device_type="cpu",
        )
        val0 = _eval_val(model, val_data, block_size=int(cfg.block_size), seed=seed + 1000, step_arg=int(step_offset))
        train_rng = random.Random(seed)
        step_times: list[float] = []
        for step in range(1, steps + 1):
            x, y = _sample_batch(train_data, block_size=int(cfg.block_size), rng=train_rng)
            t0 = time.perf_counter()
            _, loss, _ = model(x, targets=y, step=int(step_offset) + step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_times.append((time.perf_counter() - t0) * 1000.0)
        val_end = _eval_val(
            model,
            val_data,
            block_size=int(cfg.block_size),
            seed=seed + 2000,
            step_arg=int(step_offset) + steps,
        )
        decode_ms = _decode_probe_ms(model, vocab_size=int(cfg.vocab_size), block_size=int(cfg.block_size))
        rows.append(
            {
                "seed": seed,
                "val_start": val0,
                "val_end": val_end,
                "mean_step_ms": sum(step_times[1:]) / max(1, len(step_times[1:])),
                "decode_probe_ms": decode_ms,
            }
        )
    return {
        "rows": rows,
        "mean_val_start": sum(r["val_start"] for r in rows) / len(rows),
        "mean_val_end": sum(r["val_end"] for r in rows) / len(rows),
        "mean_step_ms": sum(r["mean_step_ms"] for r in rows) / len(rows),
        "mean_decode_probe_ms": sum(r["decode_probe_ms"] for r in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fixed RecursiveLLM model3 v2 vs v3 bakeoff.")
    ap.add_argument("--repo", required=True, help="Path to RecursiveLLM repo")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338])
    ap.add_argument("--config-v2", default="")
    ap.add_argument("--config-v3", default="")
    ap.add_argument("--v2-overrides-json", default="")
    ap.add_argument("--v3-overrides-json", default="")
    ap.add_argument("--optimizer-overrides-json", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    model3_mod = importlib.import_module("ouroboros.model3")
    config_cls = model3_mod.GPTConfig3
    model_cls = model3_mod.BonsaiPrimeOmega3

    data_dir = repo / "data" / "stage1_fwe"
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    with (data_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    vocab_size = int(meta.get("vocab_size", 50304))

    cfg_v2_path = Path(args.config_v2) if args.config_v2 else (repo / "ouroboros" / "config" / "train_nano_25m_model3_v2.py")
    cfg_v3_path = Path(args.config_v3) if args.config_v3 else (repo / "ouroboros" / "config" / "train_nano_25m_model3_v3.py")
    cfg_v2 = _load_cfg(config_cls, cfg_v2_path, vocab_size=vocab_size, block_size=args.block_size)
    cfg_v3 = _load_cfg(config_cls, cfg_v3_path, vocab_size=vocab_size, block_size=args.block_size)
    cfg_v2 = _apply_overrides(cfg_v2, json.loads(args.v2_overrides_json) if args.v2_overrides_json else None)
    cfg_v3 = _apply_overrides(cfg_v3, json.loads(args.v3_overrides_json) if args.v3_overrides_json else None)
    optimizer_overrides = json.loads(args.optimizer_overrides_json) if args.optimizer_overrides_json else None

    payload = {
        "repo": str(repo),
        "steps": int(args.steps),
        "block_size": int(args.block_size),
        "step_offset": int(args.step_offset),
        "seeds": [int(s) for s in args.seeds],
        "optimizer_overrides": optimizer_overrides or {},
        "results": {
            "model3_v2": _run_variant(
                model_cls,
                cfg_v2,
                train_data,
                val_data,
                steps=int(args.steps),
                step_offset=int(args.step_offset),
                seeds=[int(s) for s in args.seeds],
                optimizer_overrides=optimizer_overrides,
            ),
            "model3_v3": _run_variant(
                model_cls,
                cfg_v3,
                train_data,
                val_data,
                steps=int(args.steps),
                step_offset=int(args.step_offset),
                seeds=[int(s) for s in args.seeds],
                optimizer_overrides=optimizer_overrides,
            ),
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
