#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _run_variant(model_cls, cfg, train_data: np.memmap, val_data: np.memmap, *, steps: int, seeds: list[int]) -> dict[str, float | list]:
    rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_cls(cfg)
        opt = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=2.62e-4,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        val0 = _eval_val(model, val_data, block_size=int(cfg.block_size), seed=seed + 1000, step_arg=0)
        train_rng = random.Random(seed)
        step_times: list[float] = []
        for step in range(1, steps + 1):
            x, y = _sample_batch(train_data, block_size=int(cfg.block_size), rng=train_rng)
            t0 = time.perf_counter()
            _, loss, _ = model(x, targets=y, step=step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_times.append((time.perf_counter() - t0) * 1000.0)
        val_end = _eval_val(model, val_data, block_size=int(cfg.block_size), seed=seed + 2000, step_arg=steps)
        rows.append(
            {
                "seed": seed,
                "val_start": val0,
                "val_end": val_end,
                "mean_step_ms": sum(step_times[1:]) / max(1, len(step_times[1:])),
            }
        )
    return {
        "rows": rows,
        "mean_val_start": sum(r["val_start"] for r in rows) / len(rows),
        "mean_val_end": sum(r["val_end"] for r in rows) / len(rows),
        "mean_step_ms": sum(r["mean_step_ms"] for r in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fixed RecursiveLLM model3 v2 vs v3 bakeoff.")
    ap.add_argument("--repo", required=True, help="Path to RecursiveLLM repo")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338])
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

    cfg_v2 = _load_cfg(config_cls, repo / "ouroboros" / "config" / "train_nano_25m_model3_v2.py", vocab_size=vocab_size, block_size=args.block_size)
    cfg_v3 = _load_cfg(config_cls, repo / "ouroboros" / "config" / "train_nano_25m_model3_v3.py", vocab_size=vocab_size, block_size=args.block_size)

    payload = {
        "repo": str(repo),
        "steps": int(args.steps),
        "block_size": int(args.block_size),
        "seeds": [int(s) for s in args.seeds],
        "results": {
            "model3_v2": _run_variant(model_cls, cfg_v2, train_data, val_data, steps=int(args.steps), seeds=[int(s) for s in args.seeds]),
            "model3_v3": _run_variant(model_cls, cfg_v3, train_data, val_data, steps=int(args.steps), seeds=[int(s) for s in args.seeds]),
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
