#!/usr/bin/env python3
"""
Distill AWBWNet policy (and optional value head) from RHEA self-play corpus rows.

Replays each ``full_trace`` through the engine, encodes ego-centric observations
for the acting seat, and trains masked cross-entropy on the corpus flat action
indices (including MOVE destinations).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.encoder import encode_state  # noqa: E402
from rl.env import _get_action_mask  # noqa: E402
from rl.game_corpus import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    DEFAULT_MAP_POOL,
    DEFAULT_MAPS_DIR,
    iter_corpus_states,
)
from rl.network import ACTION_SPACE_SIZE, AWBWNet  # noqa: E402


@dataclass
class DistillSample:
    spatial: np.ndarray
    scalars: np.ndarray
    mask: np.ndarray
    action_idx: int
    value_target: float


@dataclass
class DistillResult:
    checkpoint_path: Path
    sidecar_path: Path
    games_trained: int
    steps_trained: int
    train_loss_first_epoch: float
    train_loss_final_epoch: float
    val_top1_accuracy: float
    epoch_losses: list[float] = field(default_factory=list)
    epoch_accuracies: list[float] = field(default_factory=list)


def _value_target_for_seat(winner: Any, acting_seat: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if int(acting_seat) == int(winner) else -1.0


def encode_row_samples(
    row: dict[str, Any],
    *,
    winners_only: bool = False,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
) -> list[DistillSample]:
    """Replay one corpus row and return encoded training samples."""
    winner = row.get("winner")
    samples: list[DistillSample] = []
    for st, flat_idx, acting_seat in iter_corpus_states(
        row, map_pool=map_pool, maps_dir=maps_dir
    ):
        if winners_only and winner is not None and int(acting_seat) != int(winner):
            continue
        spatial, scalars = encode_state(st, observer=acting_seat)
        mask = _get_action_mask(st)
        samples.append(
            DistillSample(
                spatial=spatial.astype(np.float32, copy=False),
                scalars=scalars.astype(np.float32, copy=False),
                mask=mask,
                action_idx=int(flat_idx),
                value_target=_value_target_for_seat(winner, acting_seat),
            )
        )
    return samples


def load_corpus_rows(
    corpus_path: Path,
    *,
    max_games: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_games is not None and len(rows) >= max_games:
                break
    return rows


def build_sample_cache(
    rows: list[dict[str, Any]],
    *,
    winners_only: bool,
    cache_limit: int | None,
    map_pool: Path,
    maps_dir: Path,
) -> tuple[list[DistillSample], int]:
    """Encode corpus rows once; return flat sample list and games used."""
    all_samples: list[DistillSample] = []
    games_used = 0
    for row in rows:
        if cache_limit is not None and games_used >= cache_limit:
            break
        row_samples = encode_row_samples(
            row,
            winners_only=winners_only,
            map_pool=map_pool,
            maps_dir=maps_dir,
        )
        if not row_samples:
            continue
        all_samples.extend(row_samples)
        games_used += 1
    return all_samples, games_used


def _load_init_weights(net: AWBWNet, init_path: Path, device: torch.device) -> None:
    ckpt = torch.load(init_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    net.load_state_dict(sd, strict=False)


def _batch_tensors(
    batch: list[DistillSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial = torch.as_tensor(
        np.stack([s.spatial for s in batch], axis=0), dtype=torch.float32, device=device
    )
    scalars = torch.as_tensor(
        np.stack([s.scalars for s in batch], axis=0), dtype=torch.float32, device=device
    )
    mask = torch.as_tensor(
        np.stack([s.mask for s in batch], axis=0), dtype=torch.bool, device=device
    )
    actions = torch.as_tensor(
        [s.action_idx for s in batch], dtype=torch.long, device=device
    )
    values = torch.as_tensor(
        [s.value_target for s in batch], dtype=torch.float32, device=device
    )
    return spatial, scalars, mask, actions, values


def _run_epoch(
    net: AWBWNet,
    optimizer: torch.optim.Optimizer,
    samples: list[DistillSample],
    *,
    batch_size: int,
    device: torch.device,
    value_loss_weight: float,
    train: bool,
) -> tuple[float, float]:
    if train:
        net.train()
        order = list(range(len(samples)))
        random.shuffle(order)
    else:
        net.eval()
        order = list(range(len(samples)))

    total_loss = 0.0
    total_correct = 0
    n_batches = 0
    n_samples = 0

    with torch.set_grad_enabled(train):
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            batch = [samples[i] for i in idxs]
            spatial, scalars, mask, actions, value_targets = _batch_tensors(batch, device)

            logits, value_pred = net(spatial, scalars, action_mask=mask)
            policy_loss = F.cross_entropy(logits, actions)
            value_loss = F.mse_loss(value_pred, value_targets)
            loss = policy_loss + float(value_loss_weight) * value_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += int((preds == actions).sum().item())
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            n_samples += len(batch)

    mean_loss = total_loss / max(1, n_batches)
    accuracy = total_correct / max(1, n_samples)
    return mean_loss, accuracy


def run_distillation(
    *,
    corpus: Path = DEFAULT_CORPUS_PATH,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
    winners_only: bool = False,
    max_games: int | None = None,
    cache_limit: int | None = None,
    init: Path | None = None,
    out: Path | None = None,
    value_loss_weight: float = 0.5,
    val_fraction: float = 0.1,
    seed: int = 0,
    map_pool: Path = DEFAULT_MAP_POOL,
    maps_dir: Path = DEFAULT_MAPS_DIR,
) -> DistillResult:
    """Train AWBWNet on corpus rows; return metrics and output paths."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")

    rows = load_corpus_rows(corpus, max_games=max_games)
    if not rows:
        raise ValueError(f"no rows in corpus: {corpus}")

    samples, games_used = build_sample_cache(
        rows,
        winners_only=winners_only,
        cache_limit=cache_limit,
        map_pool=map_pool,
        maps_dir=maps_dir,
    )
    if not samples:
        raise ValueError("no training samples after encoding corpus rows")

    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * float(val_fraction))) if len(samples) > 1 else 0
    if n_val >= len(samples):
        n_val = max(1, len(samples) // 5)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        train_samples, val_samples = samples, samples

    dev = torch.device(device)
    net = AWBWNet().to(dev)
    if init is not None:
        _load_init_weights(net, init, dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))

    epoch_losses: list[float] = []
    epoch_accuracies: list[float] = []
    for _epoch in range(int(epochs)):
        loss, _acc = _run_epoch(
            net,
            optimizer,
            train_samples,
            batch_size=int(batch_size),
            device=dev,
            value_loss_weight=float(value_loss_weight),
            train=True,
        )
        epoch_losses.append(loss)

    val_loss, val_acc = _run_epoch(
        net,
        optimizer,
        val_samples,
        batch_size=int(batch_size),
        device=dev,
        value_loss_weight=float(value_loss_weight),
        train=False,
    )
    epoch_accuracies.append(val_acc)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if out is None:
        out = Path(f"checkpoints/policy_distill/awbw_net_distill_{ts}.pt")
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".json")

    torch.save({"state_dict": net.state_dict()}, out)
    sidecar_payload = {
        "corpus": str(corpus.resolve()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(dev),
        "winners_only": bool(winners_only),
        "max_games": max_games,
        "cache_limit": cache_limit,
        "init": str(init.resolve()) if init is not None else None,
        "value_loss_weight": float(value_loss_weight),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "games_trained": int(games_used),
        "steps_trained": int(len(samples)),
        "train_steps": int(len(train_samples)),
        "val_steps": int(len(val_samples)),
        "final_train_loss": float(epoch_losses[-1]) if epoch_losses else None,
        "val_loss": float(val_loss),
        "val_top1_accuracy": float(val_acc),
        "epoch_train_losses": epoch_losses,
        "checkpoint": str(out.resolve()),
        "timestamp": ts,
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

    return DistillResult(
        checkpoint_path=out,
        sidecar_path=sidecar,
        games_trained=games_used,
        steps_trained=len(samples),
        train_loss_first_epoch=epoch_losses[0] if epoch_losses else float("nan"),
        train_loss_final_epoch=epoch_losses[-1] if epoch_losses else float("nan"),
        val_top1_accuracy=val_acc,
        epoch_losses=epoch_losses,
        epoch_accuracies=epoch_accuracies,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--winners-only", action="store_true")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--cache-limit", type=int, default=None)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pt path (default: checkpoints/policy_distill/awbw_net_distill_<ts>.pt)",
    )
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-pool", type=Path, default=DEFAULT_MAP_POOL)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    args = parser.parse_args()

    result = run_distillation(
        corpus=args.corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        winners_only=args.winners_only,
        max_games=args.max_games,
        cache_limit=args.cache_limit,
        init=args.init,
        out=args.out,
        value_loss_weight=args.value_loss_weight,
        val_fraction=args.val_fraction,
        seed=args.seed,
        map_pool=args.map_pool,
        maps_dir=args.maps_dir,
    )
    print(
        f"[distill] games={result.games_trained} steps={result.steps_trained} "
        f"loss {result.train_loss_first_epoch:.4f} -> {result.train_loss_final_epoch:.4f} "
        f"val_top1={result.val_top1_accuracy:.3f}"
    )
    print(f"[distill] wrote {result.checkpoint_path}")
    print(f"[distill] sidecar {result.sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
