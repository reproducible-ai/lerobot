#!/usr/bin/env python
"""Offline evaluation of a trained LeRobot policy on held-out episodes.

Part of the Reproducible AI Campaign reproduction of LeRobot's ACT policy.

`lerobot-train` can compute a held-out loss inline (`--eval_steps`), but that
number lives only in the training log. This script recomputes it from the saved
checkpoint as its own pipeline step, so the metric is a first-class artifact
(`metrics/metrics.json`) produced by a job that consumes the checkpoint.

It mirrors the eval branch of `src/lerobot/scripts/lerobot_train.py`:
  * the same episode hold-out rule as `make_train_eval_datasets` (the last
    ceil(n_episodes * eval_split) episodes per task),
  * the same uint8 -> float image conversion,
  * the same `loss, _ = policy(preprocessor(batch))` call,
so `eval_loss` here means what `eval_loss` means there. It additionally reports
an unnormalised action L1 error, which is easier to read for a truncated run.

No simulator is involved: this is offline behaviour-cloning error on held-out
demonstrations, not a task success rate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION


def holdout_episodes(meta: LeRobotDatasetMetadata, eval_split: float) -> tuple[list[int], str]:
    """The episodes `make_train_eval_datasets` would hold out, and how we got them."""
    all_eps = list(range(meta.total_episodes))
    try:
        episode_tasks = meta.episodes["tasks"]
        by_task: dict[str, list[int]] = {}
        for ep in all_eps:
            key = episode_tasks[ep][0] if episode_tasks[ep] else ""
            by_task.setdefault(key, []).append(ep)
        held: list[int] = []
        for eps in by_task.values():
            n_eval = math.ceil(len(eps) * eval_split)
            held.extend(eps[len(eps) - n_eval :])
        return sorted(held), f"per-task ({len(by_task)} task(s))"
    except Exception as exc:  # pragma: no cover - metadata shape fallback
        n_eval = math.ceil(len(all_eps) * eval_split)
        return all_eps[len(all_eps) - n_eval :], f"global tail (per-task grouping failed: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="…/checkpoints/<step>/pretrained_model")
    ap.add_argument("--repo-id", default="lerobot/pusht_image")
    ap.add_argument("--root", default="data/pusht_image", help="local dataset root")
    ap.add_argument("--eval-split", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all held-out batches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="metrics/metrics.json")
    cfg = ap.parse_args()

    ckpt = Path(cfg.checkpoint)
    if not (ckpt / "config.json").exists():
        raise SystemExit(f"error: no config.json in {ckpt}")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    eval_episodes, split_how = holdout_episodes(meta, cfg.eval_split)
    print(f"held-out episodes: {len(eval_episodes)} of {meta.total_episodes} ({split_how})")

    policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy_cfg.pretrained_path = ckpt
    policy_cfg.device = device.type

    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=cfg.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    print(f"held-out frames: {len(dataset)}")

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lerobot_collate_fn,
        drop_last=False,
    )

    camera_keys = list(dataset.meta.camera_keys)
    loss_sum, l1_sum, n_batches, n_samples = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if cfg.max_batches and n_batches >= cfg.max_batches:
                break
            target = batch[ACTION].detach().float().cpu()
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
            processed = preprocessor(batch)
            loss, _ = policy(processed)
            loss_sum += float(loss.item())

            # The postprocessor un-normalises and hands the action back on CPU
            # (it is the robot-facing end of the pipeline), while the batch tensors
            # are on the accelerator — so compare on CPU rather than assuming either.
            action = postprocessor(policy.predict_action_chunk(processed))
            action = action.detach().float().cpu()
            l1_sum += float(
                torch.nn.functional.l1_loss(action[:, : target.shape[1]], target).item()
            )

            n_batches += 1
            n_samples += int(target.shape[0])

    if n_batches == 0:
        raise SystemExit("error: no held-out batches were evaluated")

    metrics = {
        "policy": policy_cfg.type,
        "checkpoint": str(ckpt),
        "dataset": cfg.repo_id,
        "split": {
            "eval_split": cfg.eval_split,
            "rule": split_how,
            "held_out_episodes": len(eval_episodes),
            "total_episodes": meta.total_episodes,
        },
        "eval_batches": n_batches,
        "eval_samples": n_samples,
        "eval_loss": loss_sum / n_batches,
        "action_l1": l1_sum / n_batches,
        "note": (
            "TRUNCATED run — the policy is far from converged; these numbers "
            "demonstrate that a metric is computed, not model quality."
        ),
    }
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
