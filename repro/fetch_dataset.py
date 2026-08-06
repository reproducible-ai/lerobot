#!/usr/bin/env python
"""Fetch a public LeRobotDataset from the Hugging Face Hub into a local tree.

Part of the Reproducible AI Campaign reproduction of LeRobot's ACT policy. Kept
separate from training so the download is its own provenance step: the files this
writes are exactly the files `lerobot-train --dataset.root=<dir>` reads.

The dataset is public, so no token is used or required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="lerobot/pusht_image", help="Hub dataset repo id")
    ap.add_argument("--revision", default=None, help="optional Hub revision to pin")
    ap.add_argument("--out-dir", default="data/pusht_image", help="local dataset root")
    cfg = ap.parse_args()

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # token=False -> resolve anonymously, the same way an outside reproducer would.
    path = snapshot_download(
        repo_id=cfg.repo_id,
        repo_type="dataset",
        revision=cfg.revision,
        local_dir=str(out),
        token=False,
    )
    print(f"downloaded {cfg.repo_id} -> {path}")

    info_path = out / "meta" / "info.json"
    if not info_path.exists():
        print(f"error: {info_path} missing — not a LeRobotDataset tree", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text())
    print(
        "dataset: codebase_version={} episodes={} frames={} tasks={} fps={}".format(
            info.get("codebase_version"),
            info.get("total_episodes"),
            info.get("total_frames"),
            info.get("total_tasks"),
            info.get("fps"),
        )
    )

    total = 0
    for f in sorted(out.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            size = f.stat().st_size
            total += size
            print(f"  {f.relative_to(out)}  {size}")
    print(f"total bytes: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
