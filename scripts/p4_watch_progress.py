#!/usr/bin/env python3
"""Live progress watcher for the Tier 0 run.

Polls the TensorBoard event file every 30s and prints any new epoch's
val_acc / val_loss / train_acc / train_loss / encoder|decoder grad-norms
on a single line. Use as the bottom pane of tmux session 0 while the run
is going on.

Run:
    python scripts/p4_watch_progress.py [optional/path/to/version_dir]
"""

from __future__ import annotations

import glob
import os
import sys
import time
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_GLOB = "lightning_logs/*P4_tier0*/version_*"


def find_latest_version_dir() -> str:
    """Return the most recently modified version dir that actually contains
    a tfevents file. `version_pending` (manifest-only) is skipped because it
    is created by `train.py` *before* Lightning creates the real `version_0`
    and would otherwise win the mtime race on cold start."""
    candidates = sorted(glob.glob(DEFAULT_GLOB), key=os.path.getmtime)
    candidates = [
        d for d in candidates
        if any(f.startswith("events.out.tfevents.") for f in os.listdir(d))
    ]
    if not candidates:
        raise SystemExit(f"no version dir with tfevents matched {DEFAULT_GLOB}; pass an explicit path as argv[1]")
    return candidates[-1]


def value_at(acc: EventAccumulator, tag: str, step: int):
    """Return the value of `tag` at the closest step ≤ `step`. Lightning
    writes some scalars per-epoch (matching `val_acc.step`) but others
    per-N-batches (the grad-norm callback's default 50-step cadence) — so
    an exact match isn't guaranteed."""
    if tag not in acc.Tags().get("scalars", []):
        return None
    best = None
    for e in acc.Scalars(tag):
        if e.step <= step:
            best = e.value
        else:
            break
    return best


def main() -> None:
    version_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_version_dir()
    print(f"# watching: {version_dir}", flush=True)
    print(
        f"{'time':>8s} {'epoch':>5s} {'val_acc':>8s} {'val_loss':>9s} "
        f"{'tr_acc':>7s} {'tr_loss':>9s} {'g_enc':>9s} {'g_dec':>9s}",
        flush=True,
    )

    seen_epochs: set[int] = set()
    while True:
        try:
            acc = EventAccumulator(version_dir, size_guidance={"scalars": 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            if "val_acc" in tags:
                for e in acc.Scalars("val_acc"):
                    if e.step in seen_epochs:
                        continue
                    seen_epochs.add(e.step)
                    epoch_idx = len(seen_epochs) - 1
                    val_loss = value_at(acc, "val_loss", e.step)
                    tr_acc = value_at(acc, "train_acc", e.step)
                    tr_loss = value_at(acc, "train_loss_epoch", e.step)
                    g_enc = value_at(acc, "gradnorm/encoder", e.step)
                    g_dec = value_at(acc, "gradnorm/decoder", e.step)
                    ts = datetime.fromtimestamp(e.wall_time).strftime("%H:%M:%S")
                    print(
                        f"{ts:>8s} {epoch_idx:>5d} {e.value:>8.4f} "
                        f"{val_loss if val_loss is not None else 0:>9.3f} "
                        f"{tr_acc if tr_acc is not None else 0:>7.3f} "
                        f"{tr_loss if tr_loss is not None else 0:>9.3f} "
                        f"{g_enc if g_enc is not None else 0:>9.3f} "
                        f"{g_dec if g_dec is not None else 0:>9.3f}",
                        flush=True,
                    )
        except Exception as exc:  # pragma: no cover — diagnostic loop
            print(f"# watcher exception (will retry): {exc}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
