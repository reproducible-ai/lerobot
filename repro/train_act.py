#!/usr/bin/env python
"""Launcher for upstream's `lerobot-train` entry point, run straight from the checkout.

Part of the Reproducible AI Campaign reproduction of LeRobot's ACT policy. This
adds no training logic: it puts `src/` on `sys.path` and calls
`lerobot.scripts.lerobot_train.main()`, so every argument is upstream's own and
behaves exactly as `lerobot-train` does.

It exists so the *recorded* pipeline step is self-sufficient, which the two
obvious alternatives are not:

* `lerobot-train` is a console script created by `pip install`. The install runs
  in the untraced setup stage, so it is not part of the recorded pipeline and is
  not replayed on a rebuild host -- where the script would simply not exist.
  Recording a `lerobot==...` pin does not help either: this checkout declares a
  version that is not published on PyPI.
* `PYTHONPATH=src python -m lerobot.scripts.lerobot_train` depends on an
  environment variable set outside the recorded command, and wraps the workload
  in a shell, which costs the tracer its view of the Python process.

Doing the path insert *inside committed code* means the rebuild host gets it
from the clone, with nothing to remember and nothing to replay.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class _Tee:
    """Duplicate a stream to a log file, so the training log is a real artifact."""

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main() -> None:
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    log_path = os.environ.get("REPRO_TRAIN_LOG", os.path.join(REPO_ROOT, "logs", "train.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", buffering=1) as handle:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, handle), _Tee(stderr, handle)
        try:
            lerobot_train_main()
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
