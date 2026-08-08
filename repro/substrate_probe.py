#!/usr/bin/env python
"""Inventory the interpreter's package roots before any paid step runs.

Purely diagnostic -- it never fails the run. It exists because a recorded package
list is only as good as the roots the recorder looks at, and this host has TWO:
the workload's own `site-packages` and the base image's system `dist-packages`.
A recorder that reads one and not the other produces a freeze that is portable,
plausible and incomplete, which is the hardest kind of wrong to notice.

It prints three things, all of which belong in the run log rather than in an
operator's memory:

1. how many distributions live under each package root, so the split is visible;
2. the names under the system root, since those are exactly the ones a
   site-packages-only recorder would silently drop;
3. whether importing the workload's own entry point drags in any of the base
   image's cloud-SDK stack. That stack is mutually unsatisfiable against its own
   pinned dependencies, so if it loads it will be recorded, and a recorded
   unsatisfiable set is an honest hard failure at environment setup rather than a
   quiet omission. Knowing which of the two happened is the whole point.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import sys
from collections import defaultdict

SUBSTRATE_PREFIXES = ("sagemaker", "transformer-engine", "transformer_engine")


def root_of(dist: md.Distribution) -> str:
    try:
        loc = str(dist.locate_file(""))
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return "<unknown>"
    return os.path.normpath(loc)


def main() -> int:
    by_root: dict[str, list[str]] = defaultdict(list)
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").strip()
        if not name:
            continue
        by_root[root_of(dist)].append(f"{name}=={dist.version}")

    print("=== package roots visible to this interpreter ===")
    print(f"executable: {sys.executable}")
    for root in sorted(by_root):
        kind = (
            "dist-packages (system/base image)"
            if "dist-packages" in root
            else "site-packages" if "site-packages" in root else "other"
        )
        print(f"  {len(by_root[root]):4d}  {root}   [{kind}]")

    system_roots = [r for r in by_root if "dist-packages" in r]
    print("\n=== distributions under the system root(s) ===")
    if not system_roots:
        print("  (none -- this interpreter has no dist-packages root)")
    for root in sorted(system_roots):
        print(f"  {root}:")
        for pin in sorted(by_root[root], key=str.lower):
            print(f"    {pin}")

    installed_substrate = sorted(
        pin
        for pins in by_root.values()
        for pin in pins
        if pin.lower().startswith(SUBSTRATE_PREFIXES)
    )
    print("\n=== cloud-SDK substrate INSTALLED on this host ===")
    print("  " + (", ".join(installed_substrate) if installed_substrate else "none"))

    print("\n=== does the workload's own import graph load it? ===")
    sys.path.insert(0, os.path.abspath("src"))
    try:
        import lerobot  # noqa: F401
        from lerobot.policies import make_policy  # noqa: F401

        import accelerate  # noqa: F401

        loaded = sorted(
            m for m in sys.modules if m.split(".")[0].lower().startswith(SUBSTRATE_PREFIXES)
        )
        print("  imported: lerobot + lerobot.policies.make_policy + accelerate")
        print("  substrate modules in sys.modules: " + (", ".join(loaded) if loaded else "none"))
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the run
        print(f"  probe import failed (non-fatal): {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
