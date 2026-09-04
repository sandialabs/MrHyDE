#!/usr/bin/env python3
"""Collect linear solve iteration counts from mrhyde .log files; report mean +/- std per log."""

import argparse
import re
import statistics
from pathlib import Path


def parse_log(path: Path) -> list[int]:
    """Extract one iteration count per Belos linear solve block."""
    try:
        text = path.read_text()
    except OSError as e:
        print(f"warning: could not read {path}: {e}")
        return []

    block_delim = re.compile(r"(?m)^\*+ Belos Iterative Solver")
    blocks = block_delim.split(text)
    iters_re = re.compile(r"iters=(\d+)")
    iter_line_re = re.compile(r"Iter\s+(\d+),\s*\[")

    counts = []
    for block in blocks[1:]:  # skip preamble before first Belos header
        m = iters_re.search(block)
        if m:
            counts.append(int(m.group(1)))
        else:
            nums = [int(g) for g in iter_line_re.findall(block)]
            if nums:
                counts.append(max(nums))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect linear solve iteration counts from mrhyde .log files.")
    ap.add_argument("--dir", type=Path, default=None, help="Directory to scan (default: parent of script)")
    args = ap.parse_args()
    solvers_dir = args.dir.resolve() if args.dir else Path(__file__).resolve().parent
    for subdir in sorted(solvers_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for logpath in sorted(subdir.glob("*.log")):
            counts = parse_log(logpath)
            if not counts:
                print(f"{subdir.name}/{logpath.name}: no linear solves found")
                continue
            mean = statistics.mean(counts)
            std = statistics.stdev(counts) if len(counts) > 1 else 0.0
            rel = f"{subdir.name}/{logpath.name}"
            print(f"{rel}: {mean:.1f} +/- {std:.1f}  (n={len(counts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
