#!/usr/bin/env python3
"""Belos iterations and TimeMonitor wall times from a MrHyDE log.

    ./parse_log.py mrhyde.log

Needs verbosity: 10.
"""

import math
import sys
from pathlib import Path
import re

HEADER = re.compile(r"^\*+\s*Belos Iterative Solver:\s*(.*?)\s*$")
ITER = re.compile(r"^Iter\s+(\d+),")
VALUE = re.compile(r"([0-9.eE+-]+) \(")
UNCONVERGED = "WARNING: Belos linear solve did not converge"

ITER_TOL_FLOOR = 2
ITER_TOL_FRAC = 0.15

TIMERS = {
    "setup": "MrHyDE::LinearAlgebraInterface::buildPreconditioner()",
    "solve": "MrHyDE::LinearAlgebraInterface::linearSolver*()",
    "fwd": "MrHyDE::SolverManager::forward()",
}


def iterations(lines):
    """Last Iter n in each outer Belos block. Inner Krylov solves are skipped."""
    outer, counts, n = None, [], None
    for line in lines:
        m = HEADER.match(line)
        if m:
            if n is not None:
                counts.append(n)
            outer = outer if outer is not None else m.group(1)
            n = 0 if m.group(1) == outer else None
            continue
        it = ITER.match(line)
        if it and n is not None:
            n = int(it.group(1))
    if n is not None:
        counts.append(n)
    return counts


def timings(lines):
    """Max-over-ranks wall time from the TimeMonitor table."""
    out = {}
    for line in lines:
        parts = re.split(r"\s{2,}", line.rstrip(), maxsplit=1)
        if len(parts) != 2 or parts[0] not in TIMERS.values():
            continue
        v = VALUE.findall(parts[1])
        if v:
            key = next(k for k, label in TIMERS.items() if label == parts[0])
            out[key] = float(v[2] if len(v) >= 3 else v[0])
    return out


def stats(log):
    """Iteration/timing dict, or None if the log is missing or has no solves."""
    log = Path(log)
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    lines = text.splitlines()
    its = iterations(lines)
    if not its:
        return None
    out = {
        "solves": len(its),
        "mean": sum(its) / len(its),
        "max": max(its),
        "unconv": text.count(UNCONVERGED),
        "setup": 0.0,
        "solve": 0.0,
        "fwd": 0.0,
    }
    out.update(timings(lines))
    return out


def band(ref, iter_tol=None):
    """Accepted +/- window around a reference iteration count."""
    if iter_tol is not None:
        return iter_tol
    return max(ITER_TOL_FLOOR, int(math.ceil(ITER_TOL_FRAC * ref)))


def check(solves, mean, imax, log="mrhyde.log", iter_tol=None, res=None):
    """Regression check against recorded counts. 0 on match, 1 with a reason printed."""
    fail = []
    s = stats(log)
    if s is None:
        print("Failure: %s has no Belos solves. Is verbosity 10 set?" % log)
        return 1

    # Exact, not a band: a short count means the time loop died early.
    if s["solves"] != solves:
        fail.append("solves %d, expected %d" % (s["solves"], solves))
    if s["unconv"]:
        fail.append("%d of %d solves hit the iteration limit" % (s["unconv"], s["solves"]))
    mean_tol, max_tol = band(mean, iter_tol), band(imax, iter_tol)
    if abs(s["mean"] - mean) > mean_tol:
        fail.append("mean iters %.2f, expected %.2f +/- %d" % (s["mean"], mean, mean_tol))
    if abs(s["max"] - imax) > max_tol:
        fail.append("max iters %d, expected %d +/- %d" % (s["max"], imax, max_tol))

    detail = "; ".join(fail) if fail else ("%d solves, mean %.2f (+/- %d), max %d (+/- %d)"
                                           % (s["solves"], s["mean"], mean_tol, s["max"], max_tol))
    if res is not None:
        res.add(not fail, "iteration counts", detail)
        return 0
    for reason in fail:
        print("Failure: " + reason)
    if fail:
        return 1
    print("Success.  " + detail)
    return 0


def main():
    logs = sys.argv[1:]
    if not logs:
        sys.exit(__doc__)
    # Keep the tail of the path: it is the part that differs between runs.
    w = min(max(len(x) for x in logs), 60)
    short = lambda x: x if len(x) <= w else "..." + x[3 - w:]

    fmt = "%-*s %7s %7s %5s %7s %8s %8s %8s"
    print(fmt % (w, "log", "solves", "mean", "max", "unconv",
                 "setup s", "solve s", "fwd s"))
    for log in logs:
        s = stats(log)
        if s is None:
            print(fmt % ((w, short(log)) + ("-",) * 7))
            continue
        print(fmt % (w, short(log), s["solves"], "%.2f" % s["mean"], s["max"],
                     s["unconv"], "%.2f" % s["setup"], "%.2f" % s["solve"],
                     "%.2f" % s["fwd"]))


if __name__ == "__main__":
    main()


class Results:
    """PASS/FAIL rows for one test, echoed to stdout and to mrhyde.results."""

    def __init__(self, path="mrhyde.results"):
        self.path, self.rows = path, []

    def add(self, ok, label, detail=""):
        self.rows.append((bool(ok), label, detail))

    def write(self):
        bad = sum(1 for ok, _, _ in self.rows if not ok)
        lines = ["%-4s %-26s %s" % ("PASS" if ok else "FAIL", label, detail)
                 for ok, label, detail in self.rows]
        lines.append("%d of %d checks failed" % (bad, len(self.rows)))
        Path(self.path).write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return bad
