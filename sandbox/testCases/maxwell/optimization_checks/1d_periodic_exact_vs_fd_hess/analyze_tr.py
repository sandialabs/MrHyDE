#!/usr/bin/env python3
"""Parse mrhyde_r1_{exact,fd}.log and render three diagnostic subplots.

Row 1: ROL objective value vs outer iter (exact vs fd overlaid).
Row 2: Belos iters per time step, as violin distribution over all opt iters
       and DIRK stages (exact vs fd, offset).
Row 3: Belos iters per opt iter, as violin distribution (exact vs fd, offset).

Usage:
    /Users/abvoron/repos/ACEM/env/bin/python3 analyze_tr.py \\
        logs/mrhyde_r1_exact.log logs/mrhyde_r1_fd.log --out analyze_tr.png

The parsers stream each log once. Optional --cache writes a .npz of the parsed
tuples so re-plotting is instant.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# -- regexes and prefixes -----------------------------------------------------
# Cheap string prefixes gate the more expensive regex matches.
PFX_TS_FWD = "**** Beginning Time Step"
PFX_TS_ADJ = "**** Beginning Adjoint Time Step"
PFX_BELOS_HDR = "***** Belos Iterative Solver"
PFX_BELOS_END = "Norm of solution:"
PFX_ROL_HDR = "  iter  value"

RE_ITER = re.compile(r"^Iter\s+(\d+),")
RE_TS = re.compile(r"^\*\*\*\* Beginning (?:Adjoint )?Time Step\s+(\d+)")
RE_ROL_ROW = re.compile(
    r"^\s*(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+"
    r"([\d.eE+\-]+|---)\s+([\d.eE+\-]+|---)"
)


@dataclass
class ParsedLog:
    label: str
    # ROL trust-region iterates.
    rol_iter: list[int] = field(default_factory=list)
    rol_value: list[float] = field(default_factory=list)
    rol_gnorm: list[float] = field(default_factory=list)
    # Per Belos solve: (opt_iter_bucket, time_step_idx, belos_iters)
    #   opt_iter_bucket = index of the most-recent ROL row seen (0..N).
    #   time_step_idx = value of N in "Beginning [Adjoint] Time Step N".
    belos_opt: list[int] = field(default_factory=list)
    belos_ts: list[int] = field(default_factory=list)
    belos_iters: list[int] = field(default_factory=list)


def parse_log(path: Path, label: str) -> ParsedLog:
    """Single-pass streaming parse of a MrHyDE log."""
    p = ParsedLog(label=label)
    # State.
    cur_opt = -1              # index into rol_* lists; -1 = pre-first-iterate
    cur_ts = -1               # time-step index for the currently open block
    in_rol_hdr = False        # last non-blank was the ROL header, expect data row
    in_belos = False          # inside a Belos solve block
    belos_last = -1           # max Iter N inside the current Belos block
    line_no = 0
    t0 = time.time()
    print(f"[{label}] parsing {path}", file=sys.stderr)
    with open(path, "r", errors="replace") as f:
        for line in f:
            line_no += 1
            if line_no % 5_000_000 == 0:
                print(
                    f"[{label}]   {line_no:,} lines "
                    f"({(time.time()-t0):.1f}s)",
                    file=sys.stderr,
                )
            # Cheapest checks first, ordered by expected frequency.
            if in_belos:
                if line.startswith("Iter"):
                    m = RE_ITER.match(line)
                    if m:
                        belos_last = int(m.group(1))
                    continue
                if line.startswith(PFX_BELOS_END):
                    if belos_last >= 0 and cur_ts >= 0:
                        p.belos_opt.append(cur_opt if cur_opt >= 0 else 0)
                        p.belos_ts.append(cur_ts)
                        p.belos_iters.append(belos_last)
                    in_belos = False
                    belos_last = -1
                    continue
                # Sit inside the solve until the terminator arrives.
                continue

            # NOTE: check "*****" (Belos) before "****" (time step) because
            # "*****" also starts with "****".
            if line.startswith("*****"):
                if line.startswith(PFX_BELOS_HDR):
                    in_belos = True
                    belos_last = -1
                continue

            if line.startswith("****"):
                # Time-step banner (forward or adjoint).
                if line.startswith(PFX_TS_ADJ) or line.startswith(PFX_TS_FWD):
                    m = RE_TS.match(line)
                    if m:
                        cur_ts = int(m.group(1))
                continue

            if in_rol_hdr:
                in_rol_hdr = False
                m = RE_ROL_ROW.match(line)
                if m:
                    it = int(m.group(1))
                    try:
                        val = float(m.group(2))
                        gn = float(m.group(3))
                    except ValueError:
                        continue
                    p.rol_iter.append(it)
                    p.rol_value.append(val)
                    p.rol_gnorm.append(gn)
                    cur_opt = len(p.rol_iter) - 1
                continue

            if line.startswith(PFX_ROL_HDR):
                in_rol_hdr = True
                continue

    dt = time.time() - t0
    print(
        f"[{label}] done: {line_no:,} lines in {dt:.1f}s "
        f"-- {len(p.rol_iter)} ROL rows, {len(p.belos_iters):,} Belos solves",
        file=sys.stderr,
    )
    return p


# -- plotting -----------------------------------------------------------------
def _violin_positions(x_centers, offset):
    return np.asarray(x_centers, dtype=float) + offset


def _bin_by(keys, values, key_range):
    """Return list of arrays: values grouped by integer key in key_range."""
    buckets = [[] for _ in key_range]
    lo = key_range[0]
    hi = key_range[-1]
    for k, v in zip(keys, values):
        if lo <= k <= hi:
            buckets[k - lo].append(v)
    return [np.asarray(b, dtype=float) for b in buckets]


def _style_violin(parts, color):
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ("cmins", "cmaxes", "cbars", "cmedians", "cmeans"):
        if key in parts:
            parts[key].set_edgecolor(color)
            parts[key].set_linewidth(0.8)


def _mean_std_label(prefix: str, arr) -> str:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return f"{prefix} (no data)"
    return f"{prefix} (n={a.size:,}, mean={a.mean():.2f}, std={a.std():.2f})"


def make_figure(exact: ParsedLog, fd: ParsedLog, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_exact = "#1f77b4"
    color_fd = "#d62728"

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)

    # -- Row 1: objective value --------------------------------------------
    ax = axes[0]
    ex_last = exact.rol_iter[-1] if exact.rol_iter else -1
    fd_last = fd.rol_iter[-1] if fd.rol_iter else -1
    ax.semilogy(exact.rol_iter, exact.rol_value, "-o", ms=3.5,
                color=color_exact,
                label=f"exact ({len(exact.rol_iter)} rows, last iter={ex_last})")
    ax.semilogy(fd.rol_iter, fd.rol_value, "-s", ms=3.5,
                color=color_fd,
                label=f"fd ({len(fd.rol_iter)} rows, last iter={fd_last})")
    ax.set_xlabel("ROL trust-region iteration")
    ax.set_ylabel("objective value")
    ax.set_title("Objective vs outer iteration")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")

    # Overlay relative difference (fd - exact) / exact on twin axis when the
    # two runs cover the same iteration indices.
    common = min(len(exact.rol_iter), len(fd.rol_iter))
    if common > 1:
        ex_arr = np.asarray(exact.rol_value[:common])
        fd_arr = np.asarray(fd.rol_value[:common])
        it_arr = np.asarray(exact.rol_iter[:common])
        rel = np.abs(fd_arr - ex_arr) / np.abs(ex_arr)
        # Skip iter 0 (rel=0 breaks the log axis).
        ax2 = ax.twinx()
        ax2.semilogy(it_arr[1:], rel[1:], ":", color="k", lw=1.2,
                     label="|fd - exact| / exact")
        ax2.set_ylabel("relative difference (dashed)")
        ax2.tick_params(axis="y", labelsize=8)
        ax2.legend(loc="lower right", fontsize=8)

    if fd_last < ex_last:
        # fd run was killed mid-solve (log ends inside a Belos block); mark it.
        ax.annotate(
            f"fd log truncated after iter {fd_last}",
            xy=(fd_last, fd.rol_value[-1]),
            xytext=(fd_last + 3, fd.rol_value[-1] * 3),
            arrowprops=dict(arrowstyle="->", color=color_fd, lw=0.8),
            color=color_fd, fontsize=9,
        )

    # -- Row 2: Belos iters vs time step -----------------------------------
    ax = axes[1]
    ts_max = max(max(exact.belos_ts, default=0), max(fd.belos_ts, default=0))
    ts_range = list(range(0, ts_max + 1))
    ex_by_ts = _bin_by(exact.belos_ts, exact.belos_iters, ts_range)
    fd_by_ts = _bin_by(fd.belos_ts, fd.belos_iters, ts_range)

    off = 0.22
    ex_pos = [t - off for t, b in zip(ts_range, ex_by_ts) if b.size]
    ex_dat = [b for b in ex_by_ts if b.size]
    fd_pos = [t + off for t, b in zip(ts_range, fd_by_ts) if b.size]
    fd_dat = [b for b in fd_by_ts if b.size]

    if ex_dat:
        parts = ax.violinplot(ex_dat, positions=ex_pos, widths=0.4,
                              showmeans=False, showextrema=True,
                              showmedians=True)
        _style_violin(parts, color_exact)
    if fd_dat:
        parts = ax.violinplot(fd_dat, positions=fd_pos, widths=0.4,
                              showmeans=False, showextrema=True,
                              showmedians=True)
        _style_violin(parts, color_fd)

    ax.set_xlabel("time-step index (fwd + adj, all DIRK stages)")
    ax.set_ylabel("Belos iterations per solve")
    ax.set_title(
        "Linear-solver iteration density per time step "
        "(aggregated over all outer iters)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    # Sparse x-ticks so 200-ish ticks are readable.
    step_tick = max(1, (ts_max + 1) // 20)
    ax.set_xticks(range(0, ts_max + 1, step_tick))
    # Legend proxies with mean/std over all solves.
    ax.plot([], [], color=color_exact, lw=6, alpha=0.55,
            label=_mean_std_label("exact", exact.belos_iters))
    ax.plot([], [], color=color_fd, lw=6, alpha=0.55,
            label=_mean_std_label("fd", fd.belos_iters))
    ax.legend(loc="upper right")

    # -- Row 3: Belos iters vs opt iter ------------------------------------
    ax = axes[2]
    opt_max = max(len(exact.rol_iter), len(fd.rol_iter)) - 1
    opt_range = list(range(0, opt_max + 1))
    ex_by_opt = _bin_by(exact.belos_opt, exact.belos_iters, opt_range)
    fd_by_opt = _bin_by(fd.belos_opt, fd.belos_iters, opt_range)

    ex_pos = [t - off for t, b in zip(opt_range, ex_by_opt) if b.size]
    ex_dat = [b for b in ex_by_opt if b.size]
    fd_pos = [t + off for t, b in zip(opt_range, fd_by_opt) if b.size]
    fd_dat = [b for b in fd_by_opt if b.size]

    if ex_dat:
        parts = ax.violinplot(ex_dat, positions=ex_pos, widths=0.4,
                              showmeans=False, showextrema=True,
                              showmedians=True)
        _style_violin(parts, color_exact)
    if fd_dat:
        parts = ax.violinplot(fd_dat, positions=fd_pos, widths=0.4,
                              showmeans=False, showextrema=True,
                              showmedians=True)
        _style_violin(parts, color_fd)

    ax.set_xlabel("ROL trust-region outer iteration")
    ax.set_ylabel("Belos iterations per solve")
    ax.set_title(
        "Linear-solver iteration density per outer iteration"
    )
    ax.grid(True, axis="y", alpha=0.25)
    step_tick = max(1, (opt_max + 1) // 20)
    ax.set_xticks(range(0, opt_max + 1, step_tick))
    ax.plot([], [], color=color_exact, lw=6, alpha=0.55,
            label=_mean_std_label("exact", exact.belos_iters))
    ax.plot([], [], color=color_fd, lw=6, alpha=0.55,
            label=_mean_std_label("fd", fd.belos_iters))
    ax.legend(loc="upper right")

    fig.suptitle(
        "Trust-region race: exact-Hv vs FD-of-gradients "
        "(1D-periodic Maxwell control)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}", file=sys.stderr)


# -- cache --------------------------------------------------------------------
def save_cache(p: ParsedLog, path: Path) -> None:
    np.savez_compressed(
        path,
        label=np.array(p.label),
        rol_iter=np.asarray(p.rol_iter, dtype=np.int32),
        rol_value=np.asarray(p.rol_value, dtype=np.float64),
        rol_gnorm=np.asarray(p.rol_gnorm, dtype=np.float64),
        belos_opt=np.asarray(p.belos_opt, dtype=np.int32),
        belos_ts=np.asarray(p.belos_ts, dtype=np.int32),
        belos_iters=np.asarray(p.belos_iters, dtype=np.int32),
    )


def load_cache(path: Path) -> ParsedLog:
    z = np.load(path, allow_pickle=False)
    return ParsedLog(
        label=str(z["label"]),
        rol_iter=z["rol_iter"].tolist(),
        rol_value=z["rol_value"].tolist(),
        rol_gnorm=z["rol_gnorm"].tolist(),
        belos_opt=z["belos_opt"].tolist(),
        belos_ts=z["belos_ts"].tolist(),
        belos_iters=z["belos_iters"].tolist(),
    )


def parse_or_load(log_path: Path, label: str, cache_dir: Path | None) -> ParsedLog:
    if cache_dir is not None:
        cache = cache_dir / f"{log_path.stem}.npz"
        if cache.exists():
            print(f"[{label}] cache hit: {cache}", file=sys.stderr)
            return load_cache(cache)
        p = parse_log(log_path, label)
        save_cache(p, cache)
        print(f"[{label}] cache written: {cache}", file=sys.stderr)
        return p
    return parse_log(log_path, label)


# -- main ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exact_log", type=Path, help="mrhyde_r1_exact.log")
    ap.add_argument("fd_log", type=Path, help="mrhyde_r1_fd.log")
    ap.add_argument("--out", type=Path, default=None,
                    help="output figure path (default: analyze_tr.png next to script)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="directory for .npz caches; parse-only if omitted")
    args = ap.parse_args()

    out = args.out or Path(__file__).parent / "analyze_tr.png"
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    exact = parse_or_load(args.exact_log, "exact", args.cache_dir)
    fd = parse_or_load(args.fd_log, "fd", args.cache_dir)

    print(
        f"exact: rol={len(exact.rol_iter)} belos={len(exact.belos_iters):,} "
        f"opt-buckets used={len(set(exact.belos_opt))} "
        f"ts-max={max(exact.belos_ts, default=-1)}",
        file=sys.stderr,
    )
    print(
        f"fd:    rol={len(fd.rol_iter)} belos={len(fd.belos_iters):,} "
        f"opt-buckets used={len(set(fd.belos_opt))} "
        f"ts-max={max(fd.belos_ts, default=-1)}",
        file=sys.stderr,
    )

    make_figure(exact, fd, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
