#!/usr/bin/env python3
"""Parse hess_vec_check_lq logs; write paired plots and summaries."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path(__file__).parent / "logs"

FAMILIES = {
    "exact": "mrhyde_hv_*_exact.log",
    "fd":    "mrhyde_hv_*_fd.log",
}

MERGED_SUMMARY = LOG_DIR / "hess_vec_check_summary_merged.txt"


def parse_fd_tables(path):
    with path.open() as f:
        lines = f.readlines()
    tables = []
    i, n = 0, len(lines)
    while i < n:
        if "Step size" in lines[i]:
            i += 2
            h_vals, err_vals = [], []
            while i < n:
                parts = lines[i].split()
                if len(parts) < 4:
                    break
                try:
                    h = float(parts[0]); err = float(parts[3])
                except ValueError:
                    break
                h_vals.append(h); err_vals.append(err)
                i += 1
            tables.append((np.array(h_vals), np.array(err_vals)))
        else:
            i += 1
    grad = tables[0] if len(tables) > 0 else (np.array([]), np.array([]))
    hv   = tables[1] if len(tables) > 1 else (np.array([]), np.array([]))
    return grad, hv


HSYM_RE     = re.compile(r"<w, H\(x\)v>")
SECANT_RE   = re.compile(r"\[SECANT-IDENTITY\]\s+.*=\s+([\-+eE0-9.]+)\s+relative\s+=\s+([\-+eE0-9.]+)")
HV_ZERO_RE  = re.compile(r"\[HV-ZERO\]\s+\|\|\s*H\*0\s*\|\|\s*=\s*([\-+eE0-9.]+)")
HV_BILIN_RE = re.compile(r"\[HV-BILINEARITY\].*=\s*([\-+eE0-9.]+)\s+relative\s*=\s*([\-+eE0-9.]+)")
HV_RAY_RE   = re.compile(r"\[HV-RAYLEIGH\]\s+<v1, H v1>\s*=\s*([\-+eE0-9.]+)\s+<v2, H v2>\s*=\s*([\-+eE0-9.]+)")


def parse_hsym(path):
    with path.open() as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if HSYM_RE.search(line):
            for j in range(i + 1, min(i + 5, len(lines))):
                parts = lines[j].split()
                if len(parts) >= 3:
                    try:
                        return float(parts[2])
                    except ValueError:
                        continue
    return None


def parse_secant(path):
    with path.open() as f:
        for line in f:
            m = SECANT_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


def parse_algebraic(path):
    result = {"hv0": None, "bilin_rel": None, "ray_min": None}
    with path.open() as f:
        for line in f:
            m = HV_ZERO_RE.search(line)
            if m:
                result["hv0"] = float(m.group(1)); continue
            m = HV_BILIN_RE.search(line)
            if m:
                result["bilin_rel"] = float(m.group(2)); continue
            m = HV_RAY_RE.search(line)
            if m:
                result["ray_min"] = min(float(m.group(1)), float(m.group(2))); continue
    return result


def mode_of(path):
    stem = path.stem[len("mrhyde_hv_r1_"):]
    for suf in ("_exact", "_fd"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _short_title(path):
    t = path.stem
    if t.startswith("mrhyde_hv_r1_"):
        t = t[len("mrhyde_hv_r1_"):]
    return t


def plot_fd(log_paths, index, out_path, ylabel):
    n = len(log_paths)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, log_paths):
        grad, hv = parse_fd_tables(path)
        h, err = (grad, hv)[index]
        if len(h) == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.loglog(h, err, marker="o")
        ax.set_title(_short_title(path), fontsize=9)
        ax.set_xlabel("step size h")
        ax.grid(True, which="both", alpha=0.4)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired(exact_paths, fd_paths, index, out_path, ylabel):
    """modes x 2 grid (exact | fd); shared y per row."""
    ex_by_mode = {mode_of(p): p for p in exact_paths}
    fd_by_mode = {mode_of(p): p for p in fd_paths}
    modes = sorted(set(ex_by_mode) | set(fd_by_mode))
    n = len(modes)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.4 * n), sharex=True, squeeze=False)
    for r, mode in enumerate(modes):
        row_paths = (ex_by_mode.get(mode), fd_by_mode.get(mode))
        axes[r, 1].sharey(axes[r, 0])
        for c, (label, path) in enumerate(zip(("exact", "fd"), row_paths)):
            ax = axes[r, c]
            if path is None:
                ax.text(0.5, 0.5, "(no log)", ha="center", va="center", transform=ax.transAxes)
            else:
                grad, hv = parse_fd_tables(path)
                h, err = (grad, hv)[index]
                if len(h) == 0:
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.loglog(h, err, marker="o")
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{mode}\n{ylabel}", fontsize=8)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if r == n - 1:
                ax.set_xlabel("step size h")
            ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect(family_glob):
    paths = sorted(LOG_DIR.glob(family_glob))
    rows = {}
    for p in paths:
        m = mode_of(p)
        hsym = parse_hsym(p)
        _, s_rel = parse_secant(p)
        alg = parse_algebraic(p)
        rows[m] = {
            "path": p, "hsym": hsym, "secant_rel": s_rel,
            "hv0": alg["hv0"], "bilin_rel": alg["bilin_rel"], "ray_min": alg["ray_min"],
        }
    return rows


def write_family_summary(rows, out_path):
    header = (
        f"{'run':50s}  "
        f"{'|hsym|':>14s}  {'|secant|/ref':>14s}  "
        f"{'|hv0|':>14s}  {'|bilin|/ref':>14s}  {'<v,Hv>_min':>14s}"
    )
    lines = [header, "-" * len(header)]
    def fmt(x): return f"{x:14.4e}" if x is not None else " " * 12 + "--"
    for mode, r in rows.items():
        lines.append(
            f"{mode:50s}  {fmt(r['hsym'])}  {fmt(r['secant_rel'])}  "
            f"{fmt(r['hv0'])}  {fmt(r['bilin_rel'])}  {fmt(r['ray_min'])}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_merged(exact_rows, fd_rows, out_path):
    header = (
        f"{'mode':22s} {'path':6s}  "
        f"{'|hsym|':>12s}  {'|secant|/ref':>14s}  "
        f"{'|hv0|':>10s}  {'|bilin|/ref':>12s}  {'<v,Hv>_min':>14s}"
    )
    lines = [header, "-" * len(header)]
    def fmt(x, w): return f"{x:{w}.4e}" if x is not None else " " * (w - 2) + "--"
    modes = list(exact_rows.keys())
    for mode in modes:
        for label, rows in (("exact", exact_rows), ("fd", fd_rows)):
            r = rows.get(mode, {})
            lines.append(
                f"{mode:22s} {label:6s}  "
                f"{fmt(r.get('hsym'), 12)}  {fmt(r.get('secant_rel'), 14)}  "
                f"{fmt(r.get('hv0'), 10)}  {fmt(r.get('bilin_rel'), 12)}  {fmt(r.get('ray_min'), 14)}"
            )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)


def main():
    per_family = {}
    paths_by_family = {}
    for family, glob in FAMILIES.items():
        paths = sorted(LOG_DIR.glob(glob))
        if not paths:
            print(f"no logs for family {family} matching {glob}")
            continue
        paths_by_family[family] = paths
        per_family[family] = collect(glob)
        write_family_summary(per_family[family], LOG_DIR / f"hess_vec_check_summary_{family}.txt")
        print(f"wrote {family} summary")

    if "exact" in paths_by_family and "fd" in paths_by_family:
        plot_paired(
            paths_by_family["exact"], paths_by_family["fd"], 0,
            LOG_DIR / "grad_check_abs_error_paired.png", "abs error",
        )
        plot_paired(
            paths_by_family["exact"], paths_by_family["fd"], 1,
            LOG_DIR / "hess_vec_check_abs_error_paired.png", "norm(abs error)",
        )
        print("wrote paired PNGs")
        write_merged(per_family["exact"], per_family["fd"], MERGED_SUMMARY)
        print(f"wrote merged {MERGED_SUMMARY}")


if __name__ == "__main__":
    main()
