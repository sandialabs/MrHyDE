#!/usr/bin/env python3
"""Summarize mean linear iterations per (solver, dt, dx) from mrhyde_t*_r*.log files.
Pass subdir to scan (e.g. solvers_conv_check). Optional --out for heatmap PDF.
Notation: t# = \Delta t refinements, r# = spatial refinement level (mesh size h)."""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from linear_iter_stats import parse_log
from parse_mrhyde_log import parse_log as parse_mrhyde_log

LOG_PAT = re.compile(r"mrhyde_t(\d+)_r(\d+)\.log")


def _short_name(solver: str) -> str:
    s = solver
    for prefix in ("bicgbstab_blocktri_", "bicgbstab_blockdiag_", "bicgbstab_monolith_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if s.startswith("bicgbstab_"):
        s = s[10:]
    return s.replace("_", "-")[:14]


def _solver_ylabel(solver: str) -> str:
    """Compact directory name for heatmap y-axis."""
    label = solver
    if label.endswith("_monolith_jac"):
        label = label[: -len("_monolith_jac")]
    for prefix in ("bicgbstab_", "bicgstab_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    return label


def _y_label(solver: str) -> str:
    """Label for y-axis."""
    return _solver_ylabel(solver)


def _is_test_dir(d: Path) -> bool:
    """Dir is a test dir if it has mrhyde.tst or any mrhyde_t*_r*.log."""
    if not d.is_dir():
        return False
    if (d / "mrhyde.tst").exists():
        return True
    return next(d.glob("mrhyde_t*_r*.log"), None) is not None


def get_test_dirs(root: Path) -> list[Path]:
    dirs = [d for d in root.iterdir() if _is_test_dir(d)]
    if not dirs and _is_test_dir(root):
        return [root]
    return dirs


def _log_converged(logpath: Path) -> bool:
    """True if run completed (TimeMonitor results printed)."""
    try:
        return "TimeMonitor results" in logpath.read_text()
    except OSError:
        return False


def _get_belos_solve_time(logpath: Path) -> float | str:
    """Return Belos total solve time (MeanOverProcs/GlobalTime), or 'n/a'."""
    try:
        _, timing_df = parse_mrhyde_log(logpath)
    except Exception:
        return "n/a"
    if timing_df.empty:
        return "n/a"
    names = timing_df["timer_name"].fillna("").str.strip()
    mask = names.str.contains(r"^Belos: .*total solve time$", regex=True)
    row = timing_df[mask]
    if row.empty:
        return "n/a"
    row = row.iloc[0]
    if "MeanOverProcs" in row and row.get("MeanOverProcs") is not None:
        return float(row["MeanOverProcs"])
    if "GlobalTime" in row and row.get("GlobalTime") is not None:
        return float(row["GlobalTime"])
    return "n/a"


CellValue = tuple[float, int, bool] | None  # (mean_iters, n_solves, converged) or no data


def _cell_text(val: CellValue) -> str:
    """Format cell for table/heatmap: converged -> mean; failed+counts -> (mean/n); else n/a."""
    if val is None:
        return "n/a"
    mean, n, converged = val
    if converged:
        return f"{mean:.1f}"
    return f"({mean:.1f}/{n})"


def _cell_n_solves(val: CellValue) -> int:
    """Number of linear solves for this cell."""
    return val[1] if val is not None else 0


def _converged_n_solves_for_column(data: dict[str, dict[tuple[int, int], CellValue]], solvers: list[str], ij: tuple[int, int]) -> int | None:
    """n_solves from one converged run in this column; None if no converged run."""
    for s in solvers:
        val = data.get(s, {}).get(ij)
        if val is not None and val[2]:  # converged
            return val[1]
    return None


def discover_logs(root: Path) -> tuple[dict[str, dict[tuple[int, int], CellValue]], dict[str, dict[tuple[int, int], float | str]], set[tuple[int, int]]]:
    """Scan test dirs for mrhyde_t*_r*.log; return (data_iters, data_time, all_ij)."""
    test_dirs = get_test_dirs(root)
    data: dict[str, dict[tuple[int, int], CellValue]] = {}
    data_time: dict[str, dict[tuple[int, int], float | str]] = {}
    all_ij: set[tuple[int, int]] = set()

    for d in sorted(test_dirs):
        solver = d.name
        data[solver] = {}
        data_time[solver] = {}
        for logpath in d.glob("mrhyde_t*_r*.log"):
            m = LOG_PAT.fullmatch(logpath.name)
            if not m:
                continue
            i, j = int(m.group(1)), int(m.group(2))
            all_ij.add((i, j))
            counts = parse_log(logpath)
            converged = _log_converged(logpath)
            if counts:
                data[solver][(i, j)] = (statistics.mean(counts), len(counts), converged)
            else:
                data[solver][(i, j)] = None
            data_time[solver][(i, j)] = _get_belos_solve_time(logpath) if converged else "n/a"
    return data, data_time, all_ij


def _load_sweep_meta(root: Path) -> dict | None:
    """Load sweep metadata written by run_mesh_sweep.py, if present."""
    meta_path = root / "sweep_meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _plot_heatmaps(
    data: dict[str, dict[tuple[int, int], CellValue]],
    all_ij: set[tuple[int, int]],
    solvers: list[str],
    nsteps_list: list[int],
    ref_list: list[int],
    sweep_meta: dict | None,
    out_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import colors
    except ImportError:
        print("matplotlib required for --plot; install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    ti_vals = sorted({ij[0] for ij in all_ij})
    rj_vals = sorted({ij[1] for ij in all_ij})

    time_meta_map: dict[int, dict] = {}
    spatial_meta_map: dict[int, dict] = {}
    if sweep_meta:
        for entry in sweep_meta.get("time", []):
            idx = entry.get("index")
            if isinstance(idx, int):
                time_meta_map[idx] = entry
        for entry in sweep_meta.get("spatial", []):
            idx = entry.get("index")
            if isinstance(idx, int):
                spatial_meta_map[idx] = entry
    cmap = plt.colormaps["cividis"].copy()
    cmap.set_bad(color="white")
    n_t = len(ti_vals)
    n_r = len(rj_vals)
    fig, axes = plt.subplots(1, n_t, figsize=(4 * n_t, 1.2 * len(solvers)), squeeze=False)
    axes = axes[0]
    ylabels = [_solver_ylabel(s) for s in solvers]
    ylim = (-0.5, len(solvers) - 0.5)
    for ax_idx, t in enumerate(ti_vals):
        ax = axes[ax_idx]
        H = np.full((len(solvers), n_r), np.nan)
        for s_idx, s in enumerate(solvers):
            for r_idx, r in enumerate(rj_vals):
                v = data.get(s, {}).get((t, r))
                if v is not None:
                    H[s_idx, r_idx] = float(v[0])
        vmin, vmax = np.nanmin(H), np.nanmax(H)
        if np.isnan(vmin):
            vmin, vmax = 0, 1
        else:
            vmax = max(vmax, vmin + 1e-6)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(H, aspect="auto", norm=norm, cmap=cmap)
        ax.set_ylim(ylim[1], ylim[0])
        ax.set_xticks(np.arange(n_r))
        if spatial_meta_map and all(r in spatial_meta_map for r in rj_vals):
            xticklabels: list[str] = []
            for r in rj_vals:
                e = spatial_meta_map[r]
                nx = e.get("nx")
                ny = e.get("ny")
                nz = e.get("nz")
                xticklabels.append(rf"$\Omega_{r} = ({nx},{ny},{nz})$")
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        else:
            ax.set_xticklabels([str(ref_list[r]) if r < len(ref_list) else str(r) for r in rj_vals])
        ax.set_yticks(np.arange(len(solvers)))
        if ax_idx == 0:
            ax.set_yticklabels(ylabels, fontsize=8)
        else:
            ax.set_yticklabels([""] * len(solvers))
        if time_meta_map and t in time_meta_map:
            dt_val = time_meta_map[t].get("dt")
            if dt_val is not None:
                title = rf"$\Delta t_{t} = {dt_val}$"
            else:
                title = rf"$\Delta t_{t}$"
        elif nsteps_list and t < len(nsteps_list):
            title = f"# of $\\Delta t$ refinements (index {t})"
        else:
            title = f"# of $\\Delta t$ refinements = {t}"
        ax.set_title(title)
        for s_idx in range(len(solvers)):
            for r_idx in range(n_r):
                v = data.get(solvers[s_idx], {}).get((t, rj_vals[r_idx]))
                txt = _cell_text(v)
                cell_val = H[s_idx, r_idx]
                if np.isnan(cell_val):
                    text_color = "black"
                else:
                    nval = float(norm(cell_val))
                    text_color = "white" if nval < 0.45 else "black"
                ax.text(r_idx, s_idx, txt, ha="center", va="center", fontsize=7, color=text_color)
        cbar = fig.colorbar(
            im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.28, location="bottom",
            label="Mean linear iterations (lower is better)",
        )
    fig.suptitle(
        "Linear solver iterations vs mesh refinement and time step",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    #fig.text(0.5, 0.02, "spatial refinements", ha="center", fontsize=10)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
    plt.close(fig)


def _plot_solve_time_panels(
    data: dict[str, dict[tuple[int, int], float | str]],
    all_ij: set[tuple[int, int]],
    solvers: list[str],
    nsteps_list: list[int],
    ref_list: list[int],
    sweep_meta: dict | None,
    out_path: Path | None,
) -> None:
    """One panel per dx, x=dt, y=Belos solve time (s)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required; pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    ti_vals = sorted({ij[0] for ij in all_ij})
    rj_vals = sorted({ij[1] for ij in all_ij})

    time_meta_map: dict[int, dict] = {}
    spatial_meta_map: dict[int, dict] = {}
    if sweep_meta:
        for entry in sweep_meta.get("time", []):
            idx = entry.get("index")
            if isinstance(idx, int):
                time_meta_map[idx] = entry
        for entry in sweep_meta.get("spatial", []):
            idx = entry.get("index")
            if isinstance(idx, int):
                spatial_meta_map[idx] = entry
    n_t, n_r = len(ti_vals), len(rj_vals)
    fig, axes = plt.subplots(1, n_r, figsize=(4 * n_r, 4), squeeze=False)
    axes = axes[0]
    color_list = plt.cm.tab10(np.linspace(0, 1, max(len(solvers), 1)))[: len(solvers)]
    for ax_idx, r in enumerate(rj_vals):
        ax = axes[ax_idx]
        for s_idx, solver in enumerate(solvers):
            xs, ys = [], []
            for t_idx, t in enumerate(ti_vals):
                v = data.get(solver, {}).get((t, r), "n/a")
                if isinstance(v, (int, float)):
                    xs.append(t_idx)
                    ys.append(float(v))
            if xs:
                ax.semilogy(xs, ys, "o-", label=_solver_ylabel(solver), color=color_list[s_idx])
        ax.set_xticks(range(n_t))
        if time_meta_map:
            labels: list[str] = []
            for t in ti_vals:
                e = time_meta_map.get(t)
                if e and "dt" in e:
                    labels.append(rf"$\Delta t_{t} = {e['dt']}$")
                else:
                    labels.append(str(t))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.set_xticklabels(
                [str(nsteps_list[ti_vals[i]]) if nsteps_list and i < len(nsteps_list) else str(ti_vals[i]) for i in range(n_t)]
            )
        final_time = sweep_meta.get("final_time") if sweep_meta else None
        if final_time is not None:
            ax.set_xlabel(rf"Time step size ($T_{{\max}} = {final_time}$)")
        else:
            ax.set_xlabel("Time step size")
        ax.set_ylabel("Belos solve time (s)")
        if spatial_meta_map and r in spatial_meta_map:
            e = spatial_meta_map[r]
            nx = e.get("nx")
            ny = e.get("ny")
            nz = e.get("nz")
            title = rf"$\Omega_{r} = ({nx},{ny},{nz})$"
        elif ref_list and r < len(ref_list):
            title = f"Mesh refinement level {ref_list[r]}"
        else:
            title = f"# of $\\Delta x$ refinements = {r}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Belos solve time vs time step, by mesh refinement",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(solvers)//2, fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path, help="Path to directory to scan (relative to cwd or absolute)")
    ap.add_argument("--nsteps", default="", help="Comma-separated nsteps for column labels (optional)")
    ap.add_argument("--ref", default="", help="Comma-separated ref levels for column labels (optional)")
    ap.add_argument("--out", type=Path, default=Path("iteration_count_summary.pdf"), help="Save iteration heatmap to path")
    ap.add_argument("--out-time", type=Path, default=Path("solve_time_summary.pdf"), help="Save solve-time figure to path")
    ap.add_argument("--exclude-time", default="", help="Exclude preconditioners whose name contains this substring from the timings plot and solve-time table")
    args = ap.parse_args()
    root = Path(args.dir).resolve() if args.dir.is_absolute() else (Path.cwd() / args.dir).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    data, data_time, all_ij = discover_logs(root)
    if not all_ij:
        print("No mrhyde_t*_r*.log files found.", file=sys.stderr)
        return 1

    test_dirs = get_test_dirs(root)
    solvers = sorted(d.name for d in test_dirs)
    columns = sorted(all_ij, key=lambda x: (x[0], x[1]))  # (j, i): group by refinement

    nsteps_list = [int(x.strip()) for x in args.nsteps.split(",") if x.strip()] if args.nsteps else []
    ref_list = [int(x.strip()) for x in args.ref.split(",") if x.strip()] if args.ref else []

    sweep_meta = _load_sweep_meta(root)
    solvers_for_time = [s for s in solvers if args.exclude_time not in s] if args.exclude_time else solvers

    _plot_heatmaps(data, all_ij, solvers, nsteps_list, ref_list, sweep_meta, args.out)
    _plot_solve_time_panels(data_time, all_ij, solvers_for_time, nsteps_list, ref_list, sweep_meta, args.out_time)

    def col_header(ij: tuple[int, int]) -> str:
        i, j = ij
        if nsteps_list and ref_list and i < len(nsteps_list) and j < len(ref_list):
            return f"nsteps={nsteps_list[i]}_ref{j}"
        return f"time{i}_ref{j}"

    headers = ["solver"] + [col_header(ij) for ij in columns]
    widths = [max(len(h), 4) for h in headers]
    for s in solvers:
        widths[0] = max(widths[0], len(s))
    for c, ij in enumerate(columns):
        for s in solvers:
            val = data.get(s, {}).get(ij)
            cell = _cell_text(val)
            widths[c + 1] = max(widths[c + 1], len(cell))
    total_label = "total (linear solves)"
    widths[0] = max(widths[0], len(total_label))
    total_cell_strs: list[str] = []
    for c, ij in enumerate(columns):
        n = _converged_n_solves_for_column(data, solvers, ij)
        s = str(n) if n is not None else "-"
        total_cell_strs.append(s)
        widths[c + 1] = max(widths[c + 1], len(s))

    def row(cells: list[str]) -> str:
        return "  ".join(cells[k].rjust(widths[k]) for k in range(len(cells)))

    print(row(headers))
    print(row(["-" * w for w in widths]))
    for s in solvers:
        cells = [s]
        for ij in columns:
            val = data.get(s, {}).get(ij)
            cells.append(_cell_text(val))
        print(row(cells))
    print(row(["-" * w for w in widths]))
    print(row([total_label] + total_cell_strs))

    print("\n\n(X/Y) = (failed runs) = (Average iterations per solve / # of solves)")

    # Solve-time table
    time_widths = [max(len(h), 4) for h in headers]
    for s in solvers_for_time:
        time_widths[0] = max(time_widths[0], len(s))
    for ij in columns:
        for s in solvers_for_time:
            val = data_time.get(s, {}).get(ij, "n/a")
            cell = f"{float(val):.2f}" if isinstance(val, (int, float)) else str(val)
            idx = columns.index(ij) + 1
            time_widths[idx] = max(time_widths[idx], len(cell))

    def time_row(cells: list[str]) -> str:
        return "  ".join(cells[k].rjust(time_widths[k]) for k in range(len(cells)))

    print("\nBelos solve time (s):")
    print(time_row(headers))
    print(time_row(["-" * w for w in time_widths]))
    for s in solvers_for_time:
        cells = [s]
        for ij in columns:
            val = data_time.get(s, {}).get(ij, "n/a")
            cells.append(f"{float(val):.2f}" if isinstance(val, (int, float)) else str(val))
        print(time_row(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
