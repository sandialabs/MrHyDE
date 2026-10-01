#!/usr/bin/env python3
"""
Strong scaling analysis for MrHyDE HPC regression testing.

Analyzes how solver routines scale with increasing MPI tasks.
Provides pass/fail assessment based on qualitative efficiency thresholds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from parse_mrhyde_log import parse_log


def get_num_procs(workload_df):
    """Infer number of processors from workload DataFrame."""
    if workload_df.empty:
        return 1
    return workload_df["processor"].nunique()


def get_time_column(timing_df):
    """Determine which time column to use based on available columns."""
    if "GlobalTime" in timing_df.columns:
        return "GlobalTime"
    elif "MeanOverProcs" in timing_df.columns:
        return "MeanOverProcs"
    else:
        raise ValueError("No recognized time column in timing data")


def load_all_logs(log_files):
    """
    Load timing data from all log files.

    Returns:
        dict: {num_procs: timing_df} mapping processor count to timing data
    """
    results = {}
    for filepath in log_files:
        workload_df, timing_df = parse_log(filepath)
        num_procs = get_num_procs(workload_df)

        if timing_df.empty:
            print(f"Warning: No timing data in {filepath}, skipping")
            continue

        time_col = get_time_column(timing_df)
        timing_df = timing_df[["timer_name", time_col]].copy()
        timing_df.columns = ["timer_name", "time"]

        timing_df = timing_df[timing_df["time"] > 0]

        if num_procs in results:
            print(f"Warning: Duplicate processor count {num_procs}, "
                  f"overwriting with {filepath}")
        results[num_procs] = timing_df

    return results


def get_total_time(timing_df):
    """Get total simulation time from timing DataFrame."""
    total_row = timing_df[timing_df["timer_name"].str.contains("Total Time|MrHyDE::driver", case=False)]
    if not total_row.empty:
        return total_row["time"].values[0]
    # Fallback: sum of top-level timers
    return timing_df["time"].max()


def get_all_timer_names(all_data):
    """Union of timer names across all logs."""
    names = set()
    for timing_df in all_data.values():
        names.update(timing_df["timer_name"].tolist())
    return sorted(names)


def get_top_routines(timing_table, fraction=0.25, baseline_col=None):
    """
    Return routine names that account for the top fraction of time at baseline.

    Args:
        timing_table: DataFrame index=timer_name, columns=proc counts
        fraction: take top 25% by count (0.25)
        baseline_col: column for ranking (default: first column)

    Returns:
        list of timer_name (top fraction by time at baseline)
    """
    if baseline_col is None:
        baseline_col = timing_table.columns[0]
    series = timing_table[baseline_col].dropna().sort_values(ascending=False)
    n = max(1, int(np.ceil(fraction * len(series))))
    return series.head(n).index.tolist()


def build_timing_table(all_data, timer_names=None):
    """
    Build a table with routines as rows and processor counts as columns.

    Args:
        all_data: dict mapping num_procs to timing DataFrame
        timer_names: list of timer names to include (None = use total time only)

    Returns:
        pd.DataFrame with timer_name index and proc counts as columns
    """
    proc_counts = sorted(all_data.keys())

    if timer_names is None:
        # Just use total time
        timer_names = ["Total"]
        rows = [{"timer_name": "Total"}]
        for nprocs in proc_counts:
            rows[0][nprocs] = get_total_time(all_data[nprocs])
    else:
        rows = []
        for timer in timer_names:
            row = {"timer_name": timer}
            for nprocs in proc_counts:
                df = all_data[nprocs]
                match = df[df["timer_name"] == timer]
                if not match.empty:
                    row[nprocs] = match["time"].values[0]
                else:
                    row[nprocs] = np.nan
            rows.append(row)

    table = pd.DataFrame(rows).set_index("timer_name")
    return table


def write_scaling_table(timing_table, routine_names, savepath):
    """
    Write time vs nprocs scaling data to a text file.

    Args:
        timing_table: DataFrame index=timer_name, columns=proc counts
        routine_names: list of timer_name to include
        savepath: path to save text file
    """
    routine_names = [r for r in routine_names if r in timing_table.index]
    if not routine_names:
        return

    proc_counts = timing_table.columns.tolist()
    baseline_col = proc_counts[0]
    baseline_procs = baseline_col

    with open(savepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Top Routines Scaling Analysis\n")
        f.write("=" * 80 + "\n\n")

        for name in routine_names:
            row = timing_table.loc[name]
            times = [row[c] if not np.isnan(row[c]) else None for c in proc_counts]
            baseline_time = times[0]

            if baseline_time is None or baseline_time <= 0:
                continue

            f.write(f"{name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'MPI procs':<12} {'Time (s)':<15} {'Speedup':<15} {'Efficiency (%)':<15}\n")
            f.write("-" * 80 + "\n")

            for i, nprocs in enumerate(proc_counts):
                time_val = times[i]
                if time_val is None or time_val <= 0:
                    f.write(f"{nprocs:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}\n")
                else:
                    speedup = baseline_time / time_val
                    scale = nprocs / baseline_procs
                    efficiency = (speedup / scale) * 100
                    f.write(f"{nprocs:<12} {time_val:<15.4f} {speedup:<15.4f} {efficiency:<15.2f}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")


def compute_speedup(timing_table):
    """Compute speedup relative to smallest processor count."""
    baseline_col = timing_table.columns[0]
    speedup = timing_table.copy()
    for col in timing_table.columns:
        speedup[col] = timing_table[baseline_col] / timing_table[col]
    return speedup


def compute_efficiency(timing_table):
    """Compute parallel efficiency: T1 / (N * TN) * 100."""
    baseline_col = timing_table.columns[0]
    baseline_procs = baseline_col
    efficiency = timing_table.copy()
    for col in timing_table.columns:
        nprocs = col
        scale = nprocs / baseline_procs
        efficiency[col] = (timing_table[baseline_col] /
                           (scale * timing_table[col])) * 100
    return efficiency


def check_scaling_quality(timing_table, min_efficiency=50, verbose=True):
    """
    Check if scaling is acceptable.

    Args:
        timing_table: DataFrame with timing data
        min_efficiency: minimum acceptable parallel efficiency (%)
        verbose: print detailed output

    Returns:
        (passed, message) tuple
    """
    speedup_table = compute_speedup(timing_table)
    efficiency_table = compute_efficiency(timing_table)

    proc_counts = list(timing_table.columns)
    baseline_procs = proc_counts[0]

    if verbose:
        print("\nStrong Scaling Analysis")
        print("=" * 60)
        print(f"Processor counts: {proc_counts}")
        print("\nTiming Summary:")

    all_pass = True
    messages = []

    for timer in timing_table.index:
        if verbose:
            print(f"\n  {timer}:")

        for i, nprocs in enumerate(proc_counts):
            time_val = timing_table.loc[timer, nprocs]
            speedup = speedup_table.loc[timer, nprocs]
            eff = efficiency_table.loc[timer, nprocs]

            if np.isnan(time_val):
                if verbose:
                    print(f"    {nprocs} procs: N/A")
                continue

            if i == 0:
                if verbose:
                    print(f"    {nprocs} proc(s): {time_val:.2f}s (baseline)")
            else:
                if verbose:
                    print(f"    {nprocs} procs: {time_val:.2f}s "
                          f"(speedup {speedup:.2f}x, efficiency {eff:.0f}%)")

                # Check efficiency threshold (skip baseline)
                if eff < min_efficiency:
                    all_pass = False
                    messages.append(
                        f"{timer} at {nprocs} procs: efficiency {eff:.0f}% < {min_efficiency}%"
                    )

    if verbose:
        print("\n" + "=" * 60)
        if all_pass:
            print("Assessment: PASS")
            print("  Scaling is within acceptable range.")
        else:
            print("Assessment: FAIL")
            for msg in messages:
                print(f"  - {msg}")

    return all_pass, messages


def analyze_scaling(log_files, min_efficiency=50, verbose=True,
                    top_fraction=None, output_path=None):
    """
    Main entry point for scaling analysis.

    Args:
        log_files: list of log file paths
        min_efficiency: minimum acceptable parallel efficiency (%)
        verbose: print detailed output
        top_fraction: if set (e.g. 0.25), write side-by-side scaling table for
            top fraction of most consuming routines
        output_path: path to save the top-routines text file (used if
            top_fraction is set)

    Returns:
        0 if passed, 1 if failed
    """
    if verbose:
        print(f"Loading {len(log_files)} log files...")

    all_data = load_all_logs(log_files)

    if len(all_data) < 2:
        print("Error: Need at least 2 valid log files for scaling analysis")
        return 1

    if verbose:
        print(f"Found processor counts: {sorted(all_data.keys())}")

    # Pass/fail uses total time only
    timing_table_total = build_timing_table(all_data)
    passed, messages = check_scaling_quality(timing_table_total, min_efficiency, verbose)

    # Full timing DataFrame and optional top-fraction output
    if top_fraction is not None and output_path and 0 < top_fraction <= 1:
        all_timers = get_all_timer_names(all_data)
        full_timing_table = build_timing_table(all_data, timer_names=all_timers)
        top_routines = get_top_routines(full_timing_table, fraction=top_fraction)
        if top_routines and verbose:
            print(f"\nTop {int(100 * top_fraction)}% routines by time: {len(top_routines)}")
        write_scaling_table(full_timing_table, top_routines, output_path)
        if verbose and output_path:
            print(f"Scaling table saved to {output_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strong scaling analysis for MrHyDE HPC regression testing"
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        help="Log files to analyze (one per processor count)"
    )
    parser.add_argument(
        "--min-efficiency",
        type=float,
        default=50,
        help="Minimum acceptable parallel efficiency in %% (default: 50)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (only print pass/fail)"
    )

    args = parser.parse_args()

    for f in args.logfiles:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    status = analyze_scaling(
        args.logfiles,
        min_efficiency=args.min_efficiency,
        verbose=not args.quiet
    )
    sys.exit(status)
