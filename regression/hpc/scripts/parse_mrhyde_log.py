#!/usr/bin/env python3
"""
Parse MrHyDE log files to extract processor workload and timing tables.
"""

import re
import pandas as pd


def parse_workload(text):
    """Extract processor workload info into a DataFrame."""

    # patterns for each data type
    elem_pat = r"Processor (\d+) has (\d+) elements"
    belem_pat = r"Processor (\d+) has (\d+) boundary elements"
    vol_pat = r"Processor (\d+) is using ([\d.]+) MB to store volumetric data"
    face_pat = r"Processor (\d+) is using ([\d.]+) MB to store face data"
    bnd_pat = r"Processor (\d+) is using ([\d.]+) MB to store boundary data"

    # data per processor
    data = {}
    for match in re.finditer(elem_pat, text):
        proc = int(match.group(1))
        data.setdefault(proc, {})["elements"] = int(match.group(2))

    for match in re.finditer(belem_pat, text):
        proc = int(match.group(1))
        data.setdefault(proc, {})["boundary_elements"] = int(match.group(2))

    for match in re.finditer(vol_pat, text):
        proc = int(match.group(1))
        data.setdefault(proc, {})["volumetric_MB"] = float(match.group(2))

    for match in re.finditer(face_pat, text):
        proc = int(match.group(1))
        data.setdefault(proc, {})["face_MB"] = float(match.group(2))

    for match in re.finditer(bnd_pat, text):
        proc = int(match.group(1))
        data.setdefault(proc, {})["boundary_MB"] = float(match.group(2))

    if not data:
        return pd.DataFrame()

    rows = []
    for proc in sorted(data.keys()):
        row = {"processor": proc}
        row.update(data[proc])
        rows.append(row)

    return pd.DataFrame(rows)


def parse_timing(text):
    """Extract TimeMonitor table into a DataFrame."""

    # Find the TimeMonitor section
    section_pat = r"={50,}\n(.*?TimeMonitor.*?)\n={50,}"
    match = re.search(section_pat, text, re.DOTALL)

    if not match:
        return pd.DataFrame()

    section = match.group(0)
    lines = section.strip().split("\n")

    # format check: single-processor vs multi-processor
    is_single_proc = "Global time" in section

    data_lines = []
    for line in lines:
        if line.startswith("MrHyDE::") or line.startswith("STK_") or \
           line.startswith("panzer") or line.startswith("Solver") or \
           line.startswith("Time ") or line.startswith("Total Time"):
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    def parse_value_calls(s):
        s = s.strip()
        m = re.match(r"([\d.e+-]+)\s*\((\d+\.?\d*)\)", s)
        if m:
            return float(m.group(1)), float(m.group(2))
        return 0.0, 0

    rows = []
    for line in data_lines:
        parts = re.split(r"\s{2,}", line.strip())

        if is_single_proc:
            if len(parts) < 2:
                continue
            timer_name = parts[0]
            val, calls = parse_value_calls(parts[1])
            rows.append({
                "timer_name": timer_name,
                "GlobalTime": val,
                "GlobalTime_calls": int(calls),
            })
        else:
            if len(parts) < 5:
                continue
            timer_name = parts[0]
            min_val, min_calls = parse_value_calls(parts[1])
            mean_val, mean_calls = parse_value_calls(parts[2])
            max_val, max_calls = parse_value_calls(parts[3])
            mean_per_call, mean_per_call_n = parse_value_calls(parts[4])
            rows.append({
                "timer_name": timer_name,
                "MinOverProcs": min_val,
                "MinOverProcs_calls": int(min_calls),
                "MeanOverProcs": mean_val,
                "MeanOverProcs_calls": int(mean_calls),
                "MaxOverProcs": max_val,
                "MaxOverProcs_calls": int(max_calls),
                "MeanOverCallCounts": mean_per_call,
                "MeanOverCallCounts_calls": int(mean_per_call_n),
            })

    return pd.DataFrame(rows)


def parse_log(filepath):
    """
    Parse a MrHyDE log file.

    Returns:
        tuple: (workload_df, timing_df)
    """
    with open(filepath, "r") as f:
        text = f.read()

    workload_df = parse_workload(text)
    timing_df = parse_timing(text)

    return workload_df, timing_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_mrhyde_log.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    workload, timing = parse_log(logfile)

    print("=" * 60)
    print("PROCESSOR WORKLOAD")
    print("=" * 60)
    print(workload.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("TIMING DATA")
    print("=" * 60)
    print(timing.to_string(index=False))
