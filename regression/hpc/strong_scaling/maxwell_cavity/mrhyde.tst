#!/usr/bin/env python3
"""
HPC Strong Scaling Test: Maxwell 3D Cavity

Runs the same problem with 1, 2, and 4 MPI processes and verifies
that parallel efficiency is within acceptable bounds.

This test only runs when explicitly requested with -k hpc.
"""

import sys
import os

# Get the directory containing this test file
test_dir = os.path.dirname(os.path.abspath(__file__))
# hpc/scripts is two levels up from maxwell_cavity
hpc_scripts_dir = os.path.join(test_dir, "..", "..", "scripts")
hpc_scripts_abs = os.path.abspath(hpc_scripts_dir)
sys.path.insert(0, hpc_scripts_abs)  # Use insert(0) to prioritize this path

# Also add regression/scripts for mrhyde_test_support
regression_scripts_dir = os.path.join(test_dir, "..", "..", "..", "scripts")
regression_scripts_abs = os.path.abspath(regression_scripts_dir)
sys.path.insert(0, regression_scripts_abs)

from mrhyde_test_support import *

# Test metadata parsed by runtests.py
#TESTING active
#TESTING -n 4
#TESTING -k hpc,strong_scaling,maxwell

desc = '''HPC Strong Scaling Test: Maxwell 3D Cavity'''
its = mrhyde_test_support(desc)
its.opts.verbose = True

# Configuration
PROC_COUNTS = [1, 2, 4]
MIN_EFFICIENCY = 50  # percent

status = 0

print("=" * 70)
print("HPC Strong Scaling Test: Maxwell 3D Cavity")
print("=" * 70)

# Run MrHyDE with different processor counts
log_files = []
for nprocs in PROC_COUNTS:
    logfile = f"mrhyde_np{nprocs}.log"
    log_files.append(logfile)
    print(f"\nRunning with {nprocs} MPI process(es)...")
    run_status = its.call(f'mpiexec -n {nprocs} ../../../mrhyde >& {logfile}')
    if run_status != 0:
        print(f"  ERROR: MrHyDE failed with {nprocs} processes")
        status += 1

if status != 0:
    print("\nFailed to run MrHyDE, cannot perform scaling analysis")
    sys.exit(status)

# Perform scaling analysis
print("\n" + "=" * 70)
print("Scaling Analysis")
print("=" * 70)

try:
    print(f"Attempting to import scaling_analysis from: {hpc_scripts_abs}")
    from scaling_analysis import analyze_scaling
    print(f"Successfully imported scaling_analysis")
    print(f"Calling analyze_scaling with {len(log_files)} log files...")
    analysis_status = analyze_scaling(
        log_files,
        min_efficiency=MIN_EFFICIENCY,
        top_fraction=0.50,
        output_path="scaling_top50.txt",
    )
    print(f"Scaling analysis completed with status: {analysis_status}")
    status += analysis_status
except (ImportError, ModuleNotFoundError) as e:
    print(f"\nWarning: Could not import scaling_analysis: {e}")
    print(f"  Searched in: {hpc_scripts_abs}")
    print("Falling back to simplified analysis...")
    # Fallback: simple timing extraction without pandas
    import re

    def extract_total_time(logfile):
        """Extract total time from log file."""
        with open(logfile, 'r') as f:
            text = f.read()
        # Look for Total Time in TimeMonitor output
        match = re.search(r"Total Time[:\s]+([\d.]+)", text)
        if match:
            return float(match.group(1))
        # Fallback: look for MrHyDE::driver time
        match = re.search(r"MrHyDE::driver\s+([\d.]+)", text)
        if match:
            return float(match.group(1))
        return None

    times = {}
    for i, nprocs in enumerate(PROC_COUNTS):
        t = extract_total_time(log_files[i])
        if t:
            times[nprocs] = t

    if len(times) < 2:
        print("ERROR: Could not extract timing data from log files")
        status = 1
    else:
        baseline_procs = PROC_COUNTS[0]
        baseline_time = times[baseline_procs]
        print(f"\nTiming Summary:")
        print(f"  {baseline_procs} proc(s): {baseline_time:.2f}s (baseline)")

        for nprocs in PROC_COUNTS[1:]:
            if nprocs in times:
                t = times[nprocs]
                speedup = baseline_time / t
                scale = nprocs / baseline_procs
                efficiency = (speedup / scale) * 100
                print(f"  {nprocs} procs: {t:.2f}s "
                      f"(speedup {speedup:.2f}x, efficiency {efficiency:.0f}%)")
                if efficiency < MIN_EFFICIENCY:
                    print(f"    WARNING: Efficiency below threshold ({MIN_EFFICIENCY}%)")
                    status = 1
except Exception as e:
    print(f"\nError during scaling analysis: {e}")
    import traceback
    traceback.print_exc()
    status = 1

print("\n" + "=" * 70)
if status == 0:
    print("Success: Strong scaling test PASSED")
else:
    print("Failure: Strong scaling test FAILED")
print("=" * 70)

sys.exit(status)
