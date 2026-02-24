# MrHyDE Regression Testing

This document describes the regression testing framework for MrHyDE.

## Quick Start

```bash
# From MrHyDE build directory
cd MrHyDE/regression
ln -sf /path/to/mrhyde.exe mrhyde
python3 scripts/runtests.py
```

## Directory Structure

```
regression/
  scripts/
    runtests.py              # Main test runner
    mrhyde_test_support.py   # Test utilities for .tst scripts
  burgers/
    1D_bump/
      input.yaml             # Problem configuration
      mrhyde.tst             # Test script (executable)
      mrhyde.gold            # Expected output
  maxwell/
    PlaneWave/
      input.yaml, input_mesh.yaml, input_functions.yaml
      mrhyde.tst
      mrhyde.gold
    maxwell_hybrid/
      input.yaml, ...
      run.sh                 # Manual run script (no automated testing)
      ref/mrhyde.ocs         # Reference output for comparison
  hpc/                       # HPC tests (run with -k hpc)
    scripts/
      parse_mrhyde_log.py    # Log file parser
      scaling_analysis.py    # Scaling analysis utilities
    strong_scaling/
      maxwell_cavity/        # Strong scaling test for Maxwell
  ...
```

## Test Types

### Type A: Automated Tests (.tst + .gold)

These tests are discovered and run automatically by `runtests.py`.

**Required files:**
- `input.yaml` - Problem configuration (mesh, physics, solver parameters)
- `mrhyde.tst` - Python test script with `#TESTING` directives (uses `mrhyde_test_support.py` utilities)
- `mrhyde.gold` - Expected output for diff comparison

**How it works:**
1. `runtests.py` finds all `.tst` files recursively
2. Parses `#TESTING` directives to determine if test is active, keywords, etc.
3. Executes the `.tst` script
4. The script runs MrHyDE, cleans the log, and diffs against `.gold`
5. Exit status 0 = pass, non-zero = fail

**Example .tst file** (`burgers/1D_bump/mrhyde.tst`):
```python
#!/usr/bin/env python3
import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

desc = '''Burgers 1D bump test'''
its = mrhyde_test_support(desc)
its.opts.verbose = True

root = 'mrhyde'

# Test metadata parsed by runtests.py
#TESTING active
#TESTING -n 1
#TESTING -k Burgers,transient,1D

status = 0
status += its.call('mpiexec -n 1 ../../mrhyde >& mrhyde.log')
status += its.clean_log()
status += its.call('diff -y %s.log %s.gold' % (root, root))

if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
```

### Type B: Manual Tests (run.sh only)

These exist for development or manual verification. They are **not** run by `runtests.py`.

**Files:**
- `input*.yaml` - Problem configuration
- `run.sh` - Shell script to run MrHyDE
- `ref/` - Reference output for manual comparison

**Example** (`maxwell/maxwell_hybrid/run.sh`):
```bash
#!/bin/bash
mpiexec -n 1 ../../mrhyde >& mrhyde.log
```

## Test Script Directives (#TESTING)

Place these comments in `.tst` files to control test behavior:

| Directive | Description |
|-----------|-------------|
| `#TESTING active` | **Required** to run the test |
| `#TESTING -n N` | Number of MPI processes |
| `#TESTING -k key1,key2` | Keywords for filtering (comma = AND) |
| `#TESTING -K key1,key2` | Exclude keywords |
| `#TESTING -m machine1` | Run only on specific machines |
| `#TESTING -M machine1` | Exclude specific machines |
| `#TESTING -t hours` | Expected runtime in hours |

## Running Tests

### Basic Usage

```bash
cd MrHyDE/regression
ln -sf /path/to/mrhyde.exe mrhyde

# Run all active tests
python3 scripts/runtests.py

# List tests without running
python3 scripts/runtests.py -i

# Verbose output
python3 scripts/runtests.py -v
```

### Filtering Tests

```bash
# Run only Maxwell tests
python3 scripts/runtests.py -k Maxwells

# Run tests that are BOTH transient AND 3D
python3 scripts/runtests.py -k transient,3D

# Exclude long-running tests
python3 scripts/runtests.py -K long

# Run specific test directory
python3 scripts/runtests.py -d maxwell/PlaneWave
```

### Parallel Execution (-S flag)

The `-S CORES` flag runs multiple *tests* concurrently, not MPI processes. Each test still launches its own MPI job (as specified by `#TESTING -n`).

The scheduler tracks available cores and only launches a test when `test.nprocs <= available_cores`. Tests are sorted largest-first to optimize packing.

```bash
# If you have 8 cores and tests use 1-4 MPI procs each
python3 scripts/runtests.py -S 8
```
If all tests use 4 MPI procs, only 2 tests run concurrently (8/4=2).

Without `-S`, tests run sequentially (one at a time).

### Other Options

```bash
# Override processor count for all tests
python3 scripts/runtests.py -n 2

# Non-recursive (current directory only)
python3 scripts/runtests.py -N

# List all keywords used across tests
python3 scripts/runtests.py --list-keywords
```

## runtests.py Reference

### Working Options

| Flag | Description |
|------|-------------|
| `-d DIR` | Starting directory (default: current) |
| `-e EXT` | Test extension (default: `.tst`) |
| `-f` | Force run inactive tests |
| `-i` | Info mode: list tests, don't run |
| `-k KEYWORDS` | Include matching keywords |
| `-K KEYWORDS` | Exclude matching keywords |
| `-n NPROCS` | Override processor count |
| `--nrange MIN,MAX` | Filter by processor range |
| `-N` | Non-recursive search |
| `-S [CORES]` | Parallel execution |
| `-t TESTS` | Comma-separated test list |
| `-T FILE` | File with test list |
| `-v` | Verbose output |
| `-s` | Simple reporting |
| `--list-keywords` | List all keywords |
| `--print-keywords` | Show keywords in results |
| `--include-all` | Include inactive in report |

### Legacy Options (potentially broken)

These options were inherited from the DGM project and have hardcoded values from ~2015:

| Flag | Status |
|------|--------|
| `-b` | **Broken**: References non-existent scripts |
| `-q` | **Legacy**: Only works on old SNL clusters (redsky, glory, skybridge, curie) |
| `-c` | Passed to tests but doesn't affect runtests.py |
| `--32/--64` | Legacy bit mode |

## Running on SLURM Clusters

The built-in cluster support (`-b`, `-q`) is outdated. Use these interactive or batch scripts to run these tests on your local cluster.

## HPC Regression Testing

HPC regression tests verify parallel scaling behavior and are located in the `hpc/` directory.
These tests are not run by default. The test driver automatically excludes tests whose
keywords include `hpc` unless you explicitly request them with `-k hpc`.

### Running HPC Tests

```bash
cd MrHyDE/regression

# Run all HPC tests
python3 scripts/runtests.py -k hpc
```

### HPC Test Structure

```
regression/
  hpc/
    scripts/
      parse_mrhyde_log.py    # Log file parser
      scaling_analysis.py    # Scaling analysis utilities
    strong_scaling/
      maxwell_cavity/
        input.yaml           # Problem configuration
        mrhyde.tst           # Test script
```

### HPC Helper Scripts (standalone use)

The two helper scripts in `hpc/scripts/` can be used directly on any MrHyDE log files
that contain compatible TimeMonitor output (`verbose: 10`).

**1. Log parser: `parse_mrhyde_log.py`**

From the `regression/` directory:

```bash
# Parse a single log and print workload and timing tables
python3 hpc/scripts/parse_mrhyde_log.py \
  hpc/strong_scaling/maxwell_cavity/mrhyde_np1.log
```
This script extracts:
- Per-processor workload information (elements, memory usage)
- Timing data from the TimeMonitor section

**2. Scaling analysis: `scaling_analysis.py`**

From the `regression/` directory:

```bash
# Analyze scaling across multiple logs (for example 1, 2, 4 MPI processes)
python3 hpc/scripts/scaling_analysis.py \
  hpc/strong_scaling/maxwell_cavity/mrhyde_np1.log \
  hpc/strong_scaling/maxwell_cavity/mrhyde_np2.log \
  hpc/strong_scaling/maxwell_cavity/mrhyde_np4.log
This script:
- Loads timing data from each log (via `parse_mrhyde_log.py`)
- Computes speedup and parallel efficiency versus processor count

## Creating a New Test

1. Create directory: `regression/mymodule/mytest/`
2. Copy an existing test (e.g., `maxwell/PlaneWave/`) as template
3. Modify `input.yaml` and update `#TESTING` directives in `mrhyde.tst`
4. Run once, then `cp mrhyde.log mrhyde.gold` to create baseline
5. Verify: `./mrhyde.tst` should exit 0

As a guideline, design tests that run on the order of seconds to allow for frequent and fast regression testing.

## Output

Tests produce XML output (`TEST-package.xml`) compatible with Jenkins CI written to working directory from where `runtests.py` is run. 
The summary shows:

```
Test Results from Directory: /path/MrHyDE/regression
Total number of test(s): 139
-----------------------------------------------------------------------------------------------
  1/139       pass   14.53s  np=4                                                    maxwell/Planewave-LeapFrog1
  2/139       pass    1.17s  np=4                                                              maxwell/PlaneWave
  3/139       pass    1.05s  np=4                                                    maxwell/Planewave-LeapFrog2
  4/139       pass    1.79s  np=4                                                             phasefield/2d-3phi
  ...
  139/139     pass    1.38s  np=1                                                              levelSet/wildfire
-----------------------------------------------------------------------------------------------
 Pass: 139    Fail: 0    Skipped: 0    Total: 139

Total Runtime:     129.42s
```
