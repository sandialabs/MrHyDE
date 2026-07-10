Maxwell solver convergence tests
================================

Layout
------

- **fragments/**: YAML snippets (AMG, smoothers, etc.) included by preconditioner configs via `../../fragments/...`.
- **scripts/**: Python tools to run sweeps and post-process logs.
- **preconditioners/**: Test cases (input YAMLs and `mrhyde.tst` per case). Shared: `input_common_maxwell_hex3d.yaml`, `input_solver_common.yaml`.

Run all scripts either from the `maxwell` directory (for example `python scripts/run_mesh_sweep.py preconditioners`) or from the MrHyDE repo root using the full script path.

Collecting data
---------------

```bash
python scripts/run_mesh_sweep.py preconditioners [--resolve-imports] [--nsteps ...] [--ref ...]
```

- `--resolve-imports`: Resolve `input.short.yaml` (YAML `import` keys) into `input.yaml` before running; use when cases pull in `../../fragments/` and shared inputs.
- `--nsteps`, `--ref`: Optional comma-separated lists for time and spatial refinement; see script help for details.

This script updates mesh and solver YAMLs in `preconditioners/`, ensures `regression/mrhyde` points to the chosen executable, runs `regression/scripts/runtests.py -d <preconditioners path>`, and copies each test's `mrhyde.log` to `mrhyde_t{i}_r{j}.log` in that test's directory.

Post-processing
---------------

Summary tables and PDFs (iteration counts and Belos solve time vs mesh and time-step refinement):

```bash
python scripts/summarize_sweeps.py preconditioners \
  [--out iteration_count_summary.pdf] \
  [--out-time solve_time_summary.pdf] \
  [--exclude-time SUBSTRING]
```

- `--exclude-time SUBSTRING`: Omit any preconditioner whose name contains `SUBSTRING` from the timings plot and the solve-time table (for example `--exclude-time cheb_only`).

Per-log iteration statistics (mean and standard deviation per log file):

```bash
python scripts/linear_iter_stats.py --dir preconditioners
```
