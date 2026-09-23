# Optimization checks

Sandbox tests for the ROL gradient and Hessian-vector plumbing. Run one
before committing to a long optimization.

| Test | What it does |
| --- | --- |
| `scale_check/` | Prints each objective and regularization term's magnitude, so you can pick weights. |
| `maxwell/hess_vec_check_lq/` | Checks gradient and HessVec correctness on an LQ Maxwell control problem. |
| `maxwell/1d_periodic_exact_vs_fd_hess/` | Times exact HessVec against the FD fallback on the same problem. |
| `thermal/hess_vec_check_lq/` | Same correctness checks on thermal, plus a mesh sweep. |
| `thermal/hess_vec_check_lq_tracking/` | Same again, with a nonzero tracking target wrapped in `trk_gate`. |

## Running

Each test has its own driver (`run.sh`, `run_scan.sh`, `run_exact_sweep.sh`),
and each looks for the solver as `./mrhyde` in its own directory:

```bash
cd scale_check
ln -sf <path-to-build>/src/mrhyde mrhyde && ./run_scan.sh
MRHYDE_BIN=<path-to-build>/src/mrhyde ./run_scan.sh   # or skip the link
```

## Flags

All live under `Analysis:ROL2:General:`. Turn on only what you need.

```yaml
Do magnitude scan: true           # per-term unweighted and weighted values
Do grad+hessvec check: true       # ROL gradient and HessVec FD diagnostics
Do exact hessvec check: true      # HessVec finite difference table
Do algebraic hessvec check: true  # H*0, bilinearity, Rayleigh
Do secant identity check: true    # Hv vs grad(x+v)-grad(x)
FD Check Seed: 3                  # FD direction seed
FD Check Random Seed: 42          # random probe point, diagnostics only
FD Check Random Scale: 1.0        # probe amplitude
```

## Exact vs FD HessVec

`hessVec` takes the exact path (tangent plus second-order adjoint) when the
deck wraps the physical source in an inactive `src_gate` scalar. The tangent
sweep zeros that gate, then restores it. Without `src_gate`, ROL falls back to
finite differences of gradients.
