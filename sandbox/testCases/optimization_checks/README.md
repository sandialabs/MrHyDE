# Optimization checks

These sandboxes are pre-run checks for ROL optimization setups. Use them
to answer three questions before long runs:

1. Are objective and regularization weights on sane scales?
2. Is gradient and Hessian-vector plumbing correct on LQ problems?
3. What cost and robustness gap should we expect from exact vs FD HessVec?

## Test map

`optimization_checks/` has three active, committed test families:

- `scale_check/`: objective magnitude scan for weight tuning.
- `maxwell/hess_vec_check_lq/`: LQ correctness checks for gradient and HessVec.
- `maxwell/1d_periodic_exact_vs_fd_hess/`: runtime race, exact HessVec vs FD fallback.
- `thermal/hess_vec_check_lq/`: thermal LQ checks and mesh sweep with exact vs FD.
- `thermal/hess_vec_check_lq_tracking/`: same problem with a tracking target `Td != 0`; `trk_gate` keeps the exact path inside LQ.

## Activation flags

All options below live under `Analysis:ROL2:General:`.

```yaml
Do magnitude scan: true           # print per-term unweighted and weighted values
Do grad+hessvec check: true       # ROL gradient and HessVec FD diagnostics
Do exact hessvec check: true      # HessVec finite difference table
Do algebraic hessvec check: true  # H*0, bilinearity, Rayleigh checks
Do secant identity check: true    # compare Hv with grad(x+v)-grad(x)
FD Check Seed: 3                  # direction seed for FD checks
FD Check Random Seed: 42          # random probe point for diagnostics only
FD Check Random Scale: 1.0        # probe amplitude (default 1.0)
```

Use only the switches needed for the check you are running.

## Exact vs FD HessVec switch

The exact HessVec path requires `src_gate` in the exact deck. The tangent
sweep zeros this source gate, then restores it. If `src_gate` is missing,
`hessVec` falls back to ROL FD-of-gradients.

In short:

- `src_gate` present: exact HessVec path.
- `src_gate` absent: FD HessVec fallback path.

## What to expect

- `scale_check`: one table with per-term scales to tune weights.
- `hess_vec_check_lq` tests: small gradient and HessVec check residuals.
- exact-vs-fd sweeps: same final objective trend, but exact usually needs
  fewer inner Krylov solves and less wall time.

Each subdirectory README gives run commands and local expectations.
