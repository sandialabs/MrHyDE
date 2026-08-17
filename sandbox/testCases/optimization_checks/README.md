# Optimization checks

Pre-run diagnostics that catch bad optimization setups before you burn a
long ROL run. Every sandbox here enables one YAML flag under
`Analysis:ROL2:General:`, runs a small probe, prints a verdict, and exits.
Each subdirectory has its own `README.md` with more detailed analysis and notes.

## Layout

```
optimization_checks/
  scale_check/        physics-agnostic weight scan
  maxwell/            Maxwell E-B control problems (LQ HessVec checks, TR-CG race)
  thermal/            thermal control problems (LQ HessVec check + mesh sweep)
```

## [scale_check](./scale_check/)

The magnitude-scan probe: reports each objective and regularization term
in their unweighted and weighted forms. Helps you choose weights that
balance the terms.

```yaml
Do magnitude scan: true
FD Check Random Seed: 42        # optional; probes at a seeded random ctrl
FD Check Random Scale: 1.0e-8   # optional; amplitude (default 1.0)
```

```txt
[MAGNITUDE-SCAN] probe at seeded random ctrl (seed=42, scale=1e-08).
  term                            type      unweighted        weight      weighted
  --------------------------------------------------------------------------------
  EM Energy                       obj        2.007e-39     1.000e+35     2.007e-04
  RegObj/l2reg                    reg        1.110e-30     1.000e+05     1.110e-25
  RegObj/curlreg                  reg        1.221e-19     1.000e+05     1.221e-14
  --------------------------------------------------------------------------------
  TOTAL                                      1.221e-19                   2.007e-04
```

See [scale_check/README.md](./scale_check/README.md).

## Maxwell (`./maxwell/`)

Rank counts below are hard constraints of the mesh partition
(`Xprocs * Yprocs * Zprocs`) — MPI ranks must match.

### [maxwell/hess_vec_check_lq](./maxwell/hess_vec_check_lq/) — 11 ranks

FD gradient check plus exact (or FD-of-gradients) HessVec on a
linear-quadratic Maxwell control problem, where
`grad(x+v) - grad(x) = H v` holds exactly. Use this to confirm the
gradient and `H v` path before a long ROL run.

Under `Analysis:ROL2:General:`:

```yaml
Do grad+hessvec check: true       # FD grad + checkHessVec + checkHessSym
Do exact hessvec check: true      # HessVec FD table + symmetry
Do algebraic hessvec check: true  # H*0, bilinearity, Rayleigh
Do secant identity check: true    # Hv vs (grad(x+v) - grad(x))
FD Check Seed: 3                  # FD direction
FD Check Random Seed: 42          # probe away from ctrl=0
FD Check Random Scale: 1.0
```

Exact HessVec also needs the physical source gated off on the tangent
sweep (`src_gate` in `other_decks_exact/`). Without that gate MrHyDE falls
back to ROL FD-of-gradients `hessVec`. Details:
[maxwell/hess_vec_check_lq/README.md](./maxwell/hess_vec_check_lq/README.md).

### [maxwell/1d_periodic_exact_vs_fd_hess](./maxwell/1d_periodic_exact_vs_fd_hess/) — 11 ranks

Same LQ problem, two `H v` paths in TR Newton-CG: exact (2 sweeps/call)
vs FD-of-gradients (4 sweeps/call, noise ~`eps/h`). Historical race
showed exact using ~2x fewer sweeps and ~1.6x less wall time; the deck
still uses `Initial Radius: 1.0` which TR-clamps the exact Newton step
(see the thermal case notes below), so raising it should tighten the
comparison further. Details:
[maxwell/1d_periodic_exact_vs_fd_hess/README.md](./maxwell/1d_periodic_exact_vs_fd_hess/README.md).

## Thermal (`./thermal/`)

### [thermal/hess_vec_check_lq](./thermal/hess_vec_check_lq/) — 4 ranks

Distributed-source thermal LQ optimal control on `[0,1]^3` with a
closed-form single-mode optimum. Runs the LQ operator checks, then a mesh
sweep (N = 4, 8, 16) that compares exact vs FD HessVec on outer
iterations, inner CG count, native L2 error, and wall time. Exact
converges in 2 outer TR iterations at every mesh; FD needs 3–6 and 5–9x
more CG at coarse meshes. Details:
[thermal/hess_vec_check_lq/README.md](./thermal/hess_vec_check_lq/README.md).
