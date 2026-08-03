# Optimization checks

Pre-run diagnostics that catch bad optimization setups before you burn a
long ROL run. Every sandbox here enables one YAML flag under
`Analysis:ROL2:General:`, runs a small probe, prints a verdict, and exits.
Each subdirectory has its own `README.md` with more detailed analysis and notes.

The bigpicture summary - 

## [scale_check](./scale_check/)

The magnitude-scan probe: reports each objective and regularization term
in their unweighted and weighted forms. This is to help 
choose weights that balance each terms's contribution.

```yaml
Do magnitude scan: true
FD Check Random Seed: 42        # optional; probes at a seeded random ctrl
FD Check Random Scale: 1.0e-8   # optional; amplitude (default 1.0)
```

```txt
[MAGNITUDE-SCAN] probe at seeded random ctrl (seed=42, scale=1e-08).
  Per-term contributions at the probe point (ROL iter-0 may differ if
  the initial iterate is initialized differently):
  term                            type      unweighted        weight      weighted
  --------------------------------------------------------------------------------
  EM Energy                       obj        2.007e-39     1.000e+35     2.007e-04
  RegObj/l2reg                    reg        1.110e-30     1.000e+05     1.110e-25
  RegObj/curlreg                  reg        1.221e-19     1.000e+05     1.221e-14
  --------------------------------------------------------------------------------
  TOTAL                                      1.221e-19                   2.007e-04
```

See [scale_check/README.md](./scale_check/README.md) for more info.

## [hess_vec_check_lq](./hess_vec_check_lq/)

FD gradient check plus exact (or FD-of-gradients) HessVec on a
linear-quadratic Maxwell control problem, where
`grad(x+v) - grad(x) = H v` holds exactly. Use this to confirm the
gradient and `H v` path before a long ROL run.

### Activate

Under `Analysis:ROL2:General:`:

```yaml
Do grad+hessvec check: true       # FD grad + checkHessVec + checkHessSym
Do exact hessvec check: true      # HessVec FD table + symmetry
Do algebraic hessvec check: true  # H*0, bilinearity, Rayleigh
Do secant identity check: true    # Hv vs (grad(x+v) - grad(x)); strongest on LQ
FD Check Seed: 3                  # FD direction
FD Check Random Seed: 42          # recommended: probe away from ctrl=0
FD Check Random Scale: 1.0
```

Exact HessVec also needs the physical source gated off on the tangent
sweep (`src_gate` in `hess_vec_check_lq/other_decks_exact/`). Without that
gate, MrHyDE falls back to ROL FD-of-gradients `hessVec`.

Healthy relative-error targets, failure modes, results tables, and sample
logs: [hess_vec_check_lq/README.md](./hess_vec_check_lq/README.md).

## [1d_periodic_exact_vs_fd_hess](./1d_periodic_exact_vs_fd_hess/)

Same LQ problem, two `H v` paths in TR Newton-CG: exact (2 sweeps/call)
vs FD-of-gradients (4 sweeps/call, noise ~`eps/h`). On this race exact
used ~2x fewer sweeps and ~1.6x less wall time; final `value`/`gnorm`
were comparable. Details:
[1d_periodic_exact_vs_fd_hess/README.md](./1d_periodic_exact_vs_fd_hess/README.md).
