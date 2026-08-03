# FD gradient and Hessian-vector checks (LQ)

Sandbox check of the FD gradient and exact HessVec `H v` for a
quadratic-in-state Maxwell control problem
(`0.5*eps*|E|^2 + 0.5/mu*|B|^2`). The reduced problem is
linear-quadratic, so `H(x)` is constant and
`grad(x+v) - grad(x) = H v` holds exactly.

## Run

```bash
./run_exact_sweep.sh   # 9 configs with    src_gate (exact tangent+adjoint HessVec)
./run_fd_sweep.sh      # 9 configs without src_gate (ROL FD-of-gradients HessVec)
python plot_hess_vec_check.py
```

Sample check output in `logs/`:
- broken FD grad (iterate, non-seeded): `logs/mrhyde_hv_r1_baseline_fd.log`
- healthy seeded exact: `logs/mrhyde_hv_r1_seed_exact.log`

Both ROL decks under `rol_decks/` set:

```yaml
Do grad+hessvec check: true       # checkGradient + checkHessVec + checkHessSym
Do exact hessvec check: true
Do algebraic hessvec check: true
Do secant identity check: true    # Hv vs (grad(x+v) - grad(x))
FD Check Seed: 3                  # random direction dir for (J(x+h*dir)-J(x))/h
```

Modes named `seed*` use `rol_seed_hess.yaml`, which also sets
(for FD check only; optimizer iterate unchanged):

```yaml
FD Check Random Seed: 42
FD Check Random Scale: 1.0
```

The other modes use `rol_noscale_hess.yaml` (probe at optimizer iterate).

### Exact vs FD HessVec

Exact HessVec needs a tangent sweep with the physical source off.
`other_decks_exact/` wraps the current with an inactive scalar:

```yaml
current x: 'src_gate*(gt*exp(-2*timebub*(t<toff))*(z<zmax)*(z>zmin))'
```

`incrementalForwardModel` sets `src_gate` to 0 for the tangent, then back to 1.
`other_decks_fd/` omits the wrapper, so `Objective_MILO::hessVec` falls back
to `ROL::Objective::hessVec` (FD-of-gradients).

## Failure modes you will hit

**Degenerate iterate.** Modes without `seed` start at `ctrl_current = 0`,
so `grad'*dir` is at the noise floor and FD grad fails (e.g, 
`baseline` - `best rel_err ~ 2e1`). The log prints:

```txt
[FDCHECK-HINT] best rel_err > 1.
  Likely causes and fixes:
    1) Degenerate start point: g.d is at the noise floor, so FD sees
       curvature and noise, not the gradient.
       Fix: 'FD Check Random Seed: 42' under Analysis:ROL2:General:
       (does not change the optimizer's initial iterate).
    2) A dominant regularizer weight makes the check validate the reg
       path, not the state-adjoint objective path.
       Fix: zero every Regularization functions weight. If the
       remaining signal is below FD noise, raise 'FD Check Random
       Scale' (default 1.0).
```

Set `ctrl_current > 0` or `FD Check Random Seed` on your own deck for a real FD-grad check.

## Healthy criteria

Judge relative error, not absolute error (large `w_EM` inflates abs values).

| Check | Healthy relative residual |
|-------|---------------------------|
| FD gradient (`[GRAD-CHECK] best rel_err`) | ~`1e-4` or better with a random probe; `> 1` prints `[FDCHECK-HINT]` |
| Exact secant (`[SECANT-IDENTITY] relative`) | ~`1e-7` to `1e-8` on the state path |
| FD-of-gradients secant | ~`1e-5` on the state path (looser; still OK) |
| Bilinearity (`[HV-BILINEARITY] relative`) | ~`1e-7` (exact) / ~`1e-5` (FD) on the state path |
| `H*0` | exactly 0 |
| Rayleigh `<v, Hv>` | positive (LQ is convex) |

If curlreg dominates, HessVec residuals can look machine-eps even when the
state-adjoint path is unchecked. Zero or shrink reg weights when probing
the EM path.

## Results

### FD gradient

Path-independent: same numbers in exact and fd logs.
`w_EM` = EM Energy weight; `w_curlreg` = curlreg weight (`l2reg` is 0).

| Param config  | w_EM  | w_curlreg | Probe    | grad'*dir  | best abs_err @ h    | rel_err   |
|---------------|-------|----------:|----------|------------|---------------------|-----------|
| baseline      | 1e35  | 1e5       | iterate  | +3.10e-02  | 6.40e-01 @ 1e-12    | 2.1e+01   |
| regs0         | 1e35  | 0         | iterate  | +3.10e-02  | 6.40e-01 @ 1e-12    | 2.1e+01   |
| w1            | 1     | 1e5       | iterate  | +3.10e-37  | 3.09e-11 @ 1e-12    | 1.0e+26   |
| w1_regs1      | 1     | 1         | iterate  | +3.10e-37  | 3.09e-16 @ 1e-12    | 1.0e+21   |
| seed          | 1e35  | 1e5       | random   | +6.69e+11  | 3.04e+08 @ 1e-04    | 4.5e-04   |
| seed_regs0    | 1e35  | 0         | random   | +6.69e+11  | 3.04e+08 @ 1e-04    | 4.5e-04   |
| seed_w1       | 1     | 1e5       | random   | +2.69e-01  | 3.95e-07 @ 1e-08    | 1.5e-06   |
| seed_w1_regs1 | 1     | 1         | random   | +2.69e-06  | 7.07e-12 @ 1e-08    | 2.6e-06   |
| seed_w1_regs0 | 1     | 0         | random   | +6.69e-24  | 3.04e-27 @ 1e-04    | 4.5e-04   |

Iterate rows are the degenerate case above. Random-probe rows are the
healthy FD-grad check (`seed` / `seed_regs0`: rel_err `~4.5e-4` at `h=1e-4`).

![FD grad check paired](logs/grad_check_abs_error_paired.png)

### HessVec: exact vs FD

Both paths recover the same operator; differences are noise floor and cost.

- `|secant|/ref` -- `|Hv - (grad(x+v)-grad(x))| / |grad(x+v)-grad(x)|`.
  On LQ this is a true operator check. Exact should sit at solver noise;
  a non-noise nonzero is a bug.
- `|hv0|` -- `||H*0||`. Must be 0 (helpful sanity check).
- `|bilin|/ref` -- relative linearity residual.
- `<v,Hv>_min` -- always positive here (LQ is convex).

![HessVec check paired](logs/hess_vec_check_abs_error_paired.png)
