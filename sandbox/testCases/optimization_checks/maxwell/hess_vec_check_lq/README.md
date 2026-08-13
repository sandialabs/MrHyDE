# FD gradient and Hessian-vector checks (LQ)

This sandbox checks FD gradients and Hessian-vector products `H v` for a
quadratic Maxwell control objective (`0.5*eps*|E|^2 + 0.5/mu*|B|^2`). 

## Run

```bash
./run_exact_sweep.sh          # exact tangent+adjoint HessVec path
./run_fd_sweep.sh             # ROL FD-of-gradients HessVec path
python plot_hess_vec_check.py # optional: text summaries and PNGs
```

### Configs

- `baseline`: FD probe stays at the optimizer iterate (`ctrl = 0`), which is
  intentionally degenerate for gradient checking.
- `seed`: adds `FD Check Random Seed: 42` so the FD probe is random/nonzero
  while keeping the optimizer iterate unchanged.

Both decks in `rol_decks/` enable:

```yaml
Do grad+hessvec check: true       # checkGradient + checkHessVec + checkHessSym
Do exact hessvec check: true
Do algebraic hessvec check: true
Do secant identity check: true    # Hv vs (grad(x+v) - grad(x))
FD Check Seed: 3                  # FD direction seed
```

## Exact vs FD HessVec paths

Exact HessVec uses a tangent solve with source disabled. In
`other_decks_exact/`, current is gated as:

```yaml
current x: 'src_gate*(gt*exp(-2*timebub*(t<toff))*(z<zmax)*(z>zmin))'
```

`incrementalForwardModel` sets `src_gate=0` for tangent, then restores `1`.
In `other_decks_fd/`, there is no gate, so `Objective_MILO::hessVec` falls
back to `ROL::Objective::hessVec` (FD-of-gradients).

## Failure mode this sandbox catches

**Degenerate iterate.** With `ctrl_current = 0`, `grad'*dir` is at the
noise floor and FD gradient reports `best rel_err ~ 20`. The log prints:

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

For a meaningful FD gradient check on your own deck, use `ctrl_current > 0` or
set an FD random seed.

## Results

### FD gradient

Path-independent (same values in exact/fd logs):

| Config   | Probe   | `grad'*dir` | best abs_err @ h | rel_err | Take-away                    |
| :------- | :------ | :---------- | :--------------- | :------ | :--------------------------- |
| baseline | iterate | +3.10e-02   | 6.40e-01 @ 1e-12 | 2.1e+01 | Degenerate `g.d` near noise  |
| seed     | random  | +6.69e+11   | 3.04e+08 @ 1e-04 | 4.5e-04 | Healthy random probe         |

### HessVec: exact vs FD

`|secant|/ref` checks
`|Hv - (grad(x+v)-grad(x))| / |grad(x+v)-grad(x)|`.
`|hv0| = ||H*0||` should be 0.
`|bilin|/ref` is linearity residual.
`<v,Hv>_min` stays positive for this convex LQ problem.

| Config   | path  | `|secant|/ref` | `|bilin|/ref` | `|hv0|` | `<v,Hv>_min` |
| :------- | :---- | -------------: | ------------: | ------: | -----------: |
| baseline | exact |       3.83e-08 |      7.24e-08 |       0 |     9.42e+11 |
| baseline | fd    |       1.35e-05 |      6.89e-05 |       0 |     9.42e+11 |
| seed     | exact |       3.83e-08 |      1.16e-07 |       0 |     8.09e+11 |
| seed     | fd    |       1.35e-05 |      8.56e-05 |       0 |     8.09e+11 |

