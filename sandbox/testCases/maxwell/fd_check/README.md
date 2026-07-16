# FD gradient check: User debug hints

The change in `runFDGradientCheck` in
`code/mrhyde/MrHyDE/src/managers/analysis/analysisManager_solve.hpp`
wraps `checkGradient`, detects the common gradient-check pathologies,
and advises the user on how to resolve them.

## The two traps

$$
\text{FD}(h) \;=\; \frac{J(x + h d) - J(x)}{h}
\;=\; g \cdot d \;+\; \tfrac{h}{2}\, d^\top H d \;+\; O(h^2)
\;\pm\; \tfrac{2\varepsilon_J}{h},
$$

where `eps_J` is the noise in one objective evaluation (roundoff for
analytic terms, linear-solver noise for state-dependent terms).

Commong observed issues: 
1. If the start point is (near) a critical
   point along `d`, then `g . d` sits at the noise floor while the
   curvature and noise terms do not, so `rel_err ~ 1` at every `h`.
   Most Maxwell control decks start from `ctrl = 0`, which is such a
   point.
   Fix: redirect the check to a seeded random point.
2. For `J = w_EM J_EM(z, u(z)) + w_reg J_reg(z)`,
   FD probes whichever term dominates `J`. A dominant reg term is
   analytic and state-free so it says nothing about the state-adjoint path.
   Fix: zero every reg weight to isolate the EM path.

## The change

Activate the redirect under `Analysis:ROL2:General:`:

```yaml
Do grad+hessvec check: true
# z <- scale * uniform(-1, 1) only for the FD check; initial iterate is untouched.
FD Check Random Seed: 42     # any int
FD Check Random Scale: 1.0   # optional (default is 1.0)
```

Emitted when the FD/g.d ratio at h=1 exceeds 1e6:

```
[FDCHECK-HINT] FD approx at h=1 differs from grad'*dir by ratio <r>.
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

Emitted when redirection is used 
```
[FDCHECK-REDIRECT] FD check redirected to random parameter vector
(seed=42, scale=1.0). Optimizer initial iterate is unchanged.
```

## Examples

Redirect `FD Check Random Scale = 1.0`.
Logs in `sandbox/testCases/maxwell/fd_check/logs/`.

| Mode          | `w_EM`  | `w_reg` | Redirect | `grad'*dir` | best `abs_err @ h`  | `rel_err`   |
|---------------|---------|---------|----------|-------------|---------------------|-------------|
| noscale       | `1e35`  | `1e5`   | off      | `+6.47e-01` | `7.02e-01 @ 1e-12`  | `1.1e+00`   |
| seed          | `1e35`  | `1e5`   | on       | `-3.47e+11` | `3.76e+08 @ 1e-04`  | `1.1e-03`   |
| seed_regs0    | `1e35`  | `0`     | on       | `-6.27e+11` | `1.64e+08 @ 1e-04`  | `2.6e-04`   |
| seed_w1       | `1`     | `1e5`   | on       | `+1.08e+00` | `1.35e-06 @ 1e-08`  | `1.3e-06`   |
| seed_w1_regs1 | `1`     | `1`     | on       | `-1.64e-05` | `3.44e-12 @ 1e-08`  | `2.1e-07`   |
| seed_w1_regs0 | `1`     | `0`     | on       | `+2.67e-24` | `1.24e-26 @ 1e-03`  | `4.7e-03`   |

- `noscale`: FD sits at ctrl=0, trap 1 fires.
- `seed_w1`, `seed_w1_regs1`: the reg term (~1e19 larger per unit
  weight on this deck) dominates. Best `h ~ 1e-8 ~ sqrt(eps_mach)` 
  as expected for an analytic path.
- `seed_regs0`, `seed_w1_regs0`: the EM path is isolated; `rel_err`
  bottoms out at ~1e-3 with best `h` near `1e-4`, the noise floor of the
  state-dependent path.