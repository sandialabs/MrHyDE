# Predictions (written before any race run)

Deck: LQ Maxwell (`0.5*eps*|E|^2 + 0.5/mu*|B|^2`, zero target), seeded
`initial_value: 0.001`, `w_EM = 1e13`, `w_l2reg = w_curlreg = 1e5`,
`linear TOL = 1e-6`, NSTEPS = 50, mesh r1.

ROL: TR-CG, CG iter limit 100, CG rel tol 1e-10, outer gtol 1e-10
relative, outer limit 25.

## Why these values

Magnitude-scan math (from `scale_check` at seeded probe amplitude 1e-8):
`w_EM * (EM unweighted) = w_l2reg * (l2reg unweighted)` at
`w_EM ~ 5.5e13`. Rounded down to `1e13` so l2reg is slightly bigger,
avoiding a pure EM regime.

Conditioning estimate: previous deck at `w_EM/w_reg ~ 1e30` implied
`cond(H) ~ 1e30`, so `sqrt(cond)` iterations = infeasible. Dropping
that ratio by 22 orders puts `cond(H) ~ 1e8`, so `sqrt(cond) ~ 1e4`
theoretical CG iters, a few hundred in practice with eigenvalue
clustering. 100 CG iters should be borderline; if `flagCG = 1` on the
first outer, we bisect `w_EM` down to `1e11` and retry.

Noise floor: `linear TOL = 1e-6` gives gradient relative error
`~1e-6`. FD hessvec error `~ 2*sqrt(eta * L)` in absolute terms, so
FD's relative operator error floors at `~2*sqrt(1e-6) ~ 2e-3`. CG
can't drive its residual below this; FD should stagnate near
`~1e-3` relative gnorm reduction per outer.

## Predicted signatures

### exact
- CG hits `flagCG = 0` (residual tol met) or `flagCG = 3` (TR boundary)
  on most outers, not `flagCG = 1` (iter limit).
- Outer objective and gradient drop to roundoff (`~1e-30` and
  `~1e-14` respectively) in ~1-3 outer iters.
- `Actual / Predicted reduction ~ 1` throughout (LQ signature).

### fd
- CG hits `flagCG = 1` on every outer once the CG residual reaches
  the FD operator noise floor (~1e-3 relative).
- Outer gradient decays for the first 1-2 outers, then plateaus at
  `~1e-3 * ||g_0||`.
- Outer iter limit exceeded without meeting gtol.
- `Actual / Predicted` ratio starts near 1, degrades as FD noise
  becomes comparable to the true reduction, may trigger TR rejections.

## Cost fairness

Per CG iteration: exact does one tangent + one incremental adjoint
solve; FD does two full-gradient evaluations = 2 (forward + adjoint).
So the per-CG-iter cost is comparable. The win should show up as
"FD cannot reach the tolerance at any cost," not "exact runs faster
per iter."

Track: outer iters, total CG iters, total linear solves.

## Failure modes to watch for

- **CG-limit-bound on exact.** If exact also flags CG=1 every outer,
  conditioning is still too bad; drop `w_EM` further.
- **TR stall on both.** If TR shrinks radius aggressively, the
  quadratic model has broken down; check if `linear TOL: 1e-6` is
  producing negative-curvature CG iterates.
- **FD converges too.** If FD also reaches `1e-14`, the deck is too
  well-conditioned relative to the noise; either tighten CG rel tol
  or bump `linear TOL` to `1e-4`.
