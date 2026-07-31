# Magnitude scan

Pre-run probe that reports each objective and regularization term as
`unweighted`, `weight`, and `weighted = unweighted * weight`. Point: pick
per-term weights from the unweighted column and see the resulting balance
in the weighted column before launching ROL.

## Activate

Under `Analysis:ROL2:General:` in the deck:

```yaml
Do magnitude scan: true
FD Check Random Seed: 42        # optional; probes at a seeded random ctrl
FD Check Random Scale: 1.0e-8   # optional; amplitude of the redirect (default 1.0)
```

MrHyDE runs one forward solve, prints the table, then continues to any
other enabled diagnostics (e.g. `Do grad+hessvec check`) and to ROL per
its `Iteration Limit`. For scan-only, set `Iteration Limit: 0` under
`Status Test:` (ROL then returns without stepping) and leave the FD/HessVec
flags off.

## What you get

```
[MAGNITUDE-SCAN] probe at seeded random ctrl (seed=42, scale=1e-08).
  Per-term contributions at the probe point (ROL iter-0 may differ if
  the initial iterate is initialized differently):
  term                            type      unweighted        weight      weighted
  --------------------------------------------------------------------------------
  EM Energy                       obj        2.007e-39     1.000e+35     2.007e-04
  RegObj                          obj        0.000e+00     0.000e+00     0.000e+00
  RegObj/l2reg                    reg        1.110e-30     1.000e+05     1.110e-25
  RegObj/curlreg                  reg        1.221e-19     1.000e+05     1.221e-14
  --------------------------------------------------------------------------------
  TOTAL                                      1.221e-19                   2.007e-04
```

- `unweighted` is what each objective/regularizer contributes before its
  weight is applied. Independent of the `weight:` values in the deck --
  set them to anything and this column stays the same at a fixed probe
  point.
- `weight` is the deck's `weight:` value.
- `weighted` is `unweighted * weight`. Sum of this column is the objective
  ROL would see if evaluated at this probe point.

## Probe point vs ROL iter-0

The scan reports at the redirect point (a seeded random control at
amplitude `FD Check Random Scale`), then restores the parameter vector
before ROL runs. So on a deck with `ctrl_current` initial_value = 0,
ROL iter-0 sees ctrl=0 (zero fields, near-roundoff objective), not the
scan's TOTAL. Compare `TOTAL weighted` in the scan block with `value` at
iter 0 in the ROL history:

```
TOTAL weighted:  2.007e-04       # scan at seeded random ctrl, amplitude 1e-8
iter 0 value:    ~roundoff       # ROL at ctrl=0
```

If you want ROL to start near where the scan probed, set the deck's
`initial_value` in `other_decks/input_params.yaml` to a nonzero constant
(see `hess_vec_check_non_lq/other_decks_exact/` for `initial_value: 0.001`).
Only scalar `initial_value` is currently supported; a spatially-varying
initial iterate (e.g. `sin(x)*sin(y)`) would need to be loaded via
`Analysis: read parameters from file: true`.

## Sanity check: scale=0 matches ROL iter-0

`logs/scan_scale_0.log` reruns the same deck with
`FD Check Random Scale: 0.0` and `Iteration Limit: 1` (so ROL iter-0
prints). Scale=0 means x is redirected to zero, which equals the deck's
`initial_value = 0`, so the scan and ROL probe the same point:

```
scan TOTAL weighted: 1.538e-12
ROL iter 0 value:    1.537928e-12   # match to 4 sig figs (scan prints 4)
```

`logs/scan.log` uses `scale=1e-8` (nonzero probe) and `Iteration Limit: 0`
(no ROL iterations). The two logs together demonstrate: the scan's TOTAL
weighted is a faithful evaluation of the objective at whatever point x
is redirected to, and it converges to ROL iter-0 as scale -> 0.

## Tuning the probe amplitude

`FD Check Random Scale` controls the magnitude of the redirected ctrl.
Contributions scale as expected: `curlreg` and `l2reg` (quadratic in
ctrl) go as `scale^2`, `EM Energy` (quadratic in fields, which are
linear in ctrl-driven currents) goes as `scale^2` too. Set the scale to
match the amplitude you expect during optimization. Example: shrinking
from `1e-3` to `1e-8` (factor 1e-5) drops all three terms by 1e-10:
`EM Energy` unweighted goes from 2.006e-29 to 2.007e-39, `curlreg`
from 1.221e-09 to 1.221e-19.
