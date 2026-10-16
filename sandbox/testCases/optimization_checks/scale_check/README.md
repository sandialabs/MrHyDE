# Magnitude check

Prints `unweighted`, `weight`, and `weighted = unweighted * weight` for
every objective and regularization term, so you can pick weights that put
the terms in a useful balance.

## How to run

Set under `Analysis:ROL2:General:`:

```yaml
Do magnitude check: true
FD Check Random Seed: 42
FD Check Random Scale: 1.0e-8
```

For check-only runs, set `Iteration Limit: 0` in `Status Test:` and keep
other FD or HessVec checks off.

## Output

```text
[MAGNITUDE-CHECK] probe at seeded random ctrl (seed=42, scale=1e-08).
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

`./run_check.sh` writes `logs/check.log` (nonzero probe scale) and
`logs/check_scale_0.log` (zero probe scale).
