# Magnitude scan

Use this check to choose objective weights before a long ROL run.

It prints each objective and regularization term as:
- `unweighted`
- `weight`
- `weighted = unweighted * weight`

Start from the `unweighted` column, then choose `weight` values so the
`weighted` terms are in a reasonable balance.

## Enable

Set these under `Analysis:ROL2:General:`:

```yaml
Do magnitude scan: true
FD Check Random Seed: 42        # optional
FD Check Random Scale: 1.0e-8   # optional (default is 1.0)
```

For scan-only runs, set `Iteration Limit: 0` under `Status Test:` and
leave other FD/HessVec checks off.

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

## Important note about iter-0

The scan evaluates at a redirected control set by
`FD Check Random Scale`, then restores the control before ROL starts.

So `TOTAL weighted` from the scan can differ from ROL iter-0 `value`
when the deck starts from `ctrl_current = 0`.

## Logs in this folder

- Regenerate both logs with `./run_scan.sh`.
- `logs/scan.log`: nonzero probe (`scale=1e-8`)
- `logs/scan_scale_0.log`: zero probe (`scale=0`)

Use these logs if you want to verify the probe-point behavior in detail.
