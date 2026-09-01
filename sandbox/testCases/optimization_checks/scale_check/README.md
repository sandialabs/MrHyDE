# Magnitude scan

Use this check to tune objective and regularization weights before long
optimization runs.

## What it checks

For each term, MrHyDE prints:

- `unweighted`
- `weight`
- `weighted = unweighted * weight`

Pick weights so weighted terms are in a useful balance.

## How to run

Set under `Analysis:ROL2:General:`:

```yaml
Do magnitude scan: true
FD Check Random Seed: 42
FD Check Random Scale: 1.0e-8
```

For scan-only runs, set `Iteration Limit: 0` in `Status Test:` and keep
other FD or HessVec checks off.

## Expected output

The run prints one table like:

```text
[MAGNITUDE-SCAN] probe at seeded random ctrl (seed=42, scale=1e-08).
  term                            type      unweighted        weight      weighted
  --------------------------------------------------------------------------------
  EM Energy                       obj        2.007e-39     1.000e+35     2.007e-04
  RegObj/l2reg                    reg        1.110e-30     1.000e+05     1.110e-25
  RegObj/curlreg                  reg        1.221e-19     1.000e+05     1.221e-14
  --------------------------------------------------------------------------------
  TOTAL                                      1.221e-19                   2.007e-04
```

`TOTAL weighted` can differ from ROL iter-0 value because the scan uses a
temporary probe control, then restores the optimizer iterate.

## Artifacts

- `./run_scan.sh` regenerates logs.
- `logs/scan.log` uses nonzero probe scale.
- `logs/scan_scale_0.log` uses zero probe scale.
