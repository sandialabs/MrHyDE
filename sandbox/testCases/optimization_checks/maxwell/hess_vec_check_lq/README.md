# FD gradient and HessVec checks (LQ Maxwell)

This sandbox validates gradient and Hessian-vector checks on a linear
quadratic Maxwell control objective. It is the quick correctness gate
before longer trust-region studies.

## How to run

```bash
./run_exact_sweep.sh
./run_fd_sweep.sh
python plot_hess_vec_check.py
```

## Activation flags

Under `Analysis:ROL2:General:`:

```yaml
Do grad+hessvec check: true
Do exact hessvec check: true
Do algebraic hessvec check: true
Do secant identity check: true
FD Check Seed: 3
```

`seed` modes add `FD Check Random Seed: 42` so diagnostic probes use a
nonzero random control while the optimizer iterate stays unchanged.

## Exact vs FD path in this test

- `other_decks_exact/` has `src_gate` and uses exact HessVec.
- `other_decks_fd/` omits `src_gate` and uses FD-of-gradients fallback.

## Expected results

- Seeded FD gradient checks should be healthy (`rel_err` around `1e-4` to
  `1e-3` in current logs).
- Baseline checks at `ctrl = 0` are intentionally degenerate and can show
  large FD relative error.
- `|hv0|` should be zero.
- Exact path should give smaller secant and bilinearity residuals than the
  FD fallback in this LQ case.

### FD gradient table

| Config   | Probe   | `grad'*dir` | best abs_err @ h | rel_err | Take-away                   |
| :------- | :------ | :---------- | :--------------- | :------ | :-------------------------- |
| baseline | iterate | +3.10e-02   | 6.40e-01 @ 1e-12 | 2.1e+01 | Degenerate `g.d` near noise |
| seed     | random  | +6.69e+11   | 3.04e+08 @ 1e-04 | 4.5e-04 | Healthy random probe        |

### HessVec table (exact vs FD)

| Config   | path  | `|secant|/ref` | `|bilin|/ref` | `|hv0|` | `<v,Hv>_min` |
| :------- | :---- | -------------: | ------------: | ------: | -----------: |
| baseline | exact |       3.83e-08 |      7.24e-08 |       0 |     9.42e+11 |
| baseline | fd    |       1.35e-05 |      6.89e-05 |       0 |     9.42e+11 |
| seed     | exact |       3.83e-08 |      1.16e-07 |       0 |     8.09e+11 |
| seed     | fd    |       1.35e-05 |      8.56e-05 |       0 |     8.09e+11 |

## Common failure and fix

If FD gradient check reports large relative error:

- Add `FD Check Random Seed: 42`.
- If regularization dominates, lower or zero regularization weights for
  the diagnostic run.
