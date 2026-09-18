# FD gradient and HessVec checks (LQ Maxwell)

Gradient and HessVec correctness on a linear-quadratic Maxwell control
objective. The quick gate before longer trust-region studies.

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

## Which path runs

- `other_decks_exact/` has `src_gate` and uses exact HessVec.
- `other_decks_fd/` omits `src_gate` and uses FD-of-gradients fallback.

## Results

`|hv0|` must be zero. Seeded probes land near `rel_err` 1e-4; `baseline`
probes at `ctrl = 0` are degenerate on purpose and read much worse. Exact
beats the FD fallback on secant and bilinearity residuals.

### FD gradient

| Config   | Probe   | `grad'*dir` | best abs_err @ h | rel_err | Take-away                   |
| :------- | :------ | :---------- | :--------------- | :------ | :-------------------------- |
| baseline | iterate | +3.10e-02   | 6.40e-01 @ 1e-12 | 2.1e+01 | Degenerate `g.d` near noise |
| seed     | random  | +6.69e+11   | 3.04e+08 @ 1e-04 | 4.5e-04 | Healthy random probe        |

### HessVec, exact vs FD

| Config   | path  | `|secant|/ref` | `|bilin|/ref` | `|hv0|` | `<v,Hv>_min` |
| :------- | :---- | -------------: | ------------: | ------: | -----------: |
| baseline | exact |       3.83e-08 |      7.24e-08 |       0 |     9.42e+11 |
| baseline | fd    |       1.35e-05 |      6.89e-05 |       0 |     9.42e+11 |
| seed     | exact |       3.83e-08 |      1.16e-07 |       0 |     8.09e+11 |
| seed     | fd    |       1.35e-05 |      8.56e-05 |       0 |     8.09e+11 |

## Telling which path actually ran

`|secant|/ref` identifies the dispatch without reading the deck. On a
non-LQ problem the FD fallback reports exactly `1.00`, because `H` changes
along `[x, x+v]` and `grad(x+v)-grad(x)` is then unrelated to `H(x)v`. So
`|secant|/ref = 1` with otherwise sane diagnostics means `src_gate` was not
found and ROL fell back to FD-of-gradients. On an LQ problem both paths sit
near zero, so use `|bilin|/ref` instead: on a nonlinear problem exact is
O(1) and FD is ~1e-5.

## Common failure and fix

If FD gradient check reports large relative error:

- Add `FD Check Random Seed: 42`.
- If regularization dominates, lower or zero regularization weights for
  the diagnostic run.
