# thermal_hess_vec_check_lq_tracking

Same problem as `../hess_vec_check_lq`, but with a tracking-target objective
$\tfrac12(T - T_d)^2$, $T_d = 1$. The target is wrapped in an inactive
`trk_gate` scalar so `incrementalAdjointModel` zeroes it on the second-order
adjoint sweep, keeping the exact HessVec path inside LQ.

## Run

```bash
./run.sh                  # check + exact/fd sweep at N = 4, 8, 16
./run.sh check            # LQ operator identity checks only
NP=8 ./run.sh             # override MPI ranks (default 4)
```

## Results ($\gamma = 10^{-4}$, np=4)

Check mode (all at inner-solve floor):
`[GRAD-CHECK] 6.4e-8`,  `[HESSVEC-CHECK] 2.9e-14`,
`[SECANT-IDENTITY] 2.0e-14`,  `[HV-BILINEARITY] 8.2e-16`.

| N | exact iters / CG / wall | fd iters / CG / wall | speedup |
|:---:|:---:|:---:|:---:|
| 4  | 1 / 20 / 2.05 s | 3 / 393 / 16.7 s | 8.1x |
| 8  | 1 / 61 / 4.49 s | 4 / 481 / 33.9 s | 7.6x |
| 16 | 1 / 65 / 7.27 s | 5 / 480 / 63.8 s | 8.8x |

## Deck convention

`other_decks/exact/` declares both `src_gate` and `trk_gate`; the objective
is written as `'0.5*(T - trk_gate*1.0)*(T - trk_gate*1.0)'`. `other_decks/fd/`
omits `src_gate`, so `MrHyDE_Objective::hessVec` falls through to
FD-of-gradients. Dropping `trk_gate` from the exact deck reproduces the
`[HV-ZERO]`/`[HV-BILINEARITY]` breakdown covered by
`regression/thermal/hess_vec_check_lq_tracking_break/`.
