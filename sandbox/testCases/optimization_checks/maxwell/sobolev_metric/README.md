# Sobolev metric for an H(curl) control

A trust-region method solves `H p = -g` with truncated CG; the iteration count depends on
the inner product on the control. This example shows a Sobolev inner product beats the
Euclidean one, how to switch it on with one deck sublist, and how it holds up under mesh
refinement.

Problem: [3D transient Maxwell E-B optimal control](https://arxiv.org/pdf/2605.00212).
An H(curl) control enters as
`curl(ctrl_current)` on two blocks; the objective is field energy plus l2 and curl-curl
regularization. The matched metric `P = alpha1*M + alpha2*S` (mass + curl-curl seminorm)
uses the objective's own weights. Measurement: the truncated-CG count for one trust-region
step from a random control.

## The one knob

No sublist is the Euclidean metric; `apply: exact` factors `P`; `apply: iterative` applies
the same `P` by RefMaxwell-preconditioned CG.

```yaml
    # Riesz map usig direct-sovler
    Parameter Riesz map:          # decks/input_sobolev.yaml (exact)
      type: regularizer           # read alpha1, alpha2 from the objective
      apply: exact

    # Riesz map usig iterative solver
    Parameter Riesz map:          # decks/input_sobolev_iter.yaml (RefMaxwell)
      type: regularizer
      apply: iterative
      Iterative Settings:
        preconditioner: RefMaxwell
        xml param file: decks/refmaxwell_metric.xml
        tolerance: 1.0e-14
        # solver: CG          # Krylov solver for P^-1: CG (default) or GMRES
        # max iterations: 500 # RefMaxwell-CG iteration cap (default)
```

ROL preconditions the CG with the control vector's dual, so setting the metric is all that
is needed.

## Results

CG iterations for one trust-region step. `exact` factors `P`; `RefMaxwell` applies the same
`P` iteratively.

Refine time, `(2, 2, N_t)`:

| N_t | control DoFs/slab | Euclid | Sobolev (exact) | Sobolev (RefMaxwell) |
|---|---|---|---|---|
| 4   | 64 | 71  | 10 | 10 |
| 8   | 64 | 123 | 21 | 21 |
| 16  | 64 | 186 | 45 | 44 |
| 32  | 64 | 217 | 60 | 61 |
| 64  | 64 | 225 | 67 | 67 |
| 128 | 64 | 230 | 69 | 68 |

Refine along z, `(2, N_Z, 8)` (control grows 7x):

| N_Z | control DoFs/slab | Euclid | Sobolev (exact) | Sobolev (RefMaxwell) |
|---|---|---|---|---|
| 2  | 64  | 123 | 21 | 21 |
| 6  | 160 | 334 | 20 | 20 |
| 12 | 304 | 563 | 20 | 20 |
| 18 | 448 | 730 | 20 | 20 |

Refine transversely, `(N_X, 2, 4)` (control grows 16x):

| N_X | control DoFs/slab | Euclid | Sobolev (exact) | Sobolev (RefMaxwell) |
|---|---|---|---|---|
| 2 | 64   | 71  | 10  | 10  |
| 4 | 256  | 303 | 49  | 50  |
| 6 | 576  | 535 | 91  | 91  |
| 8 | 1024 | 754 | 109 | 109 |

Refine time and z together, `(2, 12, N_t)`:

| N_t | control DoFs/slab | Euclid | Sobolev (exact) | Sobolev (RefMaxwell) |
|---|---|---|---|---|
| 4  | 304 | 394  | 10 | 10 |
| 8  | 304 | 563  | 20 | 20 |
| 16 | 304 | 759  | 39 | 39 |
| 32 | 304 | 1116 | 78 | 78 |

RefMaxwell inner tolerance, on the hardest level of each family (outer CG count; exact
metric for reference):

| case (N_X, N_Z, N_t) | exact | 1e-14 | 1e-10 | 1e-8 |
|---|---|---|---|---|
| (2, 2, 128)  | 69  | 68  | 73  | 83  |
| (2, 18, 8)   | 20  | 20  | 21  | 26  |
| (8, 2, 4)    | 109 | 109 | 118 | 130 |
| (2, 12, 32)  | 78  | 78  | 83  | 86  |

The outer solve never cliffs: it converges even at 1e-2.
But the vector-contract defined by ROL checking whether the metric is a valid inner product does fall
off. Enable it with the check flags:

```yaml
        Do grad+hessvec check: true   # gate for the check below
        Do vector check: true         # print the reflexivity consistency check
```

Reflexivity, the metric's inner CG per apply, and the outer CG count, against the RefMaxwell
tolerance (s7):

| tol | reflexivity | metric CG/apply | outer CG |
|---|---|---|---|
| 1e-14 | 4.8e-12 | 71 | 20 |
| 1e-10 | 7.3e-8  | 62 | 21 |
| 1e-8  | 7.4e-6  | 57 | 26 |
| 1e-6  | 1.0e-3  | 43 | 34 |
| 1e-4  | 1.0e-1  | 32 | 47 |
| 1e-2  | 9.1     | 17 | 83 |

Reflexivity is `||x - P^-1(P x)||` in the P-norm: the round-trip error of applying the metric
then its inverse, zero for an exact inverse. Loosening the RefMaxwell solve tolerance never
cliffs the solve (it still converges at 1e-2!!).

A looser tolerance cuts the per-apply RefMaxwell-CG count but raises the
outer trust-region count (each an expensive reduced-Hessian apply); whether that trade makes 
sense wil likel depend on the mehs/problem. 


## Run it

```bash
ln -sf <path-to-build>/src/mrhyde mrhyde    # or: export MRHYDE_BIN=<...>/src/mrhyde
./run.sh            # the refinement sweeps (sweeps.txt), single rank, about an hour
./run_tol.sh        # the inner-tolerance study (tol.txt), a few minutes
python3 table.py    # prints all the tables above
```

Edit `sweeps.txt` for a shorter run (the `dt` rows use only the small `s1` mesh). Euclid
needs up to ~1100 CG iterations, so the inner cap in the decks is 2000.

## Files

- `decks/input_{euclid,sobolev,sobolev_iter}.yaml` - the three decks, differing only in the
  Parameter Riesz map sublist; `__MESH__`, `__NSTEPS__` filled by `run.sh`.
- `decks/refmaxwell_metric.xml` - MueLu settings for the RefMaxwell metric apply.
- `meshes/mesh_s[1-8].yaml`, `meshes.txt`, `sweeps.txt`, `tol.txt` - mesh family, sizes, and
  the refinement and tolerance run lists.
- `mkctrl.py`, `run.sh`, `run_tol.sh`, `table.py` - control generator, runners, table.

## Other options

`type: sobolev` sets `l2 weight` and `seminorm weight` by hand instead of reading them from
the objective; `solver` and `max iterations` in `Iterative Settings` tune the RefMaxwell CG.
