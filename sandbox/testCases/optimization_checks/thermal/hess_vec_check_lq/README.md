# Thermal LQ gradient and HessVec checks

Distributed-source thermal LQ optimal control on `[0,1]^3`. Checks the
gradient and HessVec identities, then compares exact vs FD HessVec cost
across mesh sizes.

## How to run

```bash
./run.sh                  # default: check + exact/fd solves at N = 4, 8, 16
./run.sh check            # gradient + LQ-operator identity checks only
./run.sh exact-1e-4-N32   # single case: exact hessvec, gamma=1e-4, NX=32
NP=8 ./run.sh             # override MPI ranks (default 4)
```

`./run.sh` writes logs to `logs/mrhyde_<mode>.log` and prints a summary.

## Results (`gamma=1e-4`, np=4)

The exact path converges in one outer iteration at every mesh; FD needs
more outer iterations and far more inner CG. Final objective values match.

### Check mode

| GRAD-CHECK | HESSVEC-CHECK | SECANT-IDENTITY | HV-BILINEARITY |
|-----------:|--------------:|----------------:|---------------:|
| 2.24e-07   | 1.03e-14      | 9.44e-15        | 8.22e-16       |

### Sweep (`./run.sh`)

| mode          | iter0 value | final value | final gnorm | native L2 err | n_outer | sum CG | wall (s) |
| :------------ | :---------: | :---------: | :---------: | :-----------: | :-----: | :----: | :------: |
| exact-1e-4-N4 |  4.17e-02   |  2.81e-02   |  1.47e-15   |   2.70e-02    |    1    |   5    |   1.7    |
| fd-1e-4-N4    |  4.17e-02   |  2.81e-02   |  2.81e-19   |   2.70e-02    |    3    |  392   |   16.3   |
| exact-1e-4-N8 |  5.64e-02   |  3.43e-02   |  4.65e-16   |   3.50e-03    |    1    |   34   |   3.2    |
| fd-1e-4-N8    |  5.64e-02   |  3.43e-02   |  4.12e-19   |   3.50e-03    |    3    |  462   |   32.0   |
| exact-1e-4-N16|  6.09e-02   |  3.59e-02   |  3.02e-15   |   5.78e-04    |    1    |   57   |   6.9    |
| fd-1e-4-N16   |  6.09e-02   |  3.59e-02   |  1.15e-19   |   5.78e-04    |    6    |  445   |   62.1   |

## Which path runs

`other_decks/exact/` declares `src_gate` and gets exact HessVec.
`other_decks/fd/` omits it and falls back to FD-of-gradients.
