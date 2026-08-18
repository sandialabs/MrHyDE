# thermal (HGRAD, steady Poisson)

Steady heat equation on `[0,1]^3` with `T=0` Dirichlet on all boundaries.
Manufactured solution `T = sin(2pi x)*sin(2pi y)*sin(2pi z)`, source
`f = 12*pi^2 * sin*sin*sin`. Single scalar T on HGRAD; direct solver.

## Run

```
./run.sh                          # tet + hex, all partitions, T=1 only
./run.sh tet                      # tet only
./run.sh --T-order 1-3            # sweep T=1..3
./run.sh --T-order 1-3 tet np8_hsfc   # one cell of the sweep
python3 l2_summary.py             # wide table across runs/
```

Outputs to `runs/<mesh>/T<order>/<partition>/{input.yaml,output.log,time.log}`.

