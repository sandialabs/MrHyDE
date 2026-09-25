# porous_mixed (HDIV flux + HVOL pressure, steady mixed Poisson)

Mixed Darcy formulation on `[0,1]^3`: `u = -grad p`, `div u = f` with
`p=0` Dirichlet on all boundaries. Manufactured `p = sin(pi x)*sin(pi y)*sin(pi z)`,
`f = 3*pi^2 * sin*sin*sin`. Fixed pairing `p=0, u=1, quadrature=2`.
Direct solver.

## Run

```
./run.sh                          # tet + hex, all partitions
./run.sh tet                      # tet only
./run.sh tet np8_hsfc             # one cell
python3 l2_summary.py             # wide table across runs/
```

Outputs to `runs/<mesh>/<partition>/{input.yaml,output.log,time.log}` (no order
subdir).

