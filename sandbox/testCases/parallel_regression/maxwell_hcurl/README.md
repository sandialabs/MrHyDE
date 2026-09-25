# maxwell_hcurl (E in HCURL + B in HDIV, transient E-B)

Transient Maxwell's equations on `[0,1]^3` with `E=0` Dirichlet on all boundaries.
Method of manufactured solutions: `Ez = sin(pi x)*sin(pi y)*sin(pi z)*cos(t)`, other
components zero; matching `B` and current `J` derived so Faraday and Ampere hold
exactly with `mu = eps = 1`, `sigma = 0`. Runs to `T = 0.5` with 10 DIRK-1,2
steps. 

## Run

```
./run.sh                          # tet + hex, all partitions, E=1 only
./run.sh tet                      # tet only
./run.sh --E-order 1-2            # sweep E=1..2 (B and quadrature follow)
./run.sh --E-order 1 tet np8_hsfc # one cell
python3 l2_summary.py             # wide table across runs/
```

Outputs to `runs/<mesh>/E<order>/<partition>/{input.yaml,output.log,time.log}`.
