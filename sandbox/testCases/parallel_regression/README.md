# parallel_regression

Each physics example runs the same problem across a matrix of
pre-decomposed Exodus meshes and checks whether the L2 error depends on the
partitioner.

## Meshes

`meshes/decompose_exodus_script` splits `meshes/3D_<mesh>/mesh.exo` into
`np{8,16,32}_{hsfc,rcb,rib,multikl,multikl_f}/` subdirectories using SEACAS
`decomp`. Requires `MRHYDE_ROOT` set so `${MRHYDE_ROOT}/trilinos_install/install/bin/decomp`
resolves.

- `meshes/3D_tet/mesh.jou` - Cubit journal for `[0,1]^3` unit cube, element size 0.1, tetmesh scheme.
- `meshes/3D_hex/mesh.jou` - Cubit journal for `[0,1]^3` with a spherical inclusion (kept), element size 0.025, hex scheme.

Both `mesh.exo` files are gitignored. Regenerate in Cubit, then:

```
export MRHYDE_ROOT=path/to/mrhyde
cd meshes && ./decompose_exodus_script          # all mesh dirs
./decompose_exodus_script 3D_tet                # one mesh dir
```

## Physics examples

Each has its own README with problem statement, run instructions, and results.

- `thermal/` - HGRAD steady Poisson (scalar T). The original tet DBC bug reproducer.
- `porous_mixed/` - HDIV flux + HVOL pressure steady mixed Poisson.
- `maxwell_hcurl/` - HCURL E + HDIV B transient Maxwell with a manufactured solution.

## Post-processing

Each example directory has an `l2_summary.py` for its log format and fields.
Run it against a runs tree:

```
cd [thermal/porous_mixed/maxwell_hcurl]        && python3 l2_summary.py runs/
```
