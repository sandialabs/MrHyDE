# Parallel Regression Quick Commands

```bash
export MRHYDE_ROOT=path/to/mrhyde  # required by decompose script

cd MrHyDE/sandbox/testCases/parallel_regression/meshes  # mesh workspace
# in Cubit: run 3D_tet/mesh.jou and 3D_hex/mesh.jou to create mesh.exo first
./decompose_exodus_script                     # build np*_* for every mesh subdir (3D_tet, 3D_hex, ...)
./decompose_exodus_script 3D_tet              # build only tet decompositions
./decompose_exodus_script 3D_hex              # build only hex decompositions

cd ../thermal                                 # thermal regression workspace
./run.sh                                      # run tet + hex, all partitions
./run.sh tet                                  # run tet only, all partitions
./run.sh hex                                  # run hex only, all partitions
./run.sh np16_hsfc                            # run one partition for tet + hex
./run.sh tet np16_hsfc                        # run one tet partition
./run.sh hex np16_hsfc                        # run one hex partition

# Outputs:
# - runs/tet/<partition>/{input.yaml,output.log,time.log}
# - runs/hex/<partition>/{input.yaml,output.log,time.log}
# - convergence tables printed at end of run.sh
```

```text
=== convergence table: tet T=1 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        1.11211e-08
np16_multikl           | OK        |            1 |        8.91788e-09
np16_multikl_f         | OK        |            1 |        2.09882e-08
np16_rcb               | OK        |            1 |        2.26861e-08
np16_rib               | OK        |            1 |        3.57205e-08
np32_hsfc              | OK        |            1 |        9.76184e-09
np32_multikl           | OK        |            1 |        1.35761e-08
np32_multikl_f         | OK        |            1 |        1.31728e-08
np32_rcb               | OK        |            1 |        8.24991e-09
np32_rib               | OK        |            1 |        1.95135e-08
np8_hsfc               | OK        |            1 |        2.93887e-08
np8_multikl            | OK        |            1 |           2.32e-09
np8_multikl_f          | OK        |            1 |        3.93654e-08
np8_rcb                | OK        |            1 |        3.18486e-08
np8_rib                | OK        |            1 |        2.39286e-08

=== convergence table: hex T=1 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        4.17547e-10
np16_multikl           | OK        |            1 |        5.00842e-10
np16_multikl_f         | OK        |            1 |        1.95305e-10
np16_rcb               | OK        |            1 |        2.86179e-10
np16_rib               | OK        |            1 |        3.51678e-10
np32_hsfc              | OK        |            1 |        3.21966e-10
np32_multikl           | OK        |            1 |        3.23029e-10
np32_multikl_f         | OK        |            1 |        1.74475e-10
np32_rcb               | OK        |            1 |        2.32696e-10
np32_rib               | OK        |            1 |        1.98936e-10
np8_hsfc               | OK        |            1 |         4.1274e-10
np8_multikl            | OK        |            1 |        4.98067e-10
np8_multikl_f          | OK        |            1 |        3.93734e-10
np8_rcb                | OK        |            1 |        2.33446e-10
np8_rib                | OK        |            1 |        3.69057e-10

=== convergence table: tet T=2 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        1.89099e-08
np16_multikl           | OK        |            1 |        2.08591e-08
np16_multikl_f         | OK        |            1 |        8.69039e-09
np16_rcb               | OK        |            1 |        1.46268e-08
np16_rib               | OK        |            1 |        7.97105e-09
np32_hsfc              | OK        |            1 |        1.56265e-08
np32_multikl           | OK        |            1 |        1.72451e-08
np32_multikl_f         | OK        |            1 |        6.62909e-09
np32_rcb               | OK        |            1 |        1.24061e-08
np32_rib               | OK        |            1 |        1.57744e-08
np8_hsfc               | OK        |            1 |        1.01113e-08
np8_multikl            | OK        |            1 |        1.57962e-08
np8_multikl_f          | OK        |            1 |        5.78246e-09
np8_rcb                | OK        |            1 |        8.92265e-09
np8_rib                | OK        |            1 |        2.18178e-08

=== convergence table: hex T=2 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        2.22543e-10
np16_multikl           | OK        |            1 |         2.3006e-10
np16_multikl_f         | OK        |            1 |        1.87459e-10
np16_rcb               | OK        |            1 |        2.00047e-10
np16_rib               | OK        |            1 |        1.78179e-10
np32_hsfc              | OK        |            1 |        3.01117e-10
np32_multikl           | OK        |            1 |        1.88141e-10
np32_multikl_f         | OK        |            1 |        1.76602e-10
np32_rcb               | OK        |            1 |        1.91551e-10
np32_rib               | OK        |            1 |        1.52839e-10
np8_hsfc               | OK        |            1 |        2.68928e-10
np8_multikl            | OK        |            1 |        1.73144e-10
np8_multikl_f          | OK        |            1 |        1.75172e-10
np8_rcb                | OK        |            1 |        2.29178e-10
np8_rib                | OK        |            1 |        1.80916e-10

=== convergence table: tet T=3 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        1.69852e-08
np16_multikl           | OK        |            1 |        1.69131e-08
np16_multikl_f         | OK        |            1 |         2.2174e-08
np16_rcb               | OK        |            1 |        1.44743e-08
np16_rib               | OK        |            1 |         2.3556e-08
np32_hsfc              | OK        |            1 |        1.91975e-08
np32_multikl           | OK        |            1 |        1.36622e-08
np32_multikl_f         | OK        |            1 |        1.68008e-08
np32_rcb               | OK        |            1 |        1.55348e-08
np32_rib               | OK        |            1 |        1.57404e-08
np8_hsfc               | OK        |            1 |        1.71068e-08
np8_multikl            | OK        |            1 |        2.17432e-08
np8_multikl_f          | OK        |            1 |        2.42839e-08
np8_rcb                | OK        |            1 |          1.809e-08
np8_rib                | OK        |            1 |        1.82674e-08

=== convergence table: hex T=3 ===
partition              | status    | newton_iters |  scaled_nonlin_res
----------------------------------------------------------------------
np16_hsfc              | OK        |            1 |        3.57748e-10
np16_multikl           | OK        |            1 |        3.86138e-10
np16_multikl_f         | OK        |            1 |        3.06952e-10
np16_rcb               | OK        |            1 |        4.27813e-10
np16_rib               | OK        |            1 |        5.62328e-10
np32_hsfc              | OK        |            1 |        3.44457e-10
np32_multikl           | OK        |            1 |        3.78846e-10
np32_multikl_f         | OK        |            1 |        2.55981e-10
np32_rcb               | OK        |            1 |        3.45755e-10
np32_rib               | OK        |            1 |        4.44593e-10
np8_hsfc               | OK        |            1 |        4.62739e-10
np8_multikl            | OK        |            1 |        4.12324e-10
np8_multikl_f          | OK        |            1 |        3.08942e-10
np8_rcb                | OK        |            1 |         4.5907e-10
np8_rib                | OK        |            1 |        4.78977e-10
```
