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
=== convergence table: tet ===
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

=== convergence table: hex ===
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
```
