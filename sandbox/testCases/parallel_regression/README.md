# Parallel Regression Quick Commands

```bash
export MRHYDE_ROOT=path/to/mrhyde  # required by decompose script

cd MrHyDE/sandbox/testCases/parallel_regression/meshes  # mesh workspace
# in Cubit: run 3D_tet/mesh.jou and 3D_hex/mesh.jou to create mesh.exo first
./decompose_exodus_script                     # build np*_* for every mesh subdir (3D_tet, 3D_hex, ...)
./decompose_exodus_script 3D_tet              # build only tet decompositions
./decompose_exodus_script 3D_hex              # build only hex decompositions

cd ../thermal                                 # thermal regression workspace
./run.sh                                      # T=1 only, tet + hex, all partitions
./run.sh tet                                  # T=1, tet only, all partitions
./run.sh hex                                  # T=1, hex only, all partitions
./run.sh np16_hsfc                            # T=1, one partition for tet + hex
./run.sh tet np16_hsfc                        # T=1, one tet partition
./run.sh hex np16_hsfc                        # T=1, one hex partition
./run.sh --T-order 3                          # T=3 only
./run.sh --T-order 1-3                        # sweep T=1..3
./run.sh --T-order 1-3 tet np16_hsfc          # sweep, one mesh + partition

# Outputs (disc T<order> subdir added per order):
# - runs/<mesh_type>/T<order>/<partition>/{input.yaml,output.log,time.log}
# - convergence tables printed at end of run.sh (one per mesh, order)

# Wide summary across all T's for a completed sweep:
python3 ../scripts/l2_iters_wide.py                     # defaults to ../thermal/runs
python3 ../scripts/l2_iters_wide.py /path/to/runs_dir   # any runs_* tree
```

Sample summary (from a fixed-mesh p-sweep on `runs`):

```text
==========================================================================================================
# tet
==========================================================================================================
partition              | T1 iters | T2 iters | T3 iters |         T1 L2  |         T2 L2  |         T3 L2 
----------------------------------------------------------------------------------------------------------
np16_hsfc              |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np16_multikl           |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np16_multikl_f         |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np16_rcb               |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np16_rib               |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np32_hsfc              |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np32_multikl           |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np32_multikl_f         |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np32_rcb               |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np32_rib               |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np8_hsfc               |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np8_multikl            |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np8_multikl_f          |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np8_rcb                |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 
np8_rib                |        1 |        1 |        1 |  3.388720e-02  |  1.859990e-03  |  1.082810e-04 

==========================================================================================================
# hex
==========================================================================================================
partition              | T1 iters | T2 iters | T3 iters |         T1 L2  |         T2 L2  |         T3 L2 
----------------------------------------------------------------------------------------------------------
np16_hsfc              |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np16_multikl           |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np16_multikl_f         |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np16_rcb               |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np16_rib               |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np32_hsfc              |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np32_multikl           |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np32_multikl_f         |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np32_rcb               |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np32_rib               |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np8_hsfc               |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np8_multikl            |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np8_multikl_f          |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np8_rcb                |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
np8_rib                |        1 |        1 |        1 |  1.200570e-03  |  4.525500e-05  |  6.034330e-07 
```
