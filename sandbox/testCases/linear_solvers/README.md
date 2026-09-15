# linear_solvers

Compares preconditioners on one 3D PEC cavity across mesh sizes. Problem
setup is in `sweep.py`.

## Run

```bash
./sweep.py gen        # write the decks
./sweep.py run        # run them
./sweep.py report     # tables
```

Two studies: `h` refines the mesh at fixed dt (the default), `cfl` holds the
finest mesh and shortens dt. `run` and `report` default to the four smallest
meshes; pass a study, deck names, case tags, `all`, or a Krylov solver, in any
order:

```bash
./sweep.py run refmaxwell_p2v3 N40x20x10
./sweep.py report jacobi refmaxwell_p2v3
./sweep.py run cfl blocktri_cheb
./sweep.py run all
```

Output: `runs/<krylov>/<deck>/<mesh>/`. Each run directory is self-contained.

If a table cell is a dash, parse the log directly:

```bash
../../../scripts/data_processing/parse_log.py runs/tfqmr/jacobi/N16x8x4/mrhyde.log
```

## Krylov solver

Default is `bicgstab`. Options: `bicgstab`, `gmres` (Block GMRES), `gcrodr`,
`tfqmr`. Add other Belos names to `KRYLOV` in `sweep.py`.

```bash
./sweep.py gen gmres && ./sweep.py run gmres && ./sweep.py report gmres
```

## Results

A representative slice of the 35 decks.
Regenerate with `./sweep.py report all` and `./sweep.py report cfl`.

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                           N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
-------------------------------------------------------------------------------------------------
jacobi                           26.8      41.6      56.7      71.0      82.6      94.2     107.3
blocktri_cheb                    12.5      15.4      17.2      22.0      26.1      27.9      33.8
blocktri_amg                     17.5      25.0      42.8      59.2      77.6      92.6     108.2
refmaxwell_p2v3                   7.4       8.2       9.8      10.8      11.7      12.2      12.8
refmaxwell_p2v3_chebpivot         6.3       7.8       9.1      10.0      10.7      11.1      11.8
refmaxwell_dl02_mode1             8.7      10.2      38.8      42.6      46.8      73.8      64.8
maxwell1_emin0                    6.0       6.6      10.2      12.0      13.7      14.8      15.9
blocktri_chebpivot                7.2       9.8      11.7      13.8      16.6      19.2      22.4
blocktri_cheb_lumpedpivot        14.4      16.8      19.8      21.9      25.0      30.6      30.6
blocktri_cheb_pointweight        12.9      15.8      20.3      22.5      27.8      31.5      36.6
blocktri_cheb_base               30.2      60.2      88.2     127.2     162.3     200.8     229.5
blocktri_cheb_gamma05            10.8      12.4      16.1      18.8      22.0      25.6      28.4
blocktri_directschur              8.5       8.5       8.7       8.0       8.1       8.0       7.9
blocktri_cheb_lower              12.9      17.1      19.3      24.1      27.4      29.5      32.3
refmaxwell_p2v3_lower             8.7       8.9      10.9      12.2      13.2      13.5      14.0
cheb_block_tri                    9.8      12.9      15.7      18.2      21.2      24.9      28.3

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                           N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
-------------------------------------------------------------------------------------------------
jacobi                           0.05      0.08      0.24      0.61      1.33      2.44      4.56
blocktri_cheb                    0.08      0.08      0.20      0.47      1.16      1.82      3.29
blocktri_amg                     0.14      0.26      0.87      2.48      6.62     14.54     21.85
refmaxwell_p2v3                  0.20      0.17      0.37      0.76      1.49      2.60      4.14
refmaxwell_p2v3_chebpivot        0.11      0.15      0.32      0.66      1.29      2.22      3.71
refmaxwell_dl02_mode1            0.12      0.15      0.83      1.93      3.78      9.59     13.19
maxwell1_emin0                   0.10      0.22      0.50      1.26      2.74      5.09      8.82
blocktri_chebpivot               0.05      0.07      0.15      0.36      0.76      1.44      2.55
blocktri_cheb_lumpedpivot        0.05      0.08      0.19      0.44      0.92      1.79      2.91
blocktri_cheb_pointweight        0.05      0.08      0.21      0.45      0.99      1.84      3.37
blocktri_cheb_base               0.07      0.13      0.30      0.95      2.04      4.31      7.97
blocktri_cheb_gamma05            0.06      0.07      0.17      0.43      0.90      1.63      2.75
blocktri_directschur             0.08      0.19      2.16     12.21     62.14    186.10    484.70
blocktri_cheb_lower              0.06      0.08      0.19      0.47      0.97      1.75      3.02
refmaxwell_p2v3_lower            0.15      0.17      0.38      0.81      1.56      2.76      4.37
cheb_block_tri                   0.05      0.07      0.20      0.46      0.87      1.78      2.90
```

CFL study, same mesh, shrinking dt:

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                         cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
---------------------------------------------------------------------
jacobi                       107.3   49.7   29.7   17.2    9.0    5.5
blocktri_cheb                 33.8   22.5   12.6    8.7    7.9    7.1
blocktri_amg                 108.2   36.8   18.1    8.8    5.6    4.5
refmaxwell_p2v3               12.8    9.0    7.5    6.0    5.2    4.4
refmaxwell_p2v3_chebpivot     11.8    8.5    6.5    5.2    4.2    3.6
refmaxwell_dl02_mode1         64.8   23.8   12.0    6.2    5.2    4.4
maxwell1_emin0                15.9   10.8    7.8    6.1    5.2    4.3
blocktri_chebpivot            22.4   15.5    9.2    4.4    3.8    3.8
blocktri_cheb_lumpedpivot     30.6   21.6   12.3    8.2    7.1    7.1
blocktri_cheb_pointweight     36.6   24.8   13.7    8.5    7.1    7.1
blocktri_cheb_base           229.5   90.8   46.3   22.5   13.3    8.6
blocktri_cheb_gamma05         28.4   19.7   12.3   10.5    8.6    7.5
blocktri_directschur           7.9    7.8    7.8    7.3    6.8    6.4
blocktri_cheb_lower           32.8   25.2   13.3    8.7    7.9    7.1
refmaxwell_p2v3_lower         14.0   10.8    8.7    7.5    5.9    4.4
cheb_block_tri                27.8   21.5   12.2    7.3    5.4    4.6
sgs_damped                       -      -      -      -    9.8    3.7
schwarz                          -      -      -      -   18.1    6.2

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                         cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
---------------------------------------------------------------------
jacobi                        4.54   2.39   1.55   1.08   0.80   0.66
blocktri_cheb                 3.17   2.48   1.83   1.70   1.61   1.51
blocktri_amg                 21.26   8.43   4.69   2.93   2.25   2.04
refmaxwell_p2v3               4.50   4.64   3.67   2.96   2.85   2.48
refmaxwell_p2v3_chebpivot     3.69   3.27   2.95   2.57   2.32   2.08
refmaxwell_dl02_mode1        17.39   7.56   5.62   3.45   2.44   2.22
maxwell1_emin0                9.03   6.67   5.07   4.32   3.86   4.33
blocktri_chebpivot            2.53   2.07   1.67   1.50   1.42   1.43
blocktri_cheb_lumpedpivot     3.02   2.46   2.05   1.60   1.58   1.62
blocktri_cheb_pointweight     3.56   2.79   2.06   1.78   1.60   1.62
blocktri_cheb_base            8.79   3.81   2.22   1.39   1.15   0.93
blocktri_cheb_gamma05         2.79   2.33   2.34   1.72   1.62   1.53
blocktri_directschur        489.80 511.00 503.10 474.90 505.90 501.20
blocktri_cheb_lower           2.85   2.31   1.65   1.35   1.25   1.19
refmaxwell_p2v3_lower         4.16   3.63   2.92   2.67   2.33   1.90
cheb_block_tri                2.42   2.01   1.45   1.32   1.03   0.97
sgs_damped                       -      -      -      -   1.55   0.84
schwarz                          -      -      -      -   2.42   1.27
```
