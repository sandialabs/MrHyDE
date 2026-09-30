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

4 ranks. Regenerate with `./sweep.py report all`,
`./sweep.py report all gmres`, `./sweep.py report cfl all`.

### h refinement, BiCGStab

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                            N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
--------------------------------------------------------------------------------------------------
jacobi                            31.3      48.3      67.8      84.4     102.4     114.2     129.4
block_diag                           -         -         -         -         -         -         -
blocktri_jac_jac                  15.1      43.4         -         -         -         -         -
blocktri_cheb                     14.5      17.6      20.3      26.2      30.7      33.0      40.2
blocktri_amg                      20.8      29.5      49.7      70.5      90.6     112.6     128.8
blocktri_hiptmair                  7.2       9.6      12.9      16.7      20.5      24.8      29.2
refmaxwell_p2v3                    8.5       9.7      11.4      12.7      13.7      14.3      15.1
refmaxwell_p2v3_filter             8.5       9.7      11.4      12.7      13.7      14.3      15.1
refmaxwell_p2v3_filter_off         8.5       9.7      11.4      12.7      13.7      14.3      15.1
refmaxwell_dl015                   8.0       9.2      11.5      12.9      13.5      14.2      15.2
refmaxwell_dl02_mode1              9.8      11.8      45.5      51.2      55.4      90.0      79.9
refmaxwell_dl02_additive           8.0       9.2      11.1      13.0      14.0      15.0      15.3
refmaxwell_dl02_121                8.0       9.2      11.3      12.4      12.5      13.8      14.0
maxwell1_emin0                     7.1       7.7      12.0      14.2      16.2      17.4      18.9
maxwell1_sa_rs                     7.1       7.7      12.7      14.4      16.7      19.4      21.6
maxwell1_sa_rs_edge                7.1       7.7      12.4      14.5      16.6      19.2      21.4
maxwell1_sa_rs_nodal               7.1       7.7      12.7      14.4      16.7      19.4      21.6
refmaxwell_unsmoothed              8.0       8.4      10.6      12.4      13.6      15.2      17.3
blocktri_ilu0                     24.5      34.5         -         -         -         -         -
blocktri_ilu1                     18.7      30.3      50.3      64.9         -         -         -
blocktri_amg_ilu                  19.6     143.0         -         -         -         -         -
block_diag_ilu                       -         -         -         -         -         -         -
blocktri_chebpivot                 7.9      11.3      13.4      16.1      19.4      23.1      25.7
blocktri_cheb_lumpedpivot         16.4      20.0      22.6      26.7      30.0      34.6      36.8
blocktri_cheb_pointweight         14.8      18.2      24.1      26.3      33.4      36.4      43.0
blocktri_directpivot               5.2      10.3         -      40.8         -         -         -
refmaxwell_p2v3_chebpivot          7.4       9.3      10.3      11.7      12.1      13.1      13.7
blocktri_cheb_base                38.5      65.6     109.3     151.1     188.3     232.5     271.8
blocktri_cheb_gamma05             12.5      14.3      18.7      22.0      26.1      29.6      34.4
blocktri_cheb_gamma08             13.9      17.3      19.4      23.3      27.5      32.7      37.6
blocktri_directschur              10.3      10.1      10.2       9.5       9.6       9.5       9.4
blocktri_cheb_lower               14.9      19.2      23.0      28.9      32.9      41.0      38.1
refmaxwell_p2v3_lower              9.7      10.6      12.8      14.3      15.5      15.7      16.5
sgs                                  -         -         -         -         -         -         -
sgs_l1                               -         -         -         -         -         -         -
sgs_damped                           -         -         -         -         -         -         -
schwarz                              -         -         -         -         -         -         -
cheb_mono                         64.5      46.5     137.9         -         -         -         -
cheb_block_diag                      -         -         -         -         -         -         -
cheb_block_tri                    11.2      14.9      18.2      21.6      24.8      29.0      33.0

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                            N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
--------------------------------------------------------------------------------------------------
jacobi                            0.05      0.23      0.34      0.69      1.61      3.06      5.21
block_diag                           -         -         -         -         -         -         -
blocktri_jac_jac                  0.07      0.15         -         -         -         -         -
blocktri_cheb                     0.05      0.10      0.21      0.44      0.94      1.62      3.02
blocktri_amg                      0.14      0.22      0.94      2.59      6.02     12.90     22.59
blocktri_hiptmair                 0.12      0.27      0.66      1.52      3.38      6.88     12.66
refmaxwell_p2v3                   0.12      0.19      0.41      0.85      1.44      2.47      4.16
refmaxwell_p2v3_filter            0.12      0.15      0.34      0.59      1.18      1.89      3.02
refmaxwell_p2v3_filter_off        0.13      0.19      0.41      0.76      1.51      2.46      4.03
refmaxwell_dl015                  0.09      0.14      0.41      0.84      1.42      2.49      4.09
refmaxwell_dl02_mode1             0.10      0.14      1.03      2.10      4.06     10.51     14.43
refmaxwell_dl02_additive          0.12      0.15      0.37      0.81      1.45      2.58      4.14
refmaxwell_dl02_121               0.13      0.17      0.48      0.95      1.74      3.17      4.66
maxwell1_emin0                    0.10      0.19      0.60      1.29      3.12      5.67      9.03
maxwell1_sa_rs                    0.11      0.18      0.59      1.52      2.93      5.65      9.67
maxwell1_sa_rs_edge               0.11      0.20      0.56      1.42      2.91      5.41      9.57
maxwell1_sa_rs_nodal              0.10      0.22      0.60      1.28      2.84      5.45      9.68
refmaxwell_unsmoothed             0.12      0.26      0.42      0.87      1.95      3.31      5.89
blocktri_ilu0                     0.08      0.15         -         -         -         -         -
blocktri_ilu1                     0.11      0.32      1.78      5.64         -         -         -
blocktri_amg_ilu                  0.12      1.13         -         -         -         -         -
block_diag_ilu                       -         -         -         -         -         -         -
blocktri_chebpivot                0.06      0.08      0.18      0.34      0.71      1.31      2.25
blocktri_cheb_lumpedpivot         0.06      0.08      0.31      0.41      0.91      1.68      2.87
blocktri_cheb_pointweight         0.05      0.08      0.20      0.44      0.97      1.77      3.05
blocktri_directpivot              0.05      0.07         -      2.49         -         -         -
refmaxwell_p2v3_chebpivot         0.11      0.18      0.37      0.67      1.19      2.13      3.48
blocktri_cheb_base                0.07      0.14      0.39      1.12      2.62      4.67      8.50
blocktri_cheb_gamma05             0.05      0.07      0.25      0.42      0.78      1.46      2.59
blocktri_cheb_gamma08             0.05      0.08      0.21      0.45      0.87      1.61      2.74
blocktri_directschur              0.09      0.18      2.43     12.62     60.48    170.80    447.30
blocktri_cheb_lower               0.06      0.09      0.24      0.52      0.93      1.85      2.83
refmaxwell_p2v3_lower             0.13      0.17      0.45      0.95      1.61      2.64      4.25
sgs                                  -         -         -         -         -         -         -
sgs_l1                               -         -         -         -         -         -         -
sgs_damped                           -         -         -         -         -         -         -
schwarz                              -         -         -         -         -         -         -
cheb_mono                         0.10      0.11      0.48         -         -         -         -
cheb_block_diag                      -         -         -         -         -         -         -
cheb_block_tri                    0.05      0.09      0.22      0.45      0.86      1.53      2.69
```

### h refinement, Block GMRES

```
Belos iterations per solve, mean, Block GMRES, 4 ranks

deck                            N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
--------------------------------------------------------------------------------------------------
jacobi                            41.2      68.3      91.8     121.4     143.7     165.2     185.6
block_diag                        44.5      82.0     118.3     163.5     196.2     222.5     253.8
blocktri_jac_jac                  18.9      32.7      97.3     165.9     239.8     321.8         -
blocktri_cheb                     18.1      25.0      30.2      34.6      38.7      43.8      48.9
blocktri_amg                      33.3      47.5      76.5     106.5     130.6     158.7     179.7
blocktri_hiptmair                 13.1      15.6      21.3      28.3      33.7      39.8      44.4
refmaxwell_p2v3                   14.3      16.1      19.6      21.2      23.3      24.3      25.3
refmaxwell_p2v3_filter            14.3      16.1      19.6      21.2      23.3      24.3      25.3
refmaxwell_p2v3_filter_off        14.3      16.1      19.6      21.2      23.3      24.3      25.3
refmaxwell_dl015                  14.0      16.1      20.0      21.2      23.0      24.4      25.5
refmaxwell_dl02_mode1             16.6      20.6      69.5      76.4      82.0     129.0     110.5
refmaxwell_dl02_additive          14.0      16.1      19.8      21.6      23.4      24.3      25.5
refmaxwell_dl02_121               14.0      16.1      19.8      21.2      22.1      23.3      23.7
maxwell1_emin0                    13.0      13.8      20.8      22.4      27.7      29.7      29.4
maxwell1_sa_rs                    13.0      13.8      21.1      24.2      29.0      31.4      32.7
maxwell1_sa_rs_edge               13.0      13.8      20.8      24.1      29.0      31.6      32.7
maxwell1_sa_rs_nodal              13.0      13.8      21.1      24.2      29.0      31.4      32.7
refmaxwell_unsmoothed             14.0      15.1      19.0      21.6      23.8      25.8      28.1
blocktri_ilu0                     36.5      50.3         -         -         -         -         -
blocktri_ilu1                     29.0      44.5      68.6      88.3         -         -         -
blocktri_amg_ilu                  31.1      95.0         -         -         -         -         -
block_diag_ilu                    72.6      95.6     145.2     186.0     224.4     262.9     300.0
blocktri_chebpivot                12.9      17.9      21.7      25.2      28.2      31.8      35.5
blocktri_cheb_lumpedpivot         19.8      23.5      26.8      31.6      35.4      40.1      45.0
blocktri_cheb_pointweight         19.0      25.9      32.0      36.9      42.4      47.7      54.3
blocktri_directpivot               7.7      12.6      17.4         -         -         -         -
refmaxwell_p2v3_chebpivot         13.1      15.1      18.0      19.4      21.2      22.1      23.2
blocktri_cheb_base                53.7      97.1     150.7     211.3     266.8     311.3     358.5
blocktri_cheb_gamma05             18.5      25.0      28.0      31.9      36.0      39.6      43.8
blocktri_cheb_gamma08             18.0      25.1      29.6      33.4      37.8      42.2      46.6
blocktri_directschur              16.2      16.3      15.0         -         -         -         -
blocktri_cheb_lower               18.6      25.0      31.0      35.2      39.5      44.1      49.2
refmaxwell_p2v3_lower             15.4      17.4      21.4      23.3      25.3      26.5      27.6
sgs                                  -         -         -         -         -         -         -
sgs_l1                               -         -         -         -         -         -         -
sgs_damped                           -         -         -         -         -         -         -
schwarz                              -         -         -         -         -         -         -
cheb_mono                         45.3      64.2     110.0     153.4     185.2     213.3     243.7
cheb_block_diag                   41.3      91.1     154.1     217.5     282.1     334.2     389.8
cheb_block_tri                    17.8      24.2      31.3      35.1      39.0      43.3      47.7

Forward solve wall time, seconds, Block GMRES, 4 ranks

deck                            N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
--------------------------------------------------------------------------------------------------
jacobi                            0.06      0.13      0.35      1.14      3.38      7.29     14.28
block_diag                        0.08      0.15      0.47      1.76      5.21     11.19     22.91
blocktri_jac_jac                  0.05      0.13      0.50      2.43      9.39     26.94         -
blocktri_cheb                     0.08      0.11      0.45      0.55      0.81      1.61      3.15
blocktri_amg                      0.14      0.22      0.90      2.47      6.56     15.76     31.08
blocktri_hiptmair                 0.13      0.18      0.56      1.73      2.96      5.91     10.52
refmaxwell_p2v3                   0.14      0.17      0.39      0.73      1.37      2.45      3.81
refmaxwell_p2v3_filter            0.11      0.15      0.32      0.68      1.10      1.85      2.87
refmaxwell_p2v3_filter_off        0.11      0.25      0.40      0.81      1.39      2.31      3.73
refmaxwell_dl015                  0.12      0.13      0.41      0.75      1.39      2.28      3.72
refmaxwell_dl02_mode1             0.17      0.16      0.91      1.92      3.72     11.92     15.96
refmaxwell_dl02_additive          0.11      0.16      0.39      0.71      1.48      2.40      3.97
refmaxwell_dl02_121               0.12      0.17      0.45      0.91      1.62      2.83      4.40
maxwell1_emin0                    0.11      0.20      0.61      1.32      2.56      4.60      7.68
maxwell1_sa_rs                    0.11      0.21      0.66      1.32      2.61      4.83      7.94
maxwell1_sa_rs_edge               0.17      0.19      0.55      1.20      2.64      4.76      7.94
maxwell1_sa_rs_nodal              0.11      0.20      0.67      1.23      2.66      4.79      7.99
refmaxwell_unsmoothed             0.10      0.19      0.43      0.89      1.76      3.07      5.31
blocktri_ilu0                     0.09      0.14         -         -         -         -         -
blocktri_ilu1                     0.11      0.30      1.60      4.31         -         -         -
blocktri_amg_ilu                  0.11      0.41         -         -         -         -         -
block_diag_ilu                    0.09      0.17      0.58      2.80      8.28     18.62     37.88
blocktri_chebpivot                0.06      0.08      0.24      0.40      0.69      1.22      2.22
blocktri_cheb_lumpedpivot         0.06      0.07      0.23      0.44      0.79      1.43      2.91
blocktri_cheb_pointweight         0.05      0.08      0.26      0.46      0.90      1.70      3.36
blocktri_directpivot              0.05      0.09      0.26         -         -         -         -
refmaxwell_p2v3_chebpivot         0.11      0.15      0.44      0.66      1.18      2.10      3.25
blocktri_cheb_base                0.08      0.17      0.57      2.81      8.77     20.03     41.59
blocktri_cheb_gamma05             0.06      0.09      0.26      0.42      0.74      1.36      2.73
blocktri_cheb_gamma08             0.05      0.08      0.29      0.42      0.77      1.48      2.81
blocktri_directschur              0.08      0.19      2.09         -         -         -         -
blocktri_cheb_lower               0.06      0.09      0.26      0.39      0.81      1.58      3.07
refmaxwell_p2v3_lower             0.11      0.19      0.40      0.78      1.47      2.39      3.87
sgs                                  -         -         -         -         -         -         -
sgs_l1                               -         -         -         -         -         -         -
sgs_damped                           -         -         -         -         -         -         -
schwarz                              -         -         -         -         -         -         -
cheb_mono                         0.07      0.13      0.39      1.52      4.69     10.50     21.49
cheb_block_diag                   0.07      0.15      0.58      2.77      9.53     22.03     47.42
cheb_block_tri                    0.06      0.09      0.29      0.43      0.81      1.50      2.98
```

BiCGStab applies the preconditioner twice per iteration, so compare its count
doubled against the GMRES count.

### dt refinement, BiCGStab

Every case is at N56x28x14; only the step size changes.

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                          cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
----------------------------------------------------------------------
jacobi                        129.4   60.1   32.6   17.0    9.9    6.0
block_diag                        -      -      -      -   35.8    9.3
blocktri_jac_jac                  -      -  153.8    9.4    8.0    7.0
blocktri_cheb                  40.2   27.3   14.7    9.9    9.0    8.0
blocktri_amg                  128.8   44.7   21.1   10.1    6.3    5.0
blocktri_hiptmair              29.2   14.8    9.1    7.1    6.0    5.0
refmaxwell_p2v3                15.1   10.7    8.8    6.8    6.0    5.0
refmaxwell_p2v3_filter         15.1   10.7    8.8    6.8    6.0    5.0
refmaxwell_p2v3_filter_off     15.1   10.7    8.8    6.8    6.0    5.0
refmaxwell_dl015               15.2   11.0    8.7    7.2    6.0    5.0
refmaxwell_dl02_mode1          79.9   28.0   14.2    7.1    6.0    5.0
refmaxwell_dl02_additive       15.3   11.0    8.7    7.2    6.0    5.0
refmaxwell_dl02_121            14.0   10.9    8.6    7.2    6.0    5.0
maxwell1_emin0                 18.9   12.9    9.1    7.1    6.0    5.0
maxwell1_sa_rs                 21.6   13.4    8.9    7.1    6.0    5.0
maxwell1_sa_rs_edge            21.4   13.5    8.9    7.1    6.0    5.0
maxwell1_sa_rs_nodal           21.6   13.4    8.9    7.1    6.0    5.0
refmaxwell_unsmoothed          17.3   10.6    8.0    7.4    6.0    5.0
blocktri_ilu0                     -   64.6   20.7   11.1    8.1    7.8
blocktri_ilu1                     -   37.9   18.1   10.6    8.1    7.7
blocktri_amg_ilu                  -      -   19.6   12.7    8.1    7.4
block_diag_ilu                    -      -      -      -      -   11.8
blocktri_chebpivot             25.7   17.8   10.5    4.8    4.0    4.0
blocktri_cheb_lumpedpivot      36.8   26.0   14.4    9.0    8.0    8.0
blocktri_cheb_pointweight      43.0   28.6   15.8    9.6    8.0    8.0
blocktri_directpivot              -      -    8.5    3.2    2.0    2.0
refmaxwell_p2v3_chebpivot      13.7    9.9    7.8    5.9    4.8    4.0
blocktri_cheb_base            271.8  108.5   56.6   27.1   15.4    9.8
blocktri_cheb_gamma05          34.4   22.9   14.0   11.9    9.8    8.5
blocktri_cheb_gamma08          37.6   25.2   14.6   10.5    9.0    8.0
blocktri_directschur            9.4    9.1    9.4    8.6    8.0    7.6
blocktri_cheb_lower            38.1   28.1   15.6    9.8    9.0    8.0
refmaxwell_p2v3_lower          16.5   12.6   10.1    8.6    6.9    5.0
sgs                               -      -      -      -   47.4    6.9
sgs_l1                            -      -      -      -   76.8   14.7
sgs_damped                        -      -      -      -   11.4    4.0
schwarz                           -      -      -      -   21.3    7.0
cheb_mono                         -   73.3   39.0   17.2      -    5.8
cheb_block_diag                   -      -      -      -   52.8   11.0
cheb_block_tri                 33.0   24.3   14.1    8.3    6.0    5.0

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                          cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
----------------------------------------------------------------------
jacobi                         4.93   2.59   1.65   1.21   0.92   0.80
block_diag                        -      -      -      -   1.64   0.90
blocktri_jac_jac                  -      -   8.74   1.28   1.21   1.16
blocktri_cheb                  2.88   2.24   1.55   1.31   1.28   1.21
blocktri_amg                  21.90   8.16   4.31   2.52   1.94   1.69
blocktri_hiptmair             12.17   6.62   4.88   4.17   3.39   2.88
refmaxwell_p2v3                4.10   3.14   2.73   2.34   2.15   2.01
refmaxwell_p2v3_filter         2.92   2.34   2.15   2.54   1.77   1.65
refmaxwell_p2v3_filter_off     3.91   3.11   2.66   2.27   2.16   1.95
refmaxwell_dl015               3.95   3.15   2.63   2.33   2.10   1.90
refmaxwell_dl02_mode1         14.10   5.57   3.31   2.11   1.94   1.77
refmaxwell_dl02_additive       3.92   3.10   2.62   2.32   2.09   1.89
refmaxwell_dl02_121            4.59   3.81   3.23   2.81   2.47   2.24
maxwell1_emin0                 8.83   6.35   4.83   4.00   3.52   3.08
maxwell1_sa_rs                 9.37   6.17   4.39   3.70   3.30   2.90
maxwell1_sa_rs_edge            9.30   6.17   4.42   3.72   3.32   2.87
maxwell1_sa_rs_nodal           9.40   6.20   4.39   3.78   3.23   2.94
refmaxwell_unsmoothed          5.70   4.03   3.20   3.07   2.71   2.42
blocktri_ilu0                     -   9.08   3.59   2.39   2.02   1.99
blocktri_ilu1                     -  20.39  12.41   9.22   8.23   8.03
blocktri_amg_ilu                  -      -   6.90   4.81   3.42   3.26
block_diag_ilu                    -      -      -      -      -   1.31
blocktri_chebpivot             2.22   1.82   1.39   1.05   1.02   1.04
blocktri_cheb_lumpedpivot      2.68   2.15   1.54   1.26   1.21   1.23
blocktri_cheb_pointweight      3.06   2.29   1.62   1.29   1.20   1.21
blocktri_directpivot              -      -  14.84  12.41  12.39  11.93
refmaxwell_p2v3_chebpivot      3.41   2.73   2.35   2.02   1.80   1.67
blocktri_cheb_base             8.47   3.81   2.30   1.49   1.13   1.00
blocktri_cheb_gamma05          2.58   2.00   1.50   1.41   1.33   1.25
blocktri_cheb_gamma08          2.79   2.09   1.52   1.32   1.26   1.21
blocktri_directschur         448.70 465.90 447.30 448.80 445.70 446.10
blocktri_cheb_lower            2.72   2.27   1.61   1.31   1.25   1.23
refmaxwell_p2v3_lower          4.18   3.48   2.98   3.30   2.33   1.93
sgs                               -      -      -      -   3.29   1.00
sgs_l1                            -      -      -      -   4.88   1.43
sgs_damped                        -      -      -      -   1.76   1.00
schwarz                           -      -      -      -   2.55   1.33
cheb_mono                         -   3.03   1.92   1.22      -   0.81
cheb_block_diag                   -      -      -      -   2.10   0.96
cheb_block_tri                 2.65   2.13   1.58   1.26   1.14   1.07
```

