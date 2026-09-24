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

A representative slice of the 35 decks, 4 ranks, BiCGStab.
Regenerate with `./sweep.py report all` and `./sweep.py report cfl`.
A dash is a case that was not run, crashed, or ran out of iterations; read its
log under `runs/bicgstab/<deck>/<case>/`.

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                           N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
-------------------------------------------------------------------------------------------------
jacobi                           31.3      48.3      67.8      84.4     102.4     114.2     129.4
blocktri_cheb                    14.5      17.6      20.3      26.2      30.7      33.0      40.2
blocktri_amg                     20.8      29.5      49.7      70.5      90.6     112.6     128.8
refmaxwell_p2v3                   8.5       9.7      11.4      12.7      13.7      14.3      15.1
refmaxwell_p2v3_chebpivot         7.4       9.3      10.3      11.7      12.1      13.1      13.7
refmaxwell_dl02_mode1             9.8      11.8      45.5      51.2      55.4      90.0      79.9
maxwell1_emin0                    7.1       7.7      12.0      14.2      16.2      17.4      18.9
blocktri_chebpivot                7.9      11.3      13.4      16.1      19.4      23.1      25.7
blocktri_cheb_lumpedpivot        16.4      20.0      22.6      26.7      30.0      34.6      36.8
blocktri_cheb_pointweight        14.8      18.2      24.1      26.3      33.4      36.4      43.0
blocktri_cheb_base               38.5      65.6     109.3     151.1     188.3     232.5     271.8
blocktri_cheb_gamma05            12.5      14.3      18.7      22.0      26.1      29.6      34.4
blocktri_directschur             10.3      10.1      10.2       9.5       9.6       9.5       9.4
blocktri_cheb_lower              14.9      19.2      23.0      28.9      32.9      41.0      38.1
refmaxwell_p2v3_lower             9.7      10.6      12.8      14.3      15.5      15.7      16.5
cheb_block_tri                   11.2      14.9      18.2      21.6      24.8      29.0      33.0

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                           N8x8x4   N16x8x4  N24x12x6  N32x16x8 N40x20x10 N48x24x12 N56x28x14
-------------------------------------------------------------------------------------------------
jacobi                           0.05      0.12      0.26      0.66      1.48      2.92      5.08
blocktri_cheb                    0.05      0.09      0.19      0.42      0.92      1.69      2.99
blocktri_amg                     0.13      0.26      1.02      3.11      6.11     12.81     22.84
refmaxwell_p2v3                  0.11      0.18      0.42      0.83      1.53      2.69      4.29
refmaxwell_p2v3_chebpivot        0.13      0.18      0.35      0.66      1.31      2.17      3.48
refmaxwell_dl02_mode1            0.12      0.14      1.01      2.07      4.11     10.37     14.55
maxwell1_emin0                   0.11      0.22      0.58      1.49      2.86      5.15      9.21
blocktri_chebpivot               0.05      0.09      0.19      0.44      0.74      1.37      2.88
blocktri_cheb_lumpedpivot        0.07      0.09      0.20      0.51      0.90      1.70      2.84
blocktri_cheb_pointweight        0.06      0.10      0.22      0.50      0.98      1.77      3.24
blocktri_cheb_base               0.07      0.16      0.38      1.15      2.38      4.96      9.25
blocktri_cheb_gamma05            0.05      0.08      0.20      0.42      0.86      1.57      2.73
blocktri_directschur             0.09      0.19      2.32     12.62     60.48    170.80    447.30
blocktri_cheb_lower              0.07      0.07      0.21      0.50      0.92      1.87      2.79
refmaxwell_p2v3_lower            0.12      0.24      0.47      0.94      1.66      2.65      4.26
cheb_block_tri                   0.05      0.09      0.20      0.50      0.89      1.60      2.62
```

CFL study, same mesh, shrinking dt:

```
Belos iterations per solve, mean, BiCGStab, 4 ranks

deck                         cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
---------------------------------------------------------------------
jacobi                       129.4   60.1   32.6   17.0    9.9    6.0
blocktri_cheb                 40.2   27.3   14.7    9.9    9.0    8.0
blocktri_amg                 128.8   44.7   21.1   10.1    6.3    5.0
refmaxwell_p2v3               15.1   10.7    8.8    6.8    6.0    5.0
refmaxwell_p2v3_chebpivot     13.7    9.9    7.8    5.9    4.8    4.0
refmaxwell_dl02_mode1         79.9   28.0   14.2    7.1    6.0    5.0
maxwell1_emin0                18.9   12.9    9.1    7.1    6.0    5.0
blocktri_chebpivot            25.7   17.8   10.5    4.8    4.0    4.0
blocktri_cheb_lumpedpivot     36.8   26.0   14.4    9.0    8.0    8.0
blocktri_cheb_pointweight     43.0   28.6   15.8    9.6    8.0    8.0
blocktri_cheb_base           271.8  108.5   56.6   27.1   15.4    9.8
blocktri_cheb_gamma05         34.4   22.9   14.0   11.9    9.8    8.5
blocktri_directschur           9.4    9.1    9.4    8.6    8.0    7.6
blocktri_cheb_lower           38.1   28.1   15.6    9.8    9.0    8.0
refmaxwell_p2v3_lower         16.5   12.6   10.1    8.6    6.9    5.0
cheb_block_tri                33.0   24.3   14.1    8.3    6.0    5.0
sgs_damped                       -      -      -      -   11.4    4.0
schwarz                          -      -      -      -   21.3    7.0

Forward solve wall time, seconds, BiCGStab, 4 ranks

deck                         cfl22   cfl8   cfl4   cfl2   cfl1 cfl0p5
---------------------------------------------------------------------
jacobi                        4.79   2.58   1.66   1.16   0.98   0.80
blocktri_cheb                 2.87   2.29   1.55   1.34   1.25   1.27
blocktri_amg                 21.78   8.12   4.36   2.52   1.94   1.76
refmaxwell_p2v3               3.94   3.17   2.68   2.29   2.11   1.95
refmaxwell_p2v3_chebpivot     3.48   2.77   2.38   1.98   1.81   1.66
refmaxwell_dl02_mode1        14.24   5.67   3.39   2.12   1.94   1.75
maxwell1_emin0                8.92   6.34   4.86   4.11   3.65   3.20
blocktri_chebpivot            2.19   1.81   1.38   1.08   1.01   1.06
blocktri_cheb_lumpedpivot     2.66   2.12   1.54   1.30   1.20   1.23
blocktri_cheb_pointweight     3.13   2.38   1.60   1.32   1.20   1.24
blocktri_cheb_base            9.24   3.85   2.39   1.61   1.28   1.09
blocktri_cheb_gamma05         2.61   1.99   1.54   1.45   1.34   1.25
blocktri_directschur        448.70 465.90 447.30 448.80 445.70 446.10
blocktri_cheb_lower           2.93   2.25   1.67   1.34   1.30   1.25
refmaxwell_p2v3_lower         4.22   3.44   3.00   2.71   2.35   1.97
cheb_block_tri                2.59   2.21   1.62   1.32   1.14   1.11
sgs_damped                       -      -      -      -   1.72   1.04
schwarz                          -      -      -      -   2.62   1.30
```
