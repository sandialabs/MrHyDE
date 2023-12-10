# Multi-resolution Hybridized Differential Equations (MrHyDE)

## Configuring MrHyDE
Configuring MrHyDE is done in CMake. 
There are only a handful of necessary options, and there are plenty of example scripts in `scripts/configure-MrHyDE/`.
The required options, as shown in the file `scripts/configure-MrHyDE/configure-MrHyDE-vaporwave-serial`, are 
```
-D Trilinos_SRC_DIR:PATH=${TRILINOS_SOURCE}
-D Trilinos_INSTALL_DIR:PATH=${TRILINOS_INSTALL}
```
Setting these values to an appropriate path for a compatible Trilinos installation is the only necessary step. 
To learn more about setting up a compatible Trilinos configuration, see some of the other configuration scripts.

### Automatic Differentiation Flags
Some other configuration flags which are valuable for experienced MrHyDE developers relate to the automatic differentiation (AD) features.
```
-D MrHyDE_MAX_DERIVS:INT=64
-D MrHyDE_DISABLE_AD:BOOL=FALSE
```
Lowering the maximum number of derivatives can increase the performance of MrHyDE for certain applications, but will result in a segmentation fault if lowered too far. Additionally, the automatic differentiation may be disabled altogether using the second option.

### Kokkos Device Flags
If no options are specified at configuration time, the default is to have all work and memory on the host device. 
Some available configuration options within MrHyDE for the Kokkos devices are in the table below. 

Option                   | Host | Assembly |Solver
-------------------------|------|----------|------
`MrHyDE_ALL_CPU`         | CPU  | CPU      | CPU
`MrHyDE_ALL_OPENMP`      | OMP  | OMP      | OMP
`MrHyDE_ALL_GPU`         | CPU  | GPU      | GPU
`MrHyDE_ASSEMBLY_GPU`    | CPU  | GPU      | CPU
`MrHyDE_ASSEMBLY_OPENMP` | CPU  | OMP      | CPU

If more than one flag is passed at the time MrHyDE is configured, then it will default to the highest option in the table. 
There are additional options that may be invoked based on which configuration in the table is chosen. 
For example, the `MrHyDE_SOLVER_OPENMP` option is available only if the `MrHyDE_ASSEMBLY_OPENMP` option has already been specified, and the `MrHyDE_SOLVER_GPU` option is available only if the `MrHyDE_ASSEMBLY_GPU` option has been specified.

### Other Flags
Some boolean configuration options are presented below for ease of reference. This list may be subject to change:
 - `MrHyDE_ALL_SERIAL`, `MrHyDE_ALL_CPU`
 - `MrHyDE_ALL_CUDA`, `MrHyDE_ALL_GPU`
 - `MrHyDE_SINGLE_PRECISION`
 - `MrHyDE_VECTOR_SIZE`
 - `MrHyDE_BUILD_DOXYGEN`

## Regression Testing
Once MrHyDE is successfully built, the build directory should contain a `src` subdirectory with the executable `mrhyde`.
To run the regression testsuite, simply place the `mrhyde` executable (or a symbolic link to it) in the `regression` directory and invoke `./runtests.py`.
This will recurse over the entire directory to run the regression testsuite, and it usually takes less than two minutes to complete.
It will display test results in real-time while running, and it will display a summary when completed.
