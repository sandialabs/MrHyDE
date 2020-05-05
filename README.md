# Multi-resolution Hybridized Differential Equations (MrHyDE)


## Configuring MrHyDE
Configuring MrHyDE is done in CMake. 
There are only a handful of necessary options, and there are plenty of example scripts in the `scripts` directory.
The required options, as shown in the file `scripts/configure`, are 
```
-D Trilinos_SRC_DIR=${TRILINOS_HOME}
-D Trilinos_INSTALL_DIR=${TRILINOS_INSTALL}
```
Setting these to the appropriate path for a compatible Trilinos installation is all that's necessary. 
To learn more about setting up a compatible Trilinos configuration, see some of the other configuration scripts.

Other options for MrHyDE are booleans for the Kokkos devices. 
If no options are specified at configuration time, the default is to have all work and memory on the host device. 
The available configuration options within MrHyDE for the Kokkos devices are in the table below. 

Option                | Host | Subgrid | Assembly
----------------------|------|---------|---------
`MrHyDE_ALL_CPU`      | CPU  | CPU     | CPU
`MrHyDE_ASSEMBLY_GPU` | CPU  | CPU     | GPU
`MrHyDE_SUBGRID_GPU`  | CPU  | GPU     | GPU

If more than one flag is passed at the time MrHyDE is configured, then it will default to the highest option in the table. 
If a device is set to GPU, its memory will reside in the CUDAUVM space. 

