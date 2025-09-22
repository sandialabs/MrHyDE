## About
This folder contains the input files for solving 2D Time-Dependent Maxwell's optimization problem modelled on a 3D-rectangular domain with periodic boundary conditions in the z axe\is. It contains input files for both the forward problem and the optimal control problem (minimizing the electromagnetic energy on some observation blocks).

## Folder structure
The structure of each example folder is as follows:

- input_forward.yaml
- input_functions_forward.yaml
- /mesh
- input_functions.yaml
- input_params.yaml
- input_rol2.yaml
- input.yaml

## Files description
Below is a description of what each file does:

- `input_forward.yaml` contains options for the forward problem. It specifies the Physics module, Discretization settings, Analysis options and Postprocessing options. It also imports the mesh settings from `input_mesh.yaml` and Functions from `input_functions_forward.yaml`.

- `input_functions_forward.yaml` contains all function definitions for each block for the forward problem.

- `/mesh` is a folder containing the mesh object in exo format.

- `input_functions.yaml` contains all function definitions for each block.

- `input_params.yaml` contains the control parameter definitions to be used in the optimization.

- `input_rol2.yaml` contains all the rol settings to be used for the optimization.

- `input.yaml` contains options for the full optimal control problem (forward and backward). It specifies the Physics module, Discretization settings, Analysis options and Postprocessing options. It also imports the mesh settings from input_mesh.yaml, parameters from input_params.yaml, R settings from `input_rol2.yaml` and Functions from `input_functions.yaml`.