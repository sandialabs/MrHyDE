# Examples for the paper Source Cloaking for Maxwell’s Equations: Well-Posedness, Adjoint Analysis, and Fully Discrete Convergence

This folder contains MrHyDE input files for the examples used in the Numerical Results section of the  Paper titled "Source Cloaking for Maxwell’s Equations: Well-Posedness, Adjoint Analysis, and Fully Discrete Convergence" authored by Harbir Antil, Denis Ridzal, Jimmie Adriazola, Rohit Khandelwal and Yaw Owusu-Agyemang.


## To run
1. Make sure Trilinos and MrHyDE are properly installed following the respective documentations.
2. Create a soft link to the `mrhyde` executable in the appropriate example folder you want to run.
3. Run the forward problem with `./mrhyde input_forward.yaml` (serial run) or `mpirun -np <NUM_PROCS> mrhyde input_forward.yaml` (parallel run).
4. Run the optimal control problem with `./mrhyde` or `mpirun -np <NUM_PROCS> mrhyde`.

## Preprocessing
- Due to the size of the 2D and 3D mesh files, we've only included the .jou files to generate the mesh on Cubit. Update the last line with the location on your computer and run the jou file to generate the mesh.
- For compliance reasons, we run the examples on a microscale, so you need to rescale the generated meshes to the 1e-6 scale to reproduce our results. You can do this rescaling with the `grepos` library in Trilinos.

For example for the 2D example, run
- `grepos thunderbird-2.5d.exo thunderbird-2.5d-scaled.exo` 

Then in grepos run 
`scale all 1e-6`
`exit`

For any questions please contact any of the authors on this paper.