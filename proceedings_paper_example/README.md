# Sandia_Proceedings_Paper_Examples
This folder contains MrHyDE input files for the examples used in the Numerical Results section of the Sandia National Lab Proceedings Paper titled "Numerical Studies of Optimal Control Problems Governed by the Time-Dependent Maxwell's Equations" authored by Yaw Owusu-Agyemang, Denis Ridzal, Tim Wildley, Rohit Khandelwal and Harbir Antil


## To run
1. Make sure Trilinos and MrHyDE are properly installed following the respective documentations.
2. Create a soft link to the `mrhyde` executable in the appropriate example folder you want to run.
3. Run the forward problem with `./mrhyde input_forward.yaml` for a serial run or `mpirun -np <NUM_PROCS> mrhyde input_forward.yaml` for a parallel run (NB: mesh decomposition must agree with number of processors used).
4. Run the optimal control problem with `./mrhyde` or `mpirun -np <NUM_PROCS> mrhyde`.