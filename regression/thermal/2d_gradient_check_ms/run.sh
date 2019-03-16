#!/bin/bash
#module purge
#module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
#module list >& env.out
. ~/.bashrc
mpiexec -n 4 ../../milo >& milo.log
rm final_params.dat milo_test_support.pyc param_stash.dat ROL_out.txt 
exit
