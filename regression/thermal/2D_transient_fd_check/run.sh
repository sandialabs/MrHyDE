#!/bin/bash
#module purge
#module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
#module list >& env.out
. ~/.bashrc
mpiexec -n 1 ../../milo >& milo.log
os=$(uname -s 2>/dev/null | tr [:lower:] [:upper:])
if [ $os == "LINUX" ]; then
  sed -i 6,15d milo.log
elif [ $os == "DARWIN" ]; then
  sed -i '' 6,15d milo.log
fi
rm final_params.dat milo_test_support.pyc param_stash.dat ROL_out.txt 
exit
