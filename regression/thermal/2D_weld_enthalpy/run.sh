#!/bin/bash
#module purge
#module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
#module list >& env.out
. ~/.bashrc
mpiexec -n 4 ../../milo >& milo.log
os=$(uname -s 2>/dev/null | tr [:lower:] [:upper:])
if [ $os == "LINUX" ]; then
  sed -i 5,22d milo.log
elif [ $os == "DARWIN" ]; then
  sed -i '' 5,22d milo.log
fi
rm milo_test_support.pyc
exit
