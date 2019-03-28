#!/bin/bash
#module purge
#module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
#module list >& env.out
. ~/.bashrc

mpiexec -n 4 ../../milo-ms >& milo.log
exit
