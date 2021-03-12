#!/usr/bin/env bash
#
# This script runs the actual tests of MILO from Jenkins.
# It expects the right Jenkins environment variables to be
# set properly.
#
module load sems-env

module load sems-python/2.7.9
module load sems-numpy/1.9.1/base

module load sems-gcc/4.9.3
module load sems-openmpi/1.10.1
module load sems-boost/1.63.0/base
module load sems-gdb/7.9.1
module load sems-hdf5/1.8.12/parallel
module load sems-netcdf/4.4.1/exo_parallel

module load sems-cmake/3.10.3

: ${WORKSPACE?"Error: [test-driver] Expected environment variable WORKSPACE does not exist"}
: ${PARAM_REBUILD_TRILINOS?"Error: [test-driver] Expected environment variable PARAM_REBUILD_TRILINOS does not exist"}
: ${PARAM_REBUILD_MILO?"Error: [test-driver] Expected environment variable PARAM_REBUILD_MILO does not exist"}
: ${PARAM_NUM_CORES?"Error: [test-driver] Expected environment variable PARAM_NUM_CORES does not exist"}

# Reset dir to known path.
cd ${WORKSPACE:?}
echo -e "pwd: `pwd`"

# Set build directories
trilinos_build_opt_path=${WORKSPACE}/trilinos/build-opt
mrhyde_build_opt_path=${WORKSPACE}/mrhyde/build-opt


# Wipe out build dir(s) if parameters say so...
if [[ $PARAM_REBUILD_TRILINOS == "true" ]]; then
      echo "PARAM_REBUILD_TRILINOS is set, clearing build dirs"
      rm -rf ${trilinos_build_opt_path}
      rm -rf ${mrhyde_build_opt_path}
fi
if [[ $PARAM_REBUILD_MILO == "true" ]]; then
      echo "PARAM_REBUILD_MILO is set, clearing mrhyde build dir"
      rm -rf ${mrhyde_build_opt_path}
fi



#----------------------------------------------------------------
#
# Configure & Build Trilinos
#
#----------------------------------------------------------------
echo "========================================"
echo "= Prepare Trilinos"
echo "========================================"
#if [ -e ${trilinos_build_opt_path} ]; then
#    rm -rf ${trilinos_build_opt_path}
#fi
if [ ! -e ${trilinos_build_opt_path} ]; then
    mkdir -p ${trilinos_build_opt_path}
fi

cp ${WORKSPACE}/mrhyde/regression/scripts/jenkins/config-trilinos.sh ${trilinos_build_opt_path}/.

cd ${WORKSPACE:?}
echo -e "pwd: `pwd`"

echo "========================================"
echo "= Configure Trilinos"
echo "========================================"
cd ${trilinos_build_opt_path:?}
echo -e "pwd: `pwd`"

./config-trilinos.sh
err=$?
if [ ! $err -eq 0 ]; then
    exit $err
fi

cd -
echo -e "pwd: `pwd`"
echo "---------------------------"
echo "- Done Configure Trilinos -"
echo "---------------------------"


echo "========================================"
echo "= Build Trilinos"
echo "========================================"
cd ${trilinos_build_opt_path}
echo -e "pwd: `pwd`"

make -j ${PARAM_NUM_CORES}
err=$?
if [ ! $err -eq 0 ]; then
    exit $err
fi
cd -
echo -e "pwd: `pwd`"
echo "-----------------------"
echo "- Done Build Trilinos -"
echo "-----------------------"


echo "========================================"
echo "= Install Trilinos"
echo "========================================"
cd ${trilinos_build_opt_path}
echo -e "pwd: `pwd`"
make install
err=$?
if [ ! $err -eq 0 ]; then
    exit $err
fi
cd -
echo -e "pwd: `pwd`"
echo "-------------------------"
echo "- Done Install Trilinos -"
echo "-------------------------"



#----------------------------------------------------------------
#
# Configure & Build Milo
#
#----------------------------------------------------------------
echo "========================================"
echo "= Prepare Milo"
echo "========================================"
#if [ -e ${mrhyde_build_opt_path} ]; then
#    rm -rf ${mrhyde_build_opt_path}
#fi
mkdir -p ${mrhyde_build_opt_path}

cp ${WORKSPACE}/mrhyde/regression/scripts/jenkins/config-mrhyde.sh ${mrhyde_build_opt_path}


echo "========================================"
echo "= Configure Milo"
echo "========================================"
cd ${mrhyde_build_opt_path}
echo -e "pwd: `pwd`"
./config-mrhyde.sh
err=$?
if [ ! $err -eq 0 ]; then
    exit $err
fi
cd -
echo -e "pwd: `pwd`"
echo "-----------------------"
echo "- Done Configure Milo -"
echo "-----------------------"


echo "========================================"
echo "= Build Milo"
echo "========================================"
cd ${mrhyde_build_opt_path}
echo -e "pwd: `pwd`"
make -j ${PARAM_NUM_CORES}
err=$?
if [ ! $err -eq 0 ]; then
    exit $err
fi
cd -
echo -e "pwd: `pwd`"
echo "-------------------"
echo "- Done Build Milo -"
echo "-------------------"


echo "========================================"
echo "= Test Milo"
echo "========================================"
# Expected mrhyde executable and path location
mrhyde_exe=mrhyde
mrhyde_exe_path=${WORKSPACE}/mrhyde/build-opt/src/

regression_path=${WORKSPACE}/mrhyde/regression

if [ -d ${WORKSPACE}/TESTING ]; then
    rm -rf ${WORKSPACE}/TESTING
fi
mkdir -p ${WORKSPACE}/TESTING

cd ${regression_path:?}
echo -e "pwd: `pwd`"

# Reset the runtests.py link
# WARNING: The symlink thing may be no longer needed, but 
#          there are two runtest.py files that are virtually
#          identical in regression and regression/scripts
#if [ -L runtests.py ]; then
#    rm runtests.py
#fi
#ln -s ${regression_path}/scripts/runtests.py runtests.py

# Verify that the mrhyde executable is there.
if [ ! -e ${mrhyde_exe_path:?}/${mrhyde_exe:?} ]; then
    echo -e "ERROR: MILO executable not found."
    echo -e "       Expected: ${mrhyde_exe_path:?}/${mrhyde_exe:?}"
    exit 16
fi

# Reset the symlink to the mrhyde executable 
if [ -L ${mrhyde_exe:?} ]; then
    rm ${mrhyde_exe:?}
fi
ln -s ${mrhyde_exe_path:?}/${mrhyde_exe:?}

# Run the mrhyde tests
set -x
./runtests.py \
      -s \
      -p Results \
      2>&1 | tee ${WORKSPACE}/TESTING/runtests.out
# -d ${regression_path} 
err=$?
set +x

# copy results to the right place(s).
mv TEST-Results.xml ${WORKSPACE}/TESTING/.

tr -d '\b\r' < ${WORKSPACE}/TESTING/runtests.out > ${WORKSPACE}/TESTING/runtests-opt.out
if [ ! $err -eq 0 ]; then
    exit $err
fi

# Reset back to workspace root
cd ${WORKSPACE:?}
echo -e "pwd: `pwd`"

echo "-------------------"
echo "- Done Test Milo  -"
echo "-------------------"

echo "========================================"
echo "= Print Results"
echo "========================================"
#head -n 5 ${WORKSPACE}/TESTING/runtests.out
#awk '/[0-9]+\/[0-9]+/{print $1 "," $2 ",time=" $3 ",np=" $5 "," $6}' ${WORKSPACE}/TESTING/runtests.out | column -t -s","
#tail -n 5 ${WORKSPACE}/TESTING/runtests.out
cat ${WORKSPACE}/TESTING/runtests.out
echo "----------------------"
echo "- Done Print Results -"
echo "----------------------"

exit $err


