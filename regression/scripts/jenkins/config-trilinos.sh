#!/usr/bin/env bash

# verify expected environment variables exist
: ${WORKSPACE?"Error: [config-mrhyde] Expected environment variable WORKSPACE does not exist"}
: ${WORKSPACE:?"Error: [config-mrhyde] Expected environment variable WORKSPACE is empty"}
: ${SEMS_CMAKE_ROOT?"Error: [config-mrhyde] Expected environment varaible SEMS_CMAKE_ROOT does not exist"}
: ${SEMS_CMAKE_ROOT:?"Error: [config-mrhyde] Expected environment varaible SEMS_CMAKE_ROOT is empty"}
: ${SEMS_BOOST_INCLUDE_PATH?"Error: [config-mrhyde] Expected environment varaible SEMS_BOOST_INCLUDE_PATH does not exist"}
: ${SEMS_BOOST_INCLUDE_PATH:?"Error: [config-mrhyde] Expected environment varaible SEMS_BOOST_INCLUDE_PATH is empty"}
: ${SEMS_BOOST_LIBRARY_PATH?"Error: [config-mrhyde] Expected environment varaible SEMS_BOOST_LIBRARY_PATH does not exist"}
: ${SEMS_BOOST_LIBRARY_PATH:?"Error: [config-mrhyde] Expected environment varaible SEMS_BOOST_LIBRARY_PATH is empty"}
: ${SEMS_NETCDF_INCLUDE_PATH?"Error: [config-mrhyde] Expected environment varaible SEMS_NETCDF_INCLUDE_PATH does not exist"}
: ${SEMS_NETCDF_INCLUDE_PATH:?"Error: [config-mrhyde] Expected environment varaible SEMS_NETCDF_INCLUDE_PATH is empty"}
: ${SEMS_NETCDF_LIBRARY_PATH?"Error: [config-mrhyde] Expected environment varaible SEMS_NETCDF_LIBRARY_PATH does not exist"}
: ${SEMS_NETCDF_LIBRARY_PATH:?"Error: [config-mrhyde] Expected environment varaible SEMS_NETCDF_LIBRARY_PATH is empty"}

EXTRA_ARGS=$@

TRILINOS_HOME="${WORKSPACE}/trilinos"
INSTALL_DIR="${WORKSPACE}/trilinos-build/SIMOPT-MILO-OPT"

CMAKE=${SEMS_CMAKE_ROOT}/bin/cmake

echo "TRILINOS_HOME: ${TRILINOS_HOME}"

# Clear out CMake files to cause a rebuild (mostly)
# rm -rf CMakeCache.txt
# rm -rf CMakeFiles

# TRILINOS_HOME='/scratch/bartv/Trilinos-github/Trilinos-simopt'
# INSTALL_DIR='/scratch/bartv/software/Trilinos-github/SIMOPT-MILO-OPT'
# BOOST_INCLUDE_DIR='/scratch/bartv/software/boost_1_64_0/install/include'
# BOOST_LIB_DIR='/scratch/bartv/software/boost_1_64_0/install/lib'
# NETCDF_LIB_DIR='/scratch/bartv/software/netcdf-4.2/install/lib'
# NETCDF_INCLUDE_DIR='/scratch/bartv/software/netcdf-4.2/install/include'

${CMAKE} \
    -D CMAKE_BUILD_TYPE:STRING=NONE \
    -D TPL_ENABLE_MPI:BOOL=ON \
    -D CMAKE_CXX_FLAGS:STRING="-O3 -ansi -pedantic -ftrapv -Wall -Wno-long-long -Wno-strict-aliasing -DBOOST_NO_HASH" \
    -D CMAKE_C_FLAGS:STRING="-O3" \
    -D CMAKE_Fortran_FLAGS:STRING="-O3" \
    -D Trilinos_ENABLE_CHECKED_STL:BOOL=OFF \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
    -D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON \
    -D Trilinos_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON \
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_Teko:BOOL=ON \
    -D Trilinos_ENABLE_Belos:BOOL=ON \
    -D Trilinos_ENABLE_ROL:BOOL=ON \
    -D ROL_ENABLE_EXAMPLES:BOOL=ON \
    -D ROL_ENABLE_TEST:BOOL=ON \
    -D Trilinos_ENABLE_AztecOO:BOOL=ON \
    -D Trilinos_ENABLE_Ifpack:BOOL=ON \
    -D Trilinos_ENABLE_Panzer:BOOL=ON \
    -D Trilinos_ENABLE_Intrepid:BOOL=ON \
    -D Trilinos_ENABLE_Intrepid2:BOOL=ON \
    -D Trilinos_ENABLE_Shards:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
    -D Trilinos_ENABLE_ML:BOOL=ON \
    -D Trilinos_ENABLE_Zoltan:BOOL=ON \
    -D Trilinos_ENABLE_FEI:BOOL=ON \
    -D Trilinos_ENABLE_Amesos:BOOL=ON \
    -D Trilinos_ENABLE_Amesos2:BOOL=ON \
    -D Trilinos_ENABLE_STKClassic:BOOL=OFF \
    -D Trilinos_ENABLE_STKMesh:BOOL=ON \
    -D Trilinos_ENABLE_STKIO:BOOL=ON \
    -D Trilinos_ENABLE_STKUtil:BOOL=ON \
    -D Trilinos_ENABLE_STKSearch:BOOL=ON \
    -D Trilinos_ENABLE_STKTopology:BOOL=ON \
    -D Trilinos_ENABLE_STKTransfer:BOOL=ON \
    -D Trilinos_ENABLE_SEACAS:BOOL=ON \
    -D Trilinos_ENABLE_Belos:BOOL=ON \
    -D Trilinos_ENABLE_MueLu:BOOL=ON \
    -D TPL_ENABLE_Matio=OFF \
    -D STK_ENABLE_TESTS:BOOL=OFF \
    -D Panzer_ENABLE_TESTS:BOOL=OFF \
    -D Panzer_ENABLE_EXAMPLES:BOOL=OFF \
    -D Panzer_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
    -D EpetraExt_ENABLE_HDF5:BOOL=OFF \
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -D CMAKE_SKIP_RULE_DEPENDENCY=ON \
    -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} \
    -D TPL_ENABLE_Boost:BOOL=ON \
    -D TPL_Boost_INCLUDE_DIRS:FILEPATH=${SEMS_BOOST_INCLUDE_PATH} \
    -D TPL_ENABLE_BoostLib:BOOL=ON \
    -D BoostLib_INCLUDE_DIRS:FILEPATH=${SEMS_BOOST_INCLUDE_PATH} \
    -D BoostLib_LIBRARY_DIRS:FILEPATH=${SEMS_BOOST_LIBRARY_PATH} \
    -D TPL_ENABLE_Netcdf:BOOL=ON \
    -D Netcdf_INCLUDE_DIRS:FILEPATH=${SEMS_NETCDF_INCLUDE_PATH} \
    -D Netcdf_LIBRARY_DIRS:FILEPATH=${SEMS_NETCDF_LIBRARY_PATH} \
    -D SEACASExodus_ENABLE_MPI:BOOL=OFF \
    -D TPL_ENABLE_GLM=OFF \
    -D TPL_ENABLE_X11=OFF \
    ${EXTRA_ARGS} \
    ${TRILINOS_HOME}
 
# Check error code from CMake & propagate if CMake failed.
err=$?
if [ ! $err -eq 0 ]; then
    echo "CMake Failed"
    exit $err
fi
