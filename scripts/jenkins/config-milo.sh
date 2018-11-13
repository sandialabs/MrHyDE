#!/usr/bin/env bash

# verify expected environment variables exist
: ${WORKSPACE?"Error: [config-milo] Expected environment variable WORKSPACE does not exist"}
: ${WORKSPACE:?"Error: [config-milo] Expected environment variable WORKSPACE is empty"}

: ${SEMS_CMAKE_ROOT?"Error: [config-milo] Expected environment varaible SEMS_CMAKE_ROOT does not exist"}
: ${SEMS_CMAKE_ROOT:?"Error: [config-milo] Expected environment varaible SEMS_CMAKE_ROOT is empty"}


EXTRA_ARGS=$@

# wipe out cmake cache file
# rm CMakeCache.txt

TRILINOS_HOME="${WORKSPACE}/trilinos"
TRILINOS_INSTALL="${WORKSPACE}/trilinos-build/SIMOPT-MILO-OPT"
MILO_HOME="${WORKSPACE}/milo"
MILO_INSTALL='${WORKSPACE}/milo-install'

CMAKE=${SEMS_CMAKE_ROOT}/bin/cmake

${CMAKE} \
    -D Trilinos_SRC_DIR=${TRILINOS_HOME} \
    -D Trilinos_INSTALL_DIR=${TRILINOS_INSTALL} \
    -D CMAKE_VERBOSE_CONFIGURE:BOOL=OFF \
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -D CMAKE_INSTALL_PREFIX:PATH=${MILO_INSTALL} \
    ${EXTRA-ARGS} \
    ${MILO_HOME}

# Check error code from CMake & propagate if CMake failed.
err=$?
if [ ! $err -eq 0 ]; then
    echo "CMake Failed"
    exit $err
fi

