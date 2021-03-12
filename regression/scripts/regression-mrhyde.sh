#!/bin/bash
############# Overview ##################################################
#                                                                       #
# Full Trilino/Milo checkout/compiling and run regessions tests:        #
#                                                                       #
# make a directory "temp-mrhyde"                                          #
# copy mrhyde/regression/scripts/regression-mrhyde.sh into temp-mrhyde        #
# execute regression-mrhyde.sh                                            #
#                                                                       #
# Running all regession tests:                                          #
#                                                                       #
# execute ./runtest.py from mrhyde/regression directory                   #
#                                                                       #
# Running specfic regressions tests:                                    #
#                                                                       #
# cd to regeression/directory                                           #
# execute mile.tst within directory                                     #
#                                                                       #
# bvbw 10/3/2016
############# Variable Definitions and Preliminaries ####################

RUN="T"
NOW=$(date +"%m_%d_%Y")
INSTALL_DIRECTORY="/Users/tmwilde/Desktop/Software/regression-mrhyde"
INSTALL_DIRECTORY_DATE="/Users/tmwilde/Desktop/Software/regression-mrhyde/${NOW}"
ARG=${BASH_ARGV[*]}

if [ ${RUN} == "T" ]
then
#    rm -rf mrhyde Trilinos
    cd ${INSTALL_DIRECTORY}
    rm -rf mrhyde 
    module purge
    module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
    mkdir "/scratch/bartv/regression-mrhyde/${NOW}"
fi

############# MILO checkout #############################################

if [ ${RUN} == "T" ]
then
#    svn co --username bartv --password $PASS svn+ssh://software.sandia.gov/svn/private/mrhyde >& mrhyde-checkout.out
     svn co svn+ssh://software.sandia.gov/svn/private/mrhyde >& ${INSTALL_DIRECTORY}/mrhyde-checkout.out
#    cd ${INSTALL_DIRECTORY}/mrhyde
#    svn up >& mrhyde-up.out
fi

############# Trilinos Debug ###########################################

if [ ${RUN} == "F" ]
then
    git clone https://github.com/trilinos/Trilinos.git
    cd ${INSTALL_DIRECTORY}/Trilinos
    mkdir ${INSTALL_DIRECTORY}/Trilinos/build-debug
    cd ${INSTALL_DIRECTORY}/Trilinos/build-debug
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/scripts/configure-trilinos-debug .
    ./configure-trilinos-debug >& configure-trilinos.out
    make -j8 install >& make-trilinos.out
    cp ${INSTALL_DIRECTORY}/Trilinos/build-debug/configure-trilinos.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/Trilinos/build-debug/make-trilinos.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Debug ###############################################

if [ ${RUN} == "F" ]
then
    cd ${INSTALL_DIRECTORY}/mrhyde
    mkdir ${INSTALL_DIRECTORY}/mrhyde/build-debug
    cd ${INSTALL_DIRECTORY}/mrhyde/build-debug
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/scripts/configure-mrhyde-debug .
    ./configure-mrhyde-debug >& configure-mrhyde-debug.out
    make -j2 >& make-mrhyde-debug.out
    cp ${INSTALL_DIRECTORY}/mrhyde/build-debug/configure-mrhyde-debug.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/mrhyde/build-debug/make-mrhyde-debug.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Debug Test #################################################

if [ ${RUN} == "F" ]
then
    cd ${INSTALL_DIRECTORY}/mrhyde/regression
    ln -s ${INSTALL_DIRECTORY}/mrhyde/build-debug/src-ms/mrhyde-ms ./mrhyde-ms
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/scripts/runtests.py .
    ./runtests.py >& runtests-debug.out
    tr -d '\b\r' < runtests.out > runtests-debug.out
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/runtests-debug.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Optimized ############################################

if [ ${RUN} == "T" ]
then
    cd ${INSTALL_DIRECTORY}/mrhyde
    mkdir ${INSTALL_DIRECTORY}/mrhyde/build-opt
    cd ${INSTALL_DIRECTORY}/mrhyde/build-opt
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/scripts/configure-mrhyde-opt .
    ./configure-mrhyde-opt >& configure-mrhyde-opt.out
    make -j2 >& make-mrhyde-opt.out
    cp ${INSTALL_DIRECTORY}/mrhyde/build-opt/configure-mrhyde-opt.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/mrhyde/build-opt/make-mrhyde-opt.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Opt Test #################################################

if [ ${RUN} == "T" ]
then
    cd ${INSTALL_DIRECTORY}/mrhyde/regression
    rm mrhyde-ms
    ln -s ${INSTALL_DIRECTORY}/mrhyde/build-opt/src-ms/mrhyde-ms ./mrhyde-ms
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/scripts/runtests.py .
    ./runtests.py >& runtests.out
    tr -d '\b\r' < runtests.out > runtests-opt.out
    cp ${INSTALL_DIRECTORY}/mrhyde/regression/runtests-opt.out ${INSTALL_DIRECTORY_DATE} 
fi
