#!/bin/bash
############# Overview ##################################################
#                                                                       #
# Full Trilino/Milo checkout/compiling and run regessions tests:        #
#                                                                       #
# make a directory "temp-milo"                                          #
# copy milo/regression/scripts/regression-milo.sh into temp-milo        #
# execute regression-milo.sh                                            #
#                                                                       #
# Running all regession tests:                                          #
#                                                                       #
# execute ./runtest.py from milo/regression directory                   #
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
INSTALL_DIRECTORY="/Users/tmwilde/Desktop/Software/regression-milo"
INSTALL_DIRECTORY_DATE="/Users/tmwilde/Desktop/Software/regression-milo/${NOW}"
ARG=${BASH_ARGV[*]}

if [ ${RUN} == "T" ]
then
#    rm -rf milo Trilinos
    cd ${INSTALL_DIRECTORY}
    rm -rf milo 
    module purge
    module load sierra-devel/gcc-4.9.3-openmpi-1.8.8
    mkdir "/scratch/bartv/regression-milo/${NOW}"
fi

############# MILO checkout #############################################

if [ ${RUN} == "T" ]
then
#    svn co --username bartv --password $PASS svn+ssh://software.sandia.gov/svn/private/milo >& milo-checkout.out
     svn co svn+ssh://software.sandia.gov/svn/private/milo >& ${INSTALL_DIRECTORY}/milo-checkout.out
#    cd ${INSTALL_DIRECTORY}/milo
#    svn up >& milo-up.out
fi

############# Trilinos Debug ###########################################

if [ ${RUN} == "F" ]
then
    git clone https://github.com/trilinos/Trilinos.git
    cd ${INSTALL_DIRECTORY}/Trilinos
    mkdir ${INSTALL_DIRECTORY}/Trilinos/build-debug
    cd ${INSTALL_DIRECTORY}/Trilinos/build-debug
    cp ${INSTALL_DIRECTORY}/milo/regression/scripts/configure-trilinos-debug .
    ./configure-trilinos-debug >& configure-trilinos.out
    make -j8 install >& make-trilinos.out
    cp ${INSTALL_DIRECTORY}/Trilinos/build-debug/configure-trilinos.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/Trilinos/build-debug/make-trilinos.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Debug ###############################################

if [ ${RUN} == "F" ]
then
    cd ${INSTALL_DIRECTORY}/milo
    mkdir ${INSTALL_DIRECTORY}/milo/build-debug
    cd ${INSTALL_DIRECTORY}/milo/build-debug
    cp ${INSTALL_DIRECTORY}/milo/regression/scripts/configure-milo-debug .
    ./configure-milo-debug >& configure-milo-debug.out
    make -j2 >& make-milo-debug.out
    cp ${INSTALL_DIRECTORY}/milo/build-debug/configure-milo-debug.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/milo/build-debug/make-milo-debug.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Debug Test #################################################

if [ ${RUN} == "F" ]
then
    cd ${INSTALL_DIRECTORY}/milo/regression
    ln -s ${INSTALL_DIRECTORY}/milo/build-debug/src-ms/milo-ms ./milo-ms
    cp ${INSTALL_DIRECTORY}/milo/regression/scripts/runtests.py .
    ./runtests.py >& runtests-debug.out
    tr -d '\b\r' < runtests.out > runtests-debug.out
    cp ${INSTALL_DIRECTORY}/milo/regression/runtests-debug.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Optimized ############################################

if [ ${RUN} == "T" ]
then
    cd ${INSTALL_DIRECTORY}/milo
    mkdir ${INSTALL_DIRECTORY}/milo/build-opt
    cd ${INSTALL_DIRECTORY}/milo/build-opt
    cp ${INSTALL_DIRECTORY}/milo/regression/scripts/configure-milo-opt .
    ./configure-milo-opt >& configure-milo-opt.out
    make -j2 >& make-milo-opt.out
    cp ${INSTALL_DIRECTORY}/milo/build-opt/configure-milo-opt.out ${INSTALL_DIRECTORY_DATE} 
    cp ${INSTALL_DIRECTORY}/milo/build-opt/make-milo-opt.out ${INSTALL_DIRECTORY_DATE} 
fi

############# MILO Opt Test #################################################

if [ ${RUN} == "T" ]
then
    cd ${INSTALL_DIRECTORY}/milo/regression
    rm milo-ms
    ln -s ${INSTALL_DIRECTORY}/milo/build-opt/src-ms/milo-ms ./milo-ms
    cp ${INSTALL_DIRECTORY}/milo/regression/scripts/runtests.py .
    ./runtests.py >& runtests.out
    tr -d '\b\r' < runtests.out > runtests-opt.out
    cp ${INSTALL_DIRECTORY}/milo/regression/runtests-opt.out ${INSTALL_DIRECTORY_DATE} 
fi
