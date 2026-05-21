FROM --platform=linux/amd64 silkeh/clang:12
WORKDIR /git-repos

# Make sure basic necessities are installed
RUN apt-get update && apt-get install -y wget zip unzip xz-utils bzip2 zlib1g-dev libblas-dev liblapack-dev ssh

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-x86_64.tar.gz && \
    tar xf cmake-3.23.1-linux-x86_64.tar.gz && \
    rm cmake-3.23.1-linux-x86_64.tar.gz && \
    mv cmake-3.23.1-linux-x86_64 /usr/local/cmake && \
    ln -s /usr/local/cmake/bin/cmake /usr/local/bin/cmake

# Install Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip && \
    unzip ninja-linux.zip && rm ninja-linux.zip && \
    mv ninja /usr/local/bin

# Configure and install OpenMPI
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz && \
    tar xf openmpi-4.1.2.tar.gz && rm openmpi-4.1.2.tar.gz && \
    cd openmpi-4.1.2 && \
    ./configure CC=clang CXX=clang++ --prefix=/usr/local && \
    make -j $(nproc) && \
    make install && ldconfig && \
    echo "export OMPI_ALLOW_RUN_AS_ROOT=1" >> ~/.bashrc && \
    echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> ~/.bashrc

# Configure and install Boost
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz && \
    tar xf boost_1_79_0.tar.gz && rm boost_1_79_0.tar.gz && \
    cd boost_1_79_0 && \
    ./bootstrap.sh --with-toolset=clang && \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 -j $(nproc) && ./b2 install --prefix=/usr/local

# Configure and install HDF5
RUN wget https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_10_11/hdf5-1_10_11.tar.gz && \
    tar xf hdf5-1_10_11.tar.gz && rm hdf5-1_10_11.tar.gz && cd hdfsrc && \
    CC=mpicc CFLAGS=-fPIC ./configure --enable-shared --enable-parallel --prefix=/usr/local && \
    make -j $(nproc) && make install

# Configure and install netcdf-c
RUN wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.8.1.tar.gz && \
    tar xf v4.8.1.tar.gz && cd netcdf-c-4.8.1 && \
    CC=mpicc LIBS=-lhdf5 ./configure --prefix=/usr/local && \
    make -j $(nproc) && make install

# Configure and build/install Trilinos
COPY config_trilinos.sh /root/config_trilinos.sh

RUN wget https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-14-4-0.tar.gz && \
    tar xf trilinos-release-14-4-0.tar.gz && \
    rm trilinos-release-14-4-0.tar.gz && \
    mv Trilinos-trilinos-release-14-4-0 Trilinos && \
    mkdir Trilinos/build && cd Trilinos/build && \
    export TRILINOS_HOME='/git-repos/Trilinos' && \
    bash /root/config_trilinos.sh && \
    ninja && cmake --install .