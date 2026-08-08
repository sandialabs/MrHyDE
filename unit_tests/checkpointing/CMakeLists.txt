# Unit tests for the Revolve checkpointing schedule (Griewank & Walther,
# "Algorithm 799: Revolve", ACM TOMS 26(1), 2000).
#
# revolve.hpp has no Trilinos, Kokkos or MPI dependencies, so this test
# deliberately links nothing and builds in a second.

INCLUDE_DIRECTORIES ( ./
  ../../src/tools
)

ADD_EXECUTABLE(revolve
revolve.cpp)

# State the standard on the target rather than inheriting it from the parent
# build, so this test compiles the same way wherever it is dropped in.
SET_TARGET_PROPERTIES(revolve PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

ADD_TEST(NAME checkpointing_revolve COMMAND revolve)

ADD_EXECUTABLE(bsw_revolve
bsw_revolve.cpp)

SET_TARGET_PROPERTIES(bsw_revolve PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

ADD_TEST(NAME checkpointing_bsw_revolve COMMAND bsw_revolve)

ADD_EXECUTABLE(threefry
threefry.cpp)

SET_TARGET_PROPERTIES(threefry PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

ADD_TEST(NAME checkpointing_threefry COMMAND threefry)

# The dense kernels bind to the BLAS/LAPACK the host already links; in this
# build those come in through the Trilinos link interface.
ADD_EXECUTABLE(dense_kernels
dense_kernels.cpp)

SET_TARGET_PROPERTIES(dense_kernels PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

TARGET_LINK_LIBRARIES(dense_kernels ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

ADD_TEST(NAME checkpointing_dense_kernels COMMAND dense_kernels)
