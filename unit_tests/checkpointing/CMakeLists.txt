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
