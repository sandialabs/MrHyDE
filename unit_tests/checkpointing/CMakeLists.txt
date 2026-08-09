# Unit tests for the checkpointing stack: classic Revolve (Griewank &
# Walther, "Algorithm 799: Revolve", ACM TOMS 26(1), 2000), the BSW
# scheduler, and the window compressors.  revolve, bsw_revolve and threefry
# link nothing and build in a second; the window and dense-kernel tests bind
# to the BLAS/LAPACK already in the Trilinos link interface.

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

ADD_EXECUTABLE(sketch_window
sketch_window.cpp)

SET_TARGET_PROPERTIES(sketch_window PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

TARGET_LINK_LIBRARIES(sketch_window ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

ADD_TEST(NAME checkpointing_sketch_window COMMAND sketch_window)

ADD_EXECUTABLE(svd_window
svd_window.cpp)

SET_TARGET_PROPERTIES(svd_window PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

TARGET_LINK_LIBRARIES(svd_window ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

ADD_TEST(NAME checkpointing_svd_window COMMAND svd_window)

ADD_EXECUTABLE(bsw_window_manager
bsw_window_manager.cpp)

SET_TARGET_PROPERTIES(bsw_window_manager PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

TARGET_LINK_LIBRARIES(bsw_window_manager ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

ADD_TEST(NAME checkpointing_bsw_window_manager COMMAND bsw_window_manager)
