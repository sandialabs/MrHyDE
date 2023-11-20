# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./
  ${Trilinos_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
  ../src
  ../src/physics
  ../src/interfaces
  ../src/managers
  ../src/tools
  ../src/subgrid
  ../src/optimization
)

LINK_DIRECTORIES (
  ${Trilinos_LIBRARY_DIRS} 
  ${Trilinos_TPL_LIBRARY_DIRS}
)

ADD_EXECUTABLE(test_views
test_views.cpp)
TARGET_LINK_LIBRARIES(test_views ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_explicit_views
test_explicit_views.cpp)
TARGET_LINK_LIBRARIES(test_explicit_views ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_vistas
test_vistas.cpp
../src/tools/vista.cpp)
TARGET_LINK_LIBRARIES(test_vistas ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_percept
test_percept.cpp
../src/subgrid/subgridMeshFactory.cpp) 
TARGET_LINK_LIBRARIES(test_percept ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

#ADD_EXECUTABLE(test_databaseviews
#test_databaseViews.cpp) 
#TARGET_LINK_LIBRARIES(test_databaseviews ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_compadre
test_compadre.cpp
../src/tools/data.cpp) 
TARGET_LINK_LIBRARIES(test_compadre ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_functions
test_functions.cpp
../src/tools/workset.cpp
../src/tools/vista.cpp
../src/tools/interpreter.cpp
../src/managers/functionManager.cpp)
TARGET_LINK_LIBRARIES(test_functions ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_series_ABC
test_series_ABC.cpp)
TARGET_LINK_LIBRARIES(test_series_ABC ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_DOFManagers
test_DOFManagers.cpp)
TARGET_LINK_LIBRARIES(test_DOFManagers ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_PanzerStk
test_PanzerStk.cpp)
TARGET_LINK_LIBRARIES(test_PanzerStk ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_Compression
test_Compression.cpp)
TARGET_LINK_LIBRARIES(test_Compression ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_SparseMass
test_SparseMass.cpp)
TARGET_LINK_LIBRARIES(test_SparseMass ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_SparseHDIVMass
test_SparseHDIVMass.cpp)
TARGET_LINK_LIBRARIES(test_SparseHDIVMass ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_parallel_for
test_parallel_for.cpp)
TARGET_LINK_LIBRARIES(test_parallel_for ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_scaling_reduction
test_scaling_reduction.cpp)
TARGET_LINK_LIBRARIES(test_scaling_reduction ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

#ADD_EXECUTABLE(test_ioss
#tIOSSConnManager.cpp)
#TARGET_LINK_LIBRARIES(test_ioss ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

