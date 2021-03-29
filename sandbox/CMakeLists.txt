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

ADD_EXECUTABLE(test_percept
test_percept.cpp
../src/subgrid/subgridMeshFactory.cpp) 
TARGET_LINK_LIBRARIES(test_percept ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_compadre
test_compadre.cpp
../src/tools/data.cpp) 
TARGET_LINK_LIBRARIES(test_compadre ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_functions
test_functions.cpp
../src/tools/workset.cpp
../src/tools/interpreter.cpp
../src/interfaces/discretizationInterface.cpp 
../src/managers/functionManager.cpp)
TARGET_LINK_LIBRARIES(test_functions ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
