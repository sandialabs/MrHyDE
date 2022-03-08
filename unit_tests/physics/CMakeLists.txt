# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./
  ${Trilinos_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
  ../../src
  ../../src/physics
  ../../src/interfaces
  ../../src/managers
  ../../src/tools
  ../../src/subgrid
  ../../src/optimization
)

LINK_DIRECTORIES (
  ${Trilinos_LIBRARY_DIRS} 
  ${Trilinos_TPL_LIBRARY_DIRS}
)

ADD_EXECUTABLE(NavierStokes
NavierStokes.cpp
../../src/physics/navierstokes.cpp
../../src/managers/functionManager.cpp
../../src/tools/interpreter.cpp
../../src/tools/workset.cpp)
TARGET_LINK_LIBRARIES(NavierStokes ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(Euler
Euler.cpp
../../src/physics/euler.cpp
../../src/managers/functionManager.cpp
../../src/tools/interpreter.cpp
../../src/tools/workset.cpp
../../src/interfaces/discretizationInterface.cpp)
TARGET_LINK_LIBRARIES(Euler ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(shallowwaterHybridized
shallowwaterHybridized.cpp
../../src/physics/shallowwaterHybridized.cpp
../../src/managers/functionManager.cpp
../../src/tools/interpreter.cpp
../../src/tools/workset.cpp
../../src/interfaces/discretizationInterface.cpp)
TARGET_LINK_LIBRARIES(shallowwaterHybridized ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
TARGET_COMPILE_DEFINITIONS(shallowwaterHybridized PRIVATE MrHyDE_NO_SOL_FIELD_EVAL)
