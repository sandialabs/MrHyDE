# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./
                     ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS}
physics
interfaces
managers
tools
subgrid
optimization
user
../sparsegrid)

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

ADD_EXECUTABLE(milo
driver.cpp
subgrid/subgridMeshFactory.cpp 
subgrid/subgridFEM.cpp 
subgrid/subgridFEM_solver.cpp 
interfaces/meshInterface.cpp 
interfaces/discretizationInterface.cpp 
interfaces/physicsInterface.cpp 
interfaces/solverInterface.cpp 
interfaces/analysisInterface.cpp 
managers/uqManager.cpp 
managers/multiscaleManager.cpp 
managers/postprocessManager.cpp 
managers/assemblyManager.cpp
managers/parameterManager.cpp
managers/functionManager.cpp
tools/data.cpp
tools/cellMetaData.cpp
tools/cell.cpp
tools/boundaryCell.cpp
tools/workset.cpp
tools/split_mpi_communicators.cpp 
physics/thermal.cpp
physics/linearelasticity.cpp
physics/helmholtz.cpp
physics/maxwell.cpp
physics/maxwells_fp.cpp
physics/maxwell_hybridized.cpp
physics/msphasefield.cpp
physics/navierstokes.cpp
physics/cdr.cpp
physics/ode.cpp
physics/porous.cpp
physics/porousHDIV.cpp
physics/porousHDIV_hybridized.cpp
physics/porousHDIV_weakGalerkin.cpp
physics/shallowwater.cpp
physics/stokes.cpp)
#physics/twophasePoNo.cpp
#physics/twophasePoPw.cpp
#physics/thermal_enthalpy.cpp
TARGET_LINK_LIBRARIES(milo ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(xml_to_yaml
tools/xml_to_yaml.cpp)
TARGET_LINK_LIBRARIES(xml_to_yaml ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_interpreter
test/test_interpreter.cpp)
TARGET_LINK_LIBRARIES(test_interpreter ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_views
test/test_views.cpp)
TARGET_LINK_LIBRARIES(test_views ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

ADD_EXECUTABLE(test_percept
test/test_percept.cpp
subgrid/subgridMeshFactory.cpp) 
TARGET_LINK_LIBRARIES(test_percept ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

IF(MrHyDE_TEST_Compadre)
  ADD_EXECUTABLE(test_compadre
  test/test_compadre.cpp
  tools/data.cpp) 
  TARGET_LINK_LIBRARIES(test_compadre ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
ENDIF()

ADD_EXECUTABLE(test_functions
test/test_functions.cpp
tools/workset.cpp
interfaces/discretizationInterface.cpp 
managers/functionManager.cpp)
TARGET_LINK_LIBRARIES(test_functions ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

# Copy the perm_xy.dat file over to the directory where test_compadre is located
SET(SOURCE_FILE_FULL "${CMAKE_CURRENT_SOURCE_DIR}/test/perm_xy.dat")
SET(DEST_FILE_FULL "${CMAKE_CURRENT_BINARY_DIR}/perm_xy.dat")
ADD_CUSTOM_COMMAND(
 OUTPUT ${DEST_FILE_FULL}
 DEPENDS ${SOURCE_FILE_FULL}
 COMMAND ${CMAKE_COMMAND} ARGS -E copy ${SOURCE_FILE_FULL} ${DEST_FILE_FULL}
 )
