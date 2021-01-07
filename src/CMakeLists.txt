# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./
  ${Trilinos_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
  physics
  interfaces
  managers
  tools
  subgrid
  optimization
)

LINK_DIRECTORIES (
  ${Trilinos_LIBRARY_DIRS} 
  ${Trilinos_TPL_LIBRARY_DIRS}
)

file(GLOB tst
  "*.cpp"
  "interfaces/*.cpp"
  "managers/*.cpp"
  "tools/*.cpp"
  "physics/*.cpp"
  "subgrid/*.cpp"
)

ADD_EXECUTABLE(milo
  ${tst}
)

TARGET_LINK_LIBRARIES(milo ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

IF(MrHyDE_ENABLE_XML_to_YAML)
  ADD_EXECUTABLE(xml_to_yaml
  ../scripts/utils/xml_to_yaml.cpp)
  TARGET_LINK_LIBRARIES(xml_to_yaml ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
ENDIF()

IF(MrHyDE_ENABLE_TESTS)
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
ENDIF()

# Copy the perm_xy.dat file over to the directory where test_compadre is located
SET(SOURCE_FILE_FULL "${CMAKE_CURRENT_SOURCE_DIR}/test/perm_xy.dat")
SET(DEST_FILE_FULL "${CMAKE_CURRENT_BINARY_DIR}/perm_xy.dat")
ADD_CUSTOM_COMMAND(
 OUTPUT ${DEST_FILE_FULL}
 DEPENDS ${SOURCE_FILE_FULL}
 COMMAND ${CMAKE_COMMAND} ARGS -E copy ${SOURCE_FILE_FULL} ${DEST_FILE_FULL}
 )
