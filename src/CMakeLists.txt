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

file(GLOB mrhyde-all
  "*.cpp"
  "subgrid/*.cpp"
  "interfaces/*.cpp"
  "managers/*.cpp"
  "tools/*.cpp"
  "physics/*.cpp"
)

ADD_EXECUTABLE(mrhyde
  ${mrhyde-all}
)

TARGET_LINK_LIBRARIES(mrhyde ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 

IF(MrHyDE_ENABLE_XML_to_YAML)
  ADD_EXECUTABLE(xml_to_yaml
  ../scripts/utils/xml_to_yaml.cpp)
  TARGET_LINK_LIBRARIES(xml_to_yaml ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
ENDIF()

# Copy the perm_xy.dat file over to the directory where test_compadre is located
SET(SOURCE_FILE_FULL "${CMAKE_CURRENT_SOURCE_DIR}/test/perm_xy.dat")
SET(DEST_FILE_FULL "${CMAKE_CURRENT_BINARY_DIR}/perm_xy.dat")
ADD_CUSTOM_COMMAND(
 OUTPUT ${DEST_FILE_FULL}
 DEPENDS ${SOURCE_FILE_FULL}
 COMMAND ${CMAKE_COMMAND} ARGS -E copy ${SOURCE_FILE_FULL} ${DEST_FILE_FULL}
 )
