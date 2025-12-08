# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./
  ${Trilinos_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/hyperparameter_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/problem_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/ouu_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/transient_prior_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/u_prior_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/interfaces/z_prior_interfaces
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/analysis/prior
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/analysis/posterior
  ${HdsaLib_SRC_DIR}/src/core/model_discrepancy/analysis/auxillary
  ${HdsaLib_SRC_DIR}/src/core/parametric_continuation/interfaces
  ${HdsaLib_SRC_DIR}/src/core/parametric_continuation/analysis
  ${HdsaLib_SRC_DIR}/src/core/base/auxillary
  ${HdsaLib_SRC_DIR}/src/core/base/linear_algebra
  ${HdsaLib_SRC_DIR}/src/core/base/vectors
  ${HdsaLib_SRC_DIR}/src/trilinos_adapter/adapters
  ${HdsaLib_SRC_DIR}/src/trilinos_adapter/core
  ${HdsaLib_SRC_DIR}/interfaces/mrhyde/md_interfaces
  ${HdsaLib_SRC_DIR}/interfaces/mrhyde/auxillary
  ${HdsaLib_SRC_DIR}/interfaces/rol
  physics
  interfaces
  managers/analysis
  managers/assembly
  managers/function
  managers/multiscale
  managers/parameter
  managers/postprocess
  managers/solver
  managers/uq
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
  "managers/analysis/*.cpp"
  "managers/assembly/*.cpp"
  "managers/function/*.cpp"
  "managers/multiscale/*.cpp"
  "managers/parameter/*.cpp"
  "managers/postprocess/*.cpp"
  "managers/solver/*.cpp"
  "managers/uq/*.cpp"
  "tools/*.cpp"
  "physics/*.cpp"
)

IF(MrHyDE_ENABLE_MIRAGE_EXTENSIONS)
  INCLUDE_DIRECTORIES ( ./
    ${MrHyDE_MIRAGE_EXTENSIONS_DIR} 
  )  
  file(GLOB mirage
    "${MrHyDE_MIRAGE_EXTENSIONS_DIR}/*.cpp"
  )
  list(APPEND mrhyde-all "${MrHyDE_MIRAGE_EXTENSIONS_DIR}/mirage.cpp")
ENDIF()

IF(MrHyDE_ENABLE_PIEZO_EXTENSIONS)
  INCLUDE_DIRECTORIES ( ./
    ${MrHyDE_PIEZO_EXTENSIONS_DIR} 
  )  
  file(GLOB piezo
    "${MrHyDE_PIEZO_EXTENSIONS_DIR}/*.cpp"
  )
  list(APPEND mrhyde-all "${MrHyDE_PIEZO_EXTENSIONS_DIR}/piezo_maxwell.cpp")
  list(APPEND mrhyde-all "${MrHyDE_PIEZO_EXTENSIONS_DIR}/piezo_mechanics.cpp")
  list(APPEND mrhyde-all "${MrHyDE_PIEZO_EXTENSIONS_DIR}/full_tensor.cpp")
ENDIF()

#IF(MrHyDE_ENABLE_FFTW)
#  ## stand-alone target for now
#  ADD_EXECUTABLE(fft_fftw 
#  ../scripts/utils/fft_fftw.cpp)
#  TARGET_INCLUDE_DIRECTORIES(fft_fftw PUBLIC 
#    ${MrHyDE_FFTW_INCLUDE_DIR})
#  ## for intel builds on HPCs
#  FIND_LIBRARY(FFTW3_LIB NAMES fftw3xc_intel PATHS ${MrHyDE_FFTW_LIB_DIR} NO_DEFAULT_PATH)
#  ## C, intel math library needed
#  TARGET_LINK_LIBRARIES(fft_fftw ${FFTW3_LIB} m)
#  TARGET_LINK_OPTIONS(fft_fftw PRIVATE -mkl)
#  TARGET_LINK_LIBRARIES(fft_fftw ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
#ENDIF()

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
