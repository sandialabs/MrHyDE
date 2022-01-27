# CMAKE File for "main" application building against an installed Trilinos

cmake_minimum_required(VERSION 2.8)

ENABLE_TESTING()

SET(TRILINOS_DIR ${Trilinos_INSTALL_DIR})
SET(Trilinos_SRC_DIR ${Trilinos_SRC_DIR})

set(CMAKE_PREFIX_PATH ${TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Check the available Kokkos devices at the time of configuration
MESSAGE("\n")
MESSAGE("=======================================================")
MESSAGE("MrHyDE configuration details")
MESSAGE("-- Checking if Kokkos has CUDA enabled")
kokkos_check(DEVICES CUDA RETURN_VALUE HAS_CUDA) # GH: This doesn't seem to work on my desktop, but it works on other machines.
IF(HAS_CUDA)
  MESSAGE("---- Found Kokkos::CUDA ")
  MESSAGE("---- MrHyDE is temporarily disabling Compadre functionality since a GPU is available.  Affects data import.")
  ADD_DEFINITIONS(-DMrHyDE_DISABLE_COMPADRE)
ELSE()
  MESSAGE("---- Did not find Kokkos::CUDA ")
ENDIF()

MESSAGE("-- Checking if Kokkos has OpenMP enabled")
kokkos_check(DEVICES OPENMP RETURN_VALUE HAS_OPENMP)
IF(HAS_OPENMP)
  MESSAGE("---- Found Kokkos::OPENMP ")
ELSE()
  MESSAGE("---- Did not find Kokkos::OPENMP ")
ENDIF()

# Flags to set the Kokkos devices at configuration time
IF(MrHyDE_ALL_SERIAL OR MrHyDE_ALL_CPU) 
  MESSAGE("-- Configuration specified using a serial configuration for MrHyDE.")
ELSEIF(MrHyDE_ALL_OPENMP) 
  IF(HAS_OPENMP)
    MESSAGE("-- Configuration specified using a OpenMP configuration for MrHyDE.")
    ADD_DEFINITIONS(-DMrHyDE_HOSTEXEC_OPENMP)
    ADD_DEFINITIONS(-DMrHyDE_ASSEMBLYSPACE_OPENMP)
    ADD_DEFINITIONS(-DMrHyDE_SOLVERSPACE_OPENMP)
  ELSE()
    MESSAGE("-- Configuration specified using OpenMP, but Trilinos was not configured with OpenMP.  Defaulting to serial.")
  ENDIF()
ELSEIF(MrHyDE_ALL_CUDA OR MrHyDE_ALL_GPU) 
  IF(HAS_CUDA)
    MESSAGE("-- Configuration specified using a CUDA configuration for MrHyDE.")
    IF(HAS_OPENMP)
      ADD_DEFINITIONS(-DMrHyDE_HOSTEXEC_OPENMP)
    ENDIF()
    ADD_DEFINITIONS(-DMrHyDE_ASSEMBLYSPACE_CUDA)
    ADD_DEFINITIONS(-DMrHyDE_SOLVERSPACE_CUDA)
  ELSE()
    MESSAGE("-- Configuration specified using CUDA, but Trilinos was not configured with CUDA.  Defaulting to serial.")
  ENDIF()
ELSEIF(MrHyDE_ASSEMBLY_CUDA OR MrHyDE_ASSEMBLY_GPU)
  IF(HAS_CUDA)
    MESSAGE("-- Configuration specified performing assembly using CUDA for MrHyDE.")
    ADD_DEFINITIONS(-DMrHyDE_ASSEMBLYSPACE_CUDA)
    IF(MrHyDE_SOLVER_CUDA OR MrHyDE_SOLVER_GPU)
      MESSAGE("---- Configuration is enabling CUDA for solver in MrHyDE.")
      ADD_DEFINITIONS(-DMrHyDE_SOLVERSPACE_CUDA)
    ELSE()
      IF(HAS_OPENMP) 
        MESSAGE("---- CUDA was enabled, but MrHyDE is defaulting to OpenMP for the linear solvers. Turn on MrHyDE_SOLVER_CUDA to change.")
        ADD_DEFINITIONS(-DMrHyDE_SOLVERSPACE_OPENMP)
      ELSE()
        MESSAGE("---- CUDA was enabled, but MrHyDE is defaulting to serial for the linear solvers. Turn on MrHyDE_SOLVER_CUDA to change.")
      ENDIF()
    ENDIF()
  ELSE()
    MESSAGE("-- Configuration requested to use CUDA for assembly in MrHyDE but no available Kokkos device was found. Defaulting to serial.")
  ENDIF()
ELSEIF(MrHyDE_ASSEMBLY_OPENMP)
  IF(HAS_OPENMP)
    MESSAGE("-- Configuration specified performing assembly using OpenMP for MrHyDE.")
    ADD_DEFINITIONS(-DMrHyDE_ASSEMBLYSPACE_OPENMP)
    IF(MrHyDE_SOLVER_OPENMP)
      MESSAGE("-- Configuration is enabling OpenMP for solver in MrHyDE.")
      ADD_DEFINITIONS(-DMrHyDE_SOLVERSPACE_OPENMP)
    ENDIF()
  ELSE()
    MESSAGE("-- Configuration requested to use OpenMP for MrHyDE but no available Kokkos device was found. Defaulting to serial.")
  ENDIF()
ELSE()
  MESSAGE("-- No device configurations specified. Defaulting to a serial configuration for MrHyDE.")
ENDIF()

IF(MrHyDE_MAX_DERIVS)
  MESSAGE("-- Setting max derivs to ${MrHyDE_MAX_DERIVS} based on input flag")
  add_definitions(-DMrHyDE_SET_MAX_DERIVS=${MrHyDE_MAX_DERIVS})
ELSE()
  MESSAGE("-- No max derivs specified. Defaulting to 64.")
ENDIF()

IF(MrHyDE_DISABLE_AD)
  MESSAGE("-- Turning OFF automatic differentiation (AD) based on input flag")
  MESSAGE("-- CAUTION: THIS SHOULD ONLY BE USED WITH THE FULLY EXPLICIT MODE")
  ADD_DEFINITIONS(-DMrHyDE_NO_AD)
ENDIF()

IF(MrHyDE_SINGLE_PRECISION)
  MESSAGE("-- Running with single precision based on input flag")
  MESSAGE("-- CAUTION: I HOPE YOU KNOW WHAT YOU ARE DOING")
  ADD_DEFINITIONS(-DMrHyDE_SINGLE_PRECISION)
ENDIF()

IF(MrHyDE_VECTOR_SIZE)
  MESSAGE("-- Setting vector size to ${MrHyDE_VECTOR_SIZE} based on input flag")
  add_definitions(-DMrHyDE_SET_VECTOR_SIZE=${MrHyDE_VECTOR_SIZE})
ELSE()
  IF(MrHyDE_MAX_DERIVS)
    MESSAGE("-- No vector size specified. Defaulting to ${MrHyDE_MAX_DERIVS}.")
  ELSE()
    MESSAGE("-- No vector size specified. Defaulting to 64.")
  ENDIF()
ENDIF()

IF(MrHyDE_BUILD_DOXYGEN)
  MESSAGE("Processing Doxygen build")
  find_package(Doxygen)
  IF(DOXYGEN_FOUND)
    MESSAGE(STATUS "Doxygen found")

    SET(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
    SET(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        COMMENT "Generating documentation with Doxygen"
        VERBATIM )
  ELSE()
    MESSAGE(FATAL_ERROR "Doxygen requested, but not found!")
  ENDIF()
ENDIF()

IF(MrHyDE_ENABLE_MIRAGE_EXTENSIONS)
  ADD_DEFINITIONS(-DMrHyDE_ENABLE_MIRAGE)
  MESSAGE("-- Mirage extensions have been enabled.")
  IF(MrHyDE_MIRAGE_EXTENSIONS_DIR) 
    MESSAGE("-- Mirage extensions directory: ${MrHyDE_MIRAGE_EXTENSIONS_DIR}")
  ELSE()
    MESSAGE(FATAL_ERROR "Mirage extensions requested, but MrHyDE_MIRAGE_EXTENSIONS_DIR was not set.")
  ENDIF()  
ENDIF()

IF(MrHyDE_ENABLE_FFTW)
  ADD_DEFINITIONS(-DMrHyDE_ENABLE_FFTW)
  MESSAGE("-- FFTW functionality enabled.")
  IF(MrHyDE_FFTW_INCLUDE_DIR)
    MESSAGE("-- FFTW include directory: ${MrHyDE_FFTW_INCLUDE_DIR}")
  ELSE()
    MESSAGE(FATAL_ERROR "FFTW requested, but MrHyDE_FFTW_INCLUDE_DIR was not set.")
  ENDIF()
  IF(MrHyDE_FFTW_LIB_DIR)
    MESSAGE("-- FFTW library directory: ${MrHyDE_FFTW_LIB_DIR}")
  ELSE()
    MESSAGE(FATAL_ERROR "FFTW requested, but MrHyDE_FFTW_LIB_DIR was not set.")
  ENDIF()
ENDIF()

MESSAGE("End of MrHyDE configuration details")
MESSAGE("=======================================================")
MESSAGE("\n")

# Make sure to use same compilers and flags as Trilinos
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

SET(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
SET(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")

#
# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos
PROJECT(mrhyde)

# SET(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")

include(CTest)

# MPI check 
MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(main_MPI TRUE)
  ADD_DEFINITIONS(-Dmain_MPI)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(main_MPI FALSE)
ENDIF()

MESSAGE("   CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

# Compile sandbox scripts
ADD_SUBDIRECTORY(src)
IF(MrHyDE_ENABLE_SANDBOX)
  ADD_SUBDIRECTORY(sandbox)
ENDIF()

# Compile unit tests
IF(MrHyDE_ENABLE_UNIT_TESTS)
  ADD_SUBDIRECTORY(unit_tests)
ENDIF()
  
#INSTALL(
#    DIRECTORY ""
#    DESTINATION ""
#    )
