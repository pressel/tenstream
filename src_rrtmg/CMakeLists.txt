project(tenstr_rrtm_wrapper)
enable_language(Fortran C)
cmake_minimum_required (VERSION 2.8.10)

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

set(INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

add_subdirectory (rrtm_lw)
add_subdirectory (rrtm_sw)
add_subdirectory (rrtmg)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/rrtm_lw")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/rrtm_sw")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/rrtmg")
