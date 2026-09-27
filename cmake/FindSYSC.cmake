# - Find SYSC
# Find SystemC (simulation builds).
#
#  SYSC_INCLUDES    - where to find systemc.h
#  SYSC_LIBRARIES   - the SystemC library
#  SYSC_FOUND       - True if SystemC was found.
#
# Searches SYSTEMC_HOME (CMake variable, then environment).

if(SYSC_INCLUDES)
  set(SYSC_FIND_QUIETLY TRUE)
endif()

set(_sysc_home "${SYSTEMC_HOME}")
if(NOT _sysc_home)
  set(_sysc_home "$ENV{SYSTEMC_HOME}")
endif()

find_path(SYSC_INCLUDES systemc.h PATHS "${_sysc_home}/include")
find_library(SYSC_LIBRARIES NAMES systemc-2.3.3 systemc
  PATHS "${_sysc_home}/lib-linux64" "${_sysc_home}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYSC DEFAULT_MSG SYSC_LIBRARIES SYSC_INCLUDES)
mark_as_advanced(SYSC_LIBRARIES SYSC_INCLUDES)
