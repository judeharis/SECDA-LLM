# - Find SECDA_CORE
# Find an out-of-tree SECDA-Core build (axi_support v6).
#
# Only a fallback: SECDA-LLM's CMakeLists.txt normally add_subdirectory()s the
# resolved SECDA-Core ($SECDA_CORE_DIR, then SECDA-DS's ../SECDA-Core, then
# third_party/secda_core), so the secda_core* targets already exist and
# ggml-secda uses them without calling this module.
#
# SECDA_CORE_INCLUDES        - the SECDA-Core repo root (headers are secda-core/...)
# SECDA_CORE_LIBRARIES       - secda_corev6 (ARM) or secda_core_simv6 (host)
# SECDA_CORE_SIM_LIBRARIES   - secda_core_simv6 (host builds only)
# SECDA_CORE_FOUND           - True if SECDA-Core was found.

if(SECDA_CORE_INCLUDES)
  set(SECDA_CORE_FIND_QUIETLY TRUE)
endif()

set(SECDA_CORE_SEARCH_ROOTS "")
foreach(root "${SECDA_LLM_SECDA_CORE_DIR}" "${SECDA_CORE_DIR}" "$ENV{SECDA_CORE_DIR}"
             "${SECDA_CORE_HOME}" "$ENV{SECDA_CORE_HOME}")
  if(root)
    list(APPEND SECDA_CORE_SEARCH_ROOTS "${root}")
  endif()
endforeach()
list(REMOVE_DUPLICATES SECDA_CORE_SEARCH_ROOTS)

find_path(SECDA_CORE_INCLUDES
  NAMES secda-core/axi_support/v6/axi_api.h
  PATHS ${SECDA_CORE_SEARCH_ROOTS})

# SECDA-Core's own CMake puts build trees at build/{host,pynq,kria}, with the
# libraries under <build>/secda-core/.
set(SECDA_CORE_LIB_SUFFIXES
  build/host/secda-core build/pynq/secda-core build/kria/secda-core
  build/secda-core out/build/secda-core build out/build)

if(BUILD_ARM)
  find_library(SECDA_CORE_LIBRARIES NAMES secda_corev6
    PATHS ${SECDA_CORE_SEARCH_ROOTS} PATH_SUFFIXES ${SECDA_CORE_LIB_SUFFIXES})
  set(_secda_core_required SECDA_CORE_LIBRARIES SECDA_CORE_INCLUDES)
else()
  find_library(SECDA_CORE_SIM_LIBRARIES NAMES secda_core_simv6
    PATHS ${SECDA_CORE_SEARCH_ROOTS} PATH_SUFFIXES ${SECDA_CORE_LIB_SUFFIXES})
  set(SECDA_CORE_LIBRARIES "${SECDA_CORE_SIM_LIBRARIES}")
  set(_secda_core_required SECDA_CORE_SIM_LIBRARIES SECDA_CORE_INCLUDES)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SECDA_CORE DEFAULT_MSG ${_secda_core_required})
mark_as_advanced(SECDA_CORE_LIBRARIES SECDA_CORE_SIM_LIBRARIES SECDA_CORE_INCLUDES)
