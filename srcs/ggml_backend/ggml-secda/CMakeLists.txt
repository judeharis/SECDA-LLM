# Supported platforms: x86-64, arm32
if (GGML_SECDA)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  else()
      set(GGML_SECDA OFF)
      message(WARNING "SECDA-LLM supports x86-64, armv7, and aarch64 but not ${CMAKE_SYSTEM_PROCESSOR}. Turning off SECDA")
  endif()
endif()



if (GGML_SECDA)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "===================GGML_SECDA====================")

    # SECDA_TOOLS is a library that provides tools and utilities for SECDA
    find_package(SECDA_TOOLS REQUIRED)
    message(STATUS "SECDA found, Libraries: ${SECDA_TOOLS_LIBRARIES}")
    message(STATUS "SECDA found, Includes: ${SECDA_TOOLS_INCLUDES}")

    if(NOT BUILD_ARM)
        # SystemC library required for x86-64 builds of SECDA (simulation)
        find_package(SYSC REQUIRED)
    endif()
    
    # These options can be used to enable support for specific quantization formats and accelerator versions in SECDA. 
    # Enabling these options will define corresponding preprocessor macros that can be used in the code to conditionally compile support for these features.
    option(SECDA_QK2 "SECDA: QK2 support" OFF)
    option(SECDA_QK3 "SECDA: QK3 support" OFF)
    option(SECDA_QK4 "SECDA: QK4 support" OFF)
    option(SECDA_QK5 "SECDA: QK5 support" OFF)
    option(SECDA_QK6 "SECDA: QK6 support" OFF)
    option(SECDA_BFPP_ACC_V1 "SECDA: BFPP ACC V1 support" OFF)
    option(SECDA_BFPP_ACC_V2 "SECDA: BFPP ACC V2 support" OFF)
    option(SECDA_BFPP_ACC_V3 "SECDA: BFPP ACC V3 support" OFF)

    # Handling Quantization Formats
    if (SECDA_QK2)
        message(STATUS "SECDA QK2 enabled")
        list(APPEND SECDA_ACC_SUPPORT "-DGGML_SECDA_QK2")
    endif()
    if (SECDA_QK3)
        message(STATUS "SECDA QK3 enabled")
        list(APPEND SECDA_ACC_SUPPORT "-DGGML_SECDA_QK3")
    endif()
    if (SECDA_QK4)
        message(STATUS "SECDA QK4 enabled")
        list(APPEND SECDA_ACC_SUPPORT "-DGGML_SECDA_QK4")
    endif()
    if (SECDA_QK5)
        message(STATUS "SECDA QK5 enabled")
        list(APPEND SECDA_ACC_SUPPORT "-DGGML_SECDA_QK5")
    endif()
    if (SECDA_QK6)
        message(STATUS "SECDA QK6 enabled")
        list(APPEND SECDA_ACC_SUPPORT "-DGGML_SECDA_QK6")
    endif()

    # Handling Accelerators
    add_subdirectory(acc_dels)
    if (SECDA_BFPP_ACC_V3)
        message(STATUS "BFPP Accelerator enabled")
        list(APPEND GGML_CDEF_PUBLIC BFPP_ACC_V3)
        set(GGML_SECDA_ACC bfpp_acc_v3_driver_connector)
    elseif (SECDA_BFPP_ACC_V2)
        message(STATUS "BFPP Accelerator enabled")
        list(APPEND GGML_CDEF_PUBLIC BFPP_ACC_V2)
        set(GGML_SECDA_ACC bfpp_acc_v2_driver_connector)
    else ()
        message(STATUS "BFPP Accelerator enabled")
        list(APPEND GGML_CDEF_PUBLIC BFPP_ACC_V1)
        set(GGML_SECDA_ACC bfpp_acc_v1_driver_connector)
    endif()

    if (BUILD_ARM)
        list(APPEND GGML_CDEF_PUBLIC GGML_SECDA_ARM)
    endif()

    set(GGML_HEADERS_SECDA ../include/ggml-secda.h)
    file(GLOB GGML_SOURCES_SECDA "*.cpp")
    file(GLOB GGML_HEADERS_SECDA "*.h")

    # Add the ggml-secda library to llama.cpp
    ggml_add_backend_library(ggml-secda ${GGML_SOURCES_SECDA} ${GGML_HEADERS_SECDA})
    target_link_libraries(ggml-secda PRIVATE ${SECDA_TOOLS_LIBRARIES} ${SYSC_LIBRARIES} ${GGML_SECDA_ACC})
    target_include_directories(ggml-secda PRIVATE ${SECDA_TOOLS_INCLUDES})
    target_link_directories(ggml-secda PRIVATE ${GGML_SECDA_ACC})
    message(STATUS "===================GGML_SECDA====================")
endif()

