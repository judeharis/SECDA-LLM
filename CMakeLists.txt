cmake_minimum_required(VERSION 3.14...3.28)

project("SECDA-LLM" LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

function(secda_llm_set_default option_name option_type default_value docstring)
	if (NOT DEFINED ${option_name})
		set(${option_name} ${default_value} CACHE ${option_type} ${docstring})
	endif()
endfunction()

set(SECDA_LLM_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to the vendored llama.cpp source tree")
# SECDA-Core: $SECDA_CORE_DIR, else SECDA-DS's own SECDA-Core when this repo sits inside
# SECDA-DS (found through the suite's .gitmodules), else the third_party/secda_core
# submodule. The same order as ./secda. Inside SECDA-DS the submodule is left
# unchecked-out, so the suite holds one SECDA-Core.
set(_secda_core_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/secda_core")
get_filename_component(_secda_suite "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
if (DEFINED ENV{SECDA_CORE_DIR})
	set(_secda_core_default "$ENV{SECDA_CORE_DIR}")
elseif (EXISTS "${_secda_suite}/.gitmodules" AND EXISTS "${_secda_suite}/SECDA-Core/secda-core")
	file(READ "${_secda_suite}/.gitmodules" _secda_suite_modules)
	string(FIND "${_secda_suite_modules}" "path = SECDA-Core" _secda_in_suite)
	if (NOT _secda_in_suite EQUAL -1)
		set(_secda_core_default "${_secda_suite}/SECDA-Core")
	endif()
endif()
set(SECDA_LLM_SECDA_CORE_DIR "${_secda_core_default}" CACHE PATH "Path to SECDA-Core")

if (NOT EXISTS "${SECDA_LLM_LLAMA_DIR}/CMakeLists.txt")
	message(FATAL_ERROR "SECDA-LLM requires llama.cpp at ${SECDA_LLM_LLAMA_DIR}")
endif()

option(SECDA_LLM_ENABLE_SECDA "Enable the SECDA ggml backend" ON)
option(SECDA_LLM_BUILD_SECDA_CORE "Build the SECDA-Core submodule when SECDA is enabled" ON)

secda_llm_set_default(CMAKE_BUILD_TYPE STRING Release "Build type")
secda_llm_set_default(LLAMA_BUILD_COMMON BOOL ON "Build llama.cpp common utilities")
secda_llm_set_default(LLAMA_BUILD_TESTS BOOL ON "Build llama.cpp tests")
secda_llm_set_default(LLAMA_BUILD_TOOLS BOOL ON "Build llama.cpp tools")
secda_llm_set_default(LLAMA_BUILD_EXAMPLES BOOL ON "Build llama.cpp examples")
secda_llm_set_default(LLAMA_BUILD_SERVER BOOL ON "Build llama.cpp server example")
secda_llm_set_default(LLAMA_BUILD_APP BOOL ON "Build the unified llama.cpp app")

if (SECDA_LLM_ENABLE_SECDA)
	secda_llm_set_default(GGML_SECDA BOOL ON "Enable the SECDA ggml backend")
	secda_llm_set_default(SECDA_CORE_BUILD_SHARED BOOL ON "Build SECDA-Core as shared libraries")
	secda_llm_set_default(SECDA_CORE_BUILD_TESTS BOOL OFF "Build SECDA-Core tests")

	if (NOT EXISTS "${SECDA_LLM_LLAMA_DIR}/ggml/src/ggml-secda/CMakeLists.txt")
		message(FATAL_ERROR
			"GGML_SECDA is enabled, but the SECDA backend is not linked into llama.cpp. "
			"Run ./setup.sh from the repository root and re-configure.")
	endif()

	if (NOT EXISTS "${SECDA_LLM_LLAMA_DIR}/ggml/include/ggml-secda.h")
		message(FATAL_ERROR
			"GGML_SECDA is enabled, but ggml-secda.h is missing from llama.cpp/ggml/include. "
			"Run ./setup.sh from the repository root and re-configure.")
	endif()

	if (SECDA_LLM_BUILD_SECDA_CORE)
		if (NOT EXISTS "${SECDA_LLM_SECDA_CORE_DIR}/CMakeLists.txt")
			message(FATAL_ERROR
				"SECDA is enabled, but SECDA-Core was not found at ${SECDA_LLM_SECDA_CORE_DIR}. "
				"Run `git submodule update --init third_party/secda_core`, set SECDA_CORE_DIR or SECDA_LLM_SECDA_CORE_DIR, "
				"or disable SECDA_LLM_BUILD_SECDA_CORE.")
		endif()

		# The include root is the submodule root: headers are #include "secda-core/...".
		set(SECDA_LLM_SECDA_CORE_INCLUDE_DIR "${SECDA_LLM_SECDA_CORE_DIR}" CACHE PATH "Include directory for the SECDA-Core submodule" FORCE)
		add_subdirectory("${SECDA_LLM_SECDA_CORE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/secda_core")

		# SECDA-Core v5 dropped the CMA path: DEF_BUF selects u-dma-buf allocation
		# (the old toolkit had a UDMA define for the same thing).
		# set(SECDA_CORE_COMMON_DEFINITIONS DMA_PROFILE DMA_PRINT DEF_BUF)
		set(SECDA_CORE_COMMON_DEFINITIONS DEF_BUF)

		if (BUILD_ARM)
			if (BUILD_KRIA)
				if (TARGET secda_core_kria)
					target_compile_definitions(secda_core_kria PUBLIC ${SECDA_CORE_COMMON_DEFINITIONS})
				endif()
			else()
				if (TARGET secda_core)
					target_compile_definitions(secda_core PUBLIC ${SECDA_CORE_COMMON_DEFINITIONS})
				endif()
			endif()
		else()
			# The old toolkit also shipped a plain sim target and a v4 sim target; SECDA-Core ships v5 and v6 only.
			foreach(secda_core_target secda_core_simv5 secda_core_simv6)
				if (TARGET ${secda_core_target})
					target_compile_definitions(${secda_core_target} PUBLIC ${SECDA_CORE_COMMON_DEFINITIONS})
				endif()
			endforeach()
		endif()
	endif()
endif()

add_subdirectory("${SECDA_LLM_LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
