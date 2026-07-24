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
set(SECDA_LLM_SECDA_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/secda_tools" CACHE PATH "Path to the vendored secda_tools source tree")

if (NOT EXISTS "${SECDA_LLM_LLAMA_DIR}/CMakeLists.txt")
	message(FATAL_ERROR "SECDA-LLM requires llama.cpp at ${SECDA_LLM_LLAMA_DIR}")
endif()

option(SECDA_LLM_ENABLE_SECDA "Enable the SECDA ggml backend" ON)
option(SECDA_LLM_BUILD_SECDA_TOOLS "Build bundled secda_tools when SECDA is enabled" ON)

secda_llm_set_default(CMAKE_BUILD_TYPE STRING Release "Build type")
secda_llm_set_default(LLAMA_BUILD_COMMON BOOL ON "Build llama.cpp common utilities")
secda_llm_set_default(LLAMA_BUILD_TESTS BOOL ON "Build llama.cpp tests")
secda_llm_set_default(LLAMA_BUILD_TOOLS BOOL ON "Build llama.cpp tools")
secda_llm_set_default(LLAMA_BUILD_EXAMPLES BOOL ON "Build llama.cpp examples")
secda_llm_set_default(LLAMA_BUILD_SERVER BOOL ON "Build llama.cpp server example")
secda_llm_set_default(LLAMA_BUILD_APP BOOL ON "Build the unified llama.cpp app")

if (SECDA_LLM_ENABLE_SECDA)
	secda_llm_set_default(GGML_SECDA BOOL ON "Enable the SECDA ggml backend")
	secda_llm_set_default(SECDA_TOOLS_BUILD_SHARED BOOL ON "Build secda_tools as shared libraries")
	secda_llm_set_default(SECDA_TOOLS_BUILD_TESTS BOOL OFF "Build secda_tools tests")

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

	if (SECDA_LLM_BUILD_SECDA_TOOLS)
		if (NOT EXISTS "${SECDA_LLM_SECDA_TOOLS_DIR}/CMakeLists.txt")
			message(FATAL_ERROR
				"SECDA is enabled, but secda_tools was not found at ${SECDA_LLM_SECDA_TOOLS_DIR}. "
				"Update SECDA_LLM_SECDA_TOOLS_DIR or disable SECDA_LLM_BUILD_SECDA_TOOLS.")
		endif()

		set(SECDA_LLM_SECDA_TOOLS_INCLUDE_DIR "${SECDA_LLM_SECDA_TOOLS_DIR}" CACHE PATH "Include directory for bundled secda_tools" FORCE)
		add_subdirectory("${SECDA_LLM_SECDA_TOOLS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/secda_tools")

		# set(SECDA_TOOLS_COMMON_DEFINITIONS DMA_PROFILE DMA_PRINT DEF_UDMA)
		set(SECDA_TOOLS_COMMON_DEFINITIONS DEF_UDMA) #UBUF ENABLED


		if (BUILD_ARM)
			if (BUILD_KRIA)
				if (TARGET secda_tools_kria)
					target_compile_definitions(secda_tools_kria PUBLIC ${SECDA_TOOLS_COMMON_DEFINITIONS})
				endif()
			else()
				if (TARGET secda_tools)
					target_compile_definitions(secda_tools PUBLIC ${SECDA_TOOLS_COMMON_DEFINITIONS})
				endif()
			endif()
		else()
			foreach(secda_tools_target secda_tools_sim secda_tools_simv4 secda_tools_simv5)
				if (TARGET ${secda_tools_target})
					target_compile_definitions(${secda_tools_target} PUBLIC ${SECDA_TOOLS_COMMON_DEFINITIONS})
				endif()
			endforeach()
		endif()
	endif()
endif()

add_subdirectory("${SECDA_LLM_LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
