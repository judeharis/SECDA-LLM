# SECDA-owned variants of llama.cpp tools. Each is built from the pinned
# llama.cpp's own source plus a patch in patches/, so the llama.cpp fork itself
# carries none of these changes:
#   secda-test-backend-ops  tests/test-backend-ops.cpp: -t, --test-params, tbo.csv,
#                           SECDA plan for perf runs (benchmark/scripts/run_synth_bench.sh)
#   secda-llama-bench       tools/llama-bench: writes llama_perf.csv
#   secda-llama-cli         tools/cli: prints the perf summary, writes llama_perf.csv
# The upstream tools are still built, unchanged.

find_program(SECDA_PATCH_EXECUTABLE patch)
if(NOT SECDA_PATCH_EXECUTABLE)
  message(FATAL_ERROR "SECDA-LLM tools need 'patch' (or set SECDA_LLM_BUILD_TOOLS=OFF)")
endif()
find_package(Threads REQUIRED)

# Same warning flags llama.cpp gives its own tools.
if(COMMAND llama_add_compile_flags)
  llama_add_compile_flags()
endif()

set(SECDA_TOOLS_PATCHED_DIR ${CMAKE_CURRENT_BINARY_DIR}/patched)
file(MAKE_DIRECTORY ${SECDA_TOOLS_PATCHED_DIR})

# secda_patched_source(<out var> <path under llama.cpp> <patch file>)
function(secda_patched_source out_var upstream_rel patch)
  get_filename_component(name ${upstream_rel} NAME)
  set(in ${SECDA_LLM_LLAMA_DIR}/${upstream_rel})
  set(out ${SECDA_TOOLS_PATCHED_DIR}/${name})
  set(pf ${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch})
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -DPATCH=${SECDA_PATCH_EXECUTABLE} -DIN=${in} -DPATCHFILE=${pf}
            -DOUT=${out} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/secda_apply_patch.cmake
    DEPENDS ${in} ${pf} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/secda_apply_patch.cmake
    COMMENT "Patching ${upstream_rel} with ${patch}"
    VERBATIM)
  set(${out_var} ${out} PARENT_SCOPE)
endfunction()

function(secda_tool_output target)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

# llama_perf.csv writer
add_library(secda-perf-csv STATIC secda_perf/secda_perf_csv.cpp)
target_include_directories(secda-perf-csv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/secda_perf)
target_link_libraries(secda-perf-csv PUBLIC llama)
set_target_properties(secda-perf-csv PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET llama-common)
  secda_patched_source(tbo_src tests/test-backend-ops.cpp test-backend-ops.patch)
  add_executable(secda-test-backend-ops ${tbo_src})
  target_link_libraries(secda-test-backend-ops PRIVATE llama llama-common Threads::Threads)
  secda_tool_output(secda-test-backend-ops)

  secda_patched_source(bench_src tools/llama-bench/llama-bench.cpp llama-bench.patch)
  add_executable(secda-llama-bench ${bench_src} ${SECDA_LLM_LLAMA_DIR}/tools/llama-bench/main.cpp)
  target_include_directories(secda-llama-bench PRIVATE ${SECDA_LLM_LLAMA_DIR}/tools/llama-bench)
  target_link_libraries(secda-llama-bench PRIVATE llama-common llama secda-perf-csv Threads::Threads)
  secda_tool_output(secda-llama-bench)
endif()

if(TARGET server-context)
  secda_patched_source(cli_src tools/cli/cli.cpp cli.patch)
  add_executable(secda-llama-cli ${cli_src} ${SECDA_LLM_LLAMA_DIR}/tools/cli/main.cpp)
  target_include_directories(secda-llama-cli PRIVATE ${SECDA_LLM_LLAMA_DIR}/tools/cli ${SECDA_LLM_LLAMA_DIR}/tools/server)
  target_link_libraries(secda-llama-cli PRIVATE server-context llama-common secda-perf-csv Threads::Threads)
  secda_tool_output(secda-llama-cli)
endif()
