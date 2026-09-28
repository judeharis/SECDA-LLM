#pragma once

// llama_perf.csv: the per-run llama.cpp performance summary SECDA-LLM's
// benchmark pipeline collects (benchmark/scripts/parse_results.py). This used
// to be written by a change in the llama.cpp fork (llama_perf_context_print);
// the SECDA tool variants (srcs/tools) now call this instead.

struct llama_context;

void secda_write_llama_perf_csv(const llama_context *ctx);
