#include "secda_perf_csv.h"

#include "ggml.h"
#include "llama.h"

#include <fstream>

// Same file, header and values as the fork's writer (llama-context.cpp,
// llama_perf_context_print), including its default stream formatting.
void secda_write_llama_perf_csv(const llama_context *ctx) {
  const auto data = llama_perf_context(ctx);
  const double t_end_ms = 1e-3 * ggml_time_us();

  std::ofstream perf_file("llama_perf.csv");
  perf_file << "load time, prompt eval time, prompt tokens, prompt ms per token,";
  perf_file << " prompt tokens per second, eval time, eval tokens,";
  perf_file << " eval ms per token, eval tokens per second, total time, total tokens" << std::endl;
  perf_file << data.t_load_ms << "," << data.t_p_eval_ms << "," << data.n_p_eval << "," << data.t_p_eval_ms / data.n_p_eval << ",";
  perf_file << 1e3 / data.t_p_eval_ms * data.n_p_eval << "," << data.t_eval_ms << "," << data.n_eval << ",";
  perf_file << data.t_eval_ms / data.n_eval << "," << 1e3 / data.t_eval_ms * data.n_eval << "," << (t_end_ms - data.t_start_ms) << "," << (data.n_p_eval + data.n_eval) << std::endl;
  perf_file.close();
}
