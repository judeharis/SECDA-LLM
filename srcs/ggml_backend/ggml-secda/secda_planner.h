#ifndef SECDA_PLANNER_H
#define SECDA_PLANNER_H

// Weight-preload planning for the SECDA backend, without any hook in llama.cpp.
//
// The driver preloads every supported MUL_MAT's weights into the accelerator's
// DMA buffers once per graph, filed by layer: the MUL_MAT's ordinal in the
// plan. That plan used to be created by a call in the llama.cpp fork
// (llama-context.cpp) on the whole graph. Here the backend builds it itself:
//
//  - ggml_backend_sched calls the backend's graph_optimize on each SECDA split of
//    a split pass (ggml-backend.cpp, before the splits get their uids), so
//    secda_planner_observe() records the pass's SECDA splits;
//  - the first graph_compute of that pass (secda_planner_resolve) runs the plan
//    over those splits, in order: the same nodes, in the same order, as the old
//    whole-graph plan whenever the scheduler puts every supported node on SECDA;
//  - graph reuse (no re-split) keeps the plan and the preloaded weights;
//  - a standalone graph (uid 0, no scheduler: test-backend-ops) is planned on
//    compute, unless an explicit graph_plan_create already planned it;
//  - each planned MUL_MAT gets a layer, its ordinal among the plan's MUL_MATs,
//    and the driver is told it before the node runs (secda_planner_set_layer).
//    SOFT_MAX nodes and incomplete evaluations therefore can't shift which
//    preloaded weights a MUL_MAT uses.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;

struct secda_span {
  ggml_tensor **nodes;
  int n_nodes;
};

struct secda_pass {
  uint64_t id = 0;
  const void *parent = nullptr; // parent graph: cgraph->visited_hash_set.keys
  std::vector<secda_span> splits;
  int first = -1; // index of the split the committed plan starts at
};

struct secda_planner_state {
  secda_pass pass;
  bool computed_since_optimize = true;
  bool auto_plan = true;
  bool warned_unrecorded = false;
};

// graph_optimize: record one SECDA split of a scheduler pass (never modifies g).
void secda_planner_observe(secda_planner_state &st, const ggml_cgraph *g);

// Top of graph_compute: make sure the driver holds the plan for g's graph.
void secda_planner_resolve(ggml_backend_t backend, secda_planner_state &st,
                           const ggml_cgraph *g);

// Before each MUL_MAT compute: tell the driver the node's planned layer, or -1
// (weights sent with the call) if it isn't planned or its weights moved.
void secda_planner_set_layer(secda_planner_state &st, const ggml_tensor *node);

// After graph_compute's node loop.
void secda_planner_end_compute(secda_planner_state &st);

// Explicit plan API (graph_plan_create / graph_plan_free).
ggml_backend_graph_plan_t secda_planner_plan_create(ggml_backend_t backend,
                                                    secda_planner_state &st,
                                                    const ggml_cgraph *g);
void secda_planner_plan_free(secda_planner_state &st,
                             ggml_backend_graph_plan_t plan);

// Backend free: drop any global reference to this backend's planner state.
void secda_planner_backend_free(secda_planner_state &st);

#endif // SECDA_PLANNER_H
