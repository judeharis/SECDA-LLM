// Weight-preload planning for the SECDA backend; see secda_planner.h.

#include "secda_planner.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ops_support.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// The plan and its exit summary, as they were in ggml-secda.cpp. A
// namespace-scope static, constructed at library load and so destroyed after
// the driver's function-local statics: the "SECDA Plan" summary still prints
// after the DMA and acc_times blocks at exit (benchmark/scripts/parse_results.py
// relies on that order).
struct ggml_backend_plan_secda {
  int supported_nodes = 0;
  int preloaded_nodes = 0;
  bool planned = false;
  int plan_counter = 0;
  int plan_reused = 0;

  void reset() {
    if (planned) {
      SECDA_COUT
          << "================================================================"
          << std::endl;
      SECDA_COUT << "SECDA Plan: " << plan_counter << " Reused: " << plan_reused
                 << " Supported nodes: " << supported_nodes
                 << " Preloaded nodes: " << preloaded_nodes << std::endl;
      SECDA_COUT
          << "================================================================"
          << std::endl;

      supported_nodes = 0;
      preloaded_nodes = 0;
      planned = false;
      plan_reused = 0;
    }
  }

  ~ggml_backend_plan_secda() {
    SECDA_COUT
        << "================================================================"
        << std::endl;
    SECDA_COUT << "SECDA Plan: " << plan_counter << " Reused: " << plan_reused
               << " Supported nodes: " << supported_nodes
               << " Preloaded nodes: " << preloaded_nodes << std::endl;
    SECDA_COUT
        << "================================================================"
        << std::endl;
  }
};

static struct ggml_backend_plan_secda secda_plan;

// Pass ids are unique across backends and contexts.
static uint64_t g_pass_counter = 0;

// Which graph the driver's current plan belongs to.
static struct {
  enum { NONE, PASS, UID, STANDALONE } kind = NONE;
  const secda_planner_state *st = nullptr;
  uint64_t key = 0; // pass id or uid
} g_owner;

// A standalone graph planned through graph_plan_create: computing it again
// must not replan (test-backend-ops perf mode times the planned graph).
static struct {
  bool active = false;
  const secda_planner_state *st = nullptr;
  const ggml_cgraph *g = nullptr;
  ggml_tensor **nodes = nullptr;
  int n_nodes = 0;
  std::vector<const ggml_tensor *> node_ptrs;
  std::vector<const void *> src0; // MUL_MAT src[0]->data, else nullptr
} g_sticky;

// The layer of each planned MUL_MAT: its ordinal among the plan's MUL_MATs,
// which is also the index preload_weights_alloc filed its weights under. The
// driver is told it before each MUL_MAT (secda_planner_set_layer), so SOFT_MAX
// nodes, skipped nodes or an evaluation that stops part-way can't shift it.
struct secda_layer {
  int layer;
  const void *src0; // weights planned (and preloaded) for this node
};
static struct {
  bool active = false; // the graph being computed is the planned one
  bool warned = false;
  std::unordered_map<const ggml_tensor *, secda_layer> of;
} g_layers;

// Plan the supported nodes of spans[0..n_spans), in order, as one graph: the
// body of the old ggml_secda_graph_plan_create across all spans.
static void secda_plan_spans(ggml_backend_t backend, const secda_span *spans,
                             int n_spans) {
  g_layers.of.clear();
  g_layers.warned = false;
  secda_plan.reset();
  resetPlan_T();
  SECDA_COUT << std::endl;
  SECDA_COUT
      << "================================================================"
      << std::endl;
  SECDA_COUT << "SECDA Graph Plan Create" << std::endl;

  // Code to to preload weights
  // Need to adapt to support different MatMul Quantization types
  int layer = 0; // MUL_MAT ordinal

#ifdef SECDA_LOG
  std::ofstream plans_file("_plans/plans" +
                               std::to_string(secda_plan.plan_counter) + ".csv",
                           std::ios::out);
  plans_file << "plan_count,layer,M,K,N,wgt_type,weight_size,preloaded"
             << std::endl;
#endif

  for (int s = 0; s < n_spans; s++) {
  for (int i = 0; i < spans[s].n_nodes; i++) {
    struct ggml_tensor *node = spans[s].nodes[i];
    bool node_supported = ggml_backend_supports_op(backend, node);
    if (node_supported) {
      // Only MUL_MAT nodes have weights to preload and a layer; a SOFT_MAX
      // node's src0/src1 are logits/mask. SOFT_MAX still counts in
      // supported_nodes (the plan summary and the driver's counter wrap).
      if (node->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor *src0 = node->src[0];
        const struct ggml_tensor *src1 = node->src[1];
        const enum ggml_type type = src0->type;

        const int64_t K = src1->ne[0];
        const int64_t M = node->ne[0];

        int wgt_type = 3;
        if (type == GGML_TYPE_Q6_K) wgt_type = 6;
        if (type == GGML_TYPE_Q5_K) wgt_type = 5;
        if (type == GGML_TYPE_Q4_K) wgt_type = 4;
        if (type == GGML_TYPE_Q3_K) wgt_type = 3;
        if (type == GGML_TYPE_Q2_K) wgt_type = 2;

        int64_t weight_size = 0;
        if (wgt_type == 6) weight_size = M * (K / 256) * sizeof(block_q6_K) + 64;
        if (wgt_type == 5) weight_size = M * (K / 256) * sizeof(block_q5_K) + 64;
        if (wgt_type == 4) weight_size = M * (K / 256) * sizeof(block_q4_K) + 64;
        if (wgt_type == 3) weight_size = M * (K / 256) * sizeof(block_q3_K) + 64;
        if (wgt_type == 2) weight_size = M * (K / 256) * sizeof(block_q2_K) + 64;

        bool preloaded =
            preload_weights_alloc(weight_size, layer, M, K, src0->data, wgt_type);
#ifdef SECDA_LOG
        const int64_t N = src1->ne[1];
        plans_file << secda_plan.plan_counter << "," << layer << "," << M << ","
                   << K << "," << N << "," << wgt_type << "," << weight_size
                   << "," << (preloaded ? 1 : 0) << std::endl;
#endif
        if (preloaded) secda_plan.preloaded_nodes++;
        g_layers.of[node] = {layer, src0->data};
        layer++;
      }
      secda_plan.supported_nodes++;
    }
  }
  }
#ifdef SECDA_LOG
  plans_file.close();
#endif
  updatePlan_T(secda_plan.supported_nodes);

  SECDA_COUT << "SECDA Supported nodes: " << secda_plan.supported_nodes
             << " Preloaded nodes: " << secda_plan.preloaded_nodes << std::endl;
  SECDA_COUT
      << "================================================================"
      << std::endl;
  secda_plan.planned = true;
  secda_plan.plan_counter++;
  g_layers.active = true;
}

static bool sticky_matches(const secda_planner_state &st, const ggml_cgraph *g) {
  if (g_sticky.st != &st || g_sticky.g != g || g_sticky.nodes != g->nodes ||
      g_sticky.n_nodes != g->n_nodes)
    return false;
  for (int i = 0; i < g->n_nodes; i++) {
    const ggml_tensor *node = g->nodes[i];
    if (g_sticky.node_ptrs[i] != node) return false;
    const void *s0 = node->op == GGML_OP_MUL_MAT && node->src[0]
                         ? node->src[0]->data
                         : nullptr;
    if (g_sticky.src0[i] != s0) return false;
  }
  return true;
}

void secda_planner_observe(secda_planner_state &st, const ggml_cgraph *g) {
  secda_pass &pass = st.pass;
  const void *parent = g->visited_hash_set.keys;
  bool new_pass = st.computed_since_optimize || pass.id == 0 ||
                  parent != pass.parent ||
                  (!pass.splits.empty() &&
                   g->nodes < pass.splits.back().nodes + pass.splits.back().n_nodes);
  if (new_pass) {
    pass.splits.clear();
    pass.id = ++g_pass_counter;
    pass.parent = parent;
    pass.first = -1;
  }
  pass.splits.push_back({g->nodes, g->n_nodes});
  st.computed_since_optimize = false;
}

void secda_planner_resolve(ggml_backend_t backend, secda_planner_state &st,
                           const ggml_cgraph *g) {
  secda_pass &pass = st.pass;

  // A split of the recorded pass: exact for scheduler splits, containment for
  // the sub-views the scheduler computes when an eval callback is set.
  int k = -1;
  if (g->visited_hash_set.keys == pass.parent) {
    for (int s = 0; s < (int)pass.splits.size(); s++) {
      const secda_span &sp = pass.splits[s];
      if (sp.nodes <= g->nodes && g->nodes + g->n_nodes <= sp.nodes + sp.n_nodes) {
        k = s;
        break;
      }
    }
  }
  if (k >= 0) {
    bool owned = g_owner.kind == g_owner.PASS && g_owner.st == &st &&
                 g_owner.key == pass.id && k >= pass.first;
    if (!owned) {
      pass.first = k;
      secda_plan_spans(backend, &pass.splits[k], (int)pass.splits.size() - k);
      g_owner.kind = g_owner.PASS;
      g_owner.st = &st;
      g_owner.key = pass.id;
      g_sticky.active = false;
    } else if (k == pass.first && g->nodes == pass.splits[k].nodes) {
      secda_plan.plan_reused++; // once per evaluation of a reused graph
    }
    g_layers.active = true;
    return;
  }
  g_layers.active = false;

  if (g->uid != 0) {
    // A scheduler split graph_optimize never saw: plan per split (correct, but
    // every split re-preloads). Only if upstream stops calling graph_optimize.
    if (!st.warned_unrecorded) {
      std::cerr << "SECDA WARNING: split not recorded by graph_optimize; "
                   "planning per split"
                << std::endl;
      st.warned_unrecorded = true;
    }
    if (!(g_owner.kind == g_owner.UID && g_owner.st == &st && g_owner.key == g->uid)) {
      secda_span sp = {g->nodes, g->n_nodes};
      secda_plan_spans(backend, &sp, 1);
      g_owner.kind = g_owner.UID;
      g_owner.st = &st;
      g_owner.key = g->uid;
      g_sticky.active = false;
    } else {
      secda_plan.plan_reused++;
      g_layers.active = true;
    }
    return;
  }

  // Standalone uid-0 graph (test-backend-ops; no scheduler).
  g_layers.active = false;
  if (g_sticky.active && sticky_matches(st, g)) {
    g_layers.active = true;
    return;
  }
  // Perf warm-up: runs unplanned, sending the weights with each call.
  if (!st.auto_plan) return;
  secda_span sp = {g->nodes, g->n_nodes};
  secda_plan_spans(backend, &sp, 1);
  g_owner.kind = g_owner.STANDALONE;
  g_owner.st = &st;
  g_owner.key = 0;
  g_sticky.active = false;
}

void secda_planner_set_layer(secda_planner_state &st, const ggml_tensor *node) {
  GGML_UNUSED(st);
  int layer = -1;
  if (g_layers.active) {
    auto it = g_layers.of.find(node);
    if (it == g_layers.of.end()) {
      if (!g_layers.warned)
        std::cerr << "SECDA WARNING: " << node->name
                  << " is not in the plan; sending its weights with the call"
                  << std::endl;
      g_layers.warned = true;
    } else if (it->second.src0 != node->src[0]->data) {
      if (!g_layers.warned)
        std::cerr << "SECDA WARNING: " << node->name
                  << " weights moved since planning; sending them with the call"
                  << std::endl;
      g_layers.warned = true;
    } else {
      layer = it->second.layer;
    }
  }
  setLayer_T(layer);
}

void secda_planner_end_compute(secda_planner_state &st) {
  st.computed_since_optimize = true;
}

ggml_backend_graph_plan_t secda_planner_plan_create(ggml_backend_t backend,
                                                    secda_planner_state &st,
                                                    const ggml_cgraph *g) {
  // Scheduler graphs are planned on compute; an explicit plan for one is a
  // no-op (it keeps an older llama.cpp that still calls plan_create harmless).
  if (g->uid != 0) return &secda_plan;
  secda_span sp = {g->nodes, g->n_nodes};
  secda_plan_spans(backend, &sp, 1);
  g_owner.kind = g_owner.STANDALONE;
  g_owner.st = &st;
  g_owner.key = 0;
  g_sticky.active = true;
  g_sticky.st = &st;
  g_sticky.g = g;
  g_sticky.nodes = g->nodes;
  g_sticky.n_nodes = g->n_nodes;
  g_sticky.node_ptrs.assign(g->nodes, g->nodes + g->n_nodes);
  g_sticky.src0.resize(g->n_nodes);
  for (int i = 0; i < g->n_nodes; i++) {
    const ggml_tensor *node = g->nodes[i];
    g_sticky.src0[i] = node->op == GGML_OP_MUL_MAT && node->src[0]
                           ? node->src[0]->data
                           : nullptr;
  }
  return &secda_plan;
}

void secda_planner_plan_free(secda_planner_state &st,
                             ggml_backend_graph_plan_t plan) {
  if (plan == (ggml_backend_graph_plan_t)&secda_plan && g_sticky.st == &st)
    g_sticky.active = false;
}

void secda_planner_backend_free(secda_planner_state &st) {
  if (g_owner.st == &st) {
    g_owner.kind = g_owner.NONE;
    g_owner.st = nullptr;
    g_owner.key = 0;
  }
  if (g_sticky.st == &st) {
    g_sticky.active = false;
    g_sticky.st = nullptr;
  }
  g_layers.active = false;
}
