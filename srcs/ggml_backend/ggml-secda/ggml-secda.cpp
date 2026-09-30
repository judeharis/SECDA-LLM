#include "ggml-secda.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ops_support.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sys/stat.h>
#include <vector>

// ************************************* //
// SECDA_GRAPH_STATS: per-node timing table
// ************************************* //

// With SECDA_GRAPH_STATS set (and not "0"), every node this backend computes is
// timed and written to _gstats/graph_stats.csv, in the 14-column format the old
// fork's GGML_PERF graph print used (run_llama_bench.sh collects it). Only
// SECDA's own nodes are seen here, so Backend is always SECDA; each row is one
// run of the node, so PerfRuns is 1. Graph counts this backend's compute calls
// (one per SECDA split), not whole llama graphs as the old print did. PerfCycles
// is ggml_cycles(). Off by default.
namespace {
struct secda_graph_stats {
  FILE *fp = nullptr;
  int graph = 0;

  static secda_graph_stats *get() {
    static secda_graph_stats *stats = [] {
      const char *env = getenv("SECDA_GRAPH_STATS");
      if (!env || !*env || strcmp(env, "0") == 0) return (secda_graph_stats *)nullptr;
      mkdir("_gstats", 0755);
      FILE *fp = fopen("_gstats/graph_stats.csv", "w");
      if (!fp) {
        GGML_LOG_ERROR("SECDA_GRAPH_STATS: cannot open _gstats/graph_stats.csv\n");
        return (secda_graph_stats *)nullptr;
      }
      fprintf(fp, "Graph,Index,M,N,K,Backend,Op,Src0Type,Src1Type,PerfRuns,"
                  "PerfCycles,PerfCyclesPerRun,PerfTimeUs,PerfTimeUsPerRun\n");
      auto *s = new secda_graph_stats();
      s->fp = fp;
      return s;
    }();
    return stats;
  }

  void row(int index, const struct ggml_tensor *node, int64_t cycles,
           int64_t time_us) {
    const struct ggml_tensor *src0 = node->src[0];
    const struct ggml_tensor *src1 = node->src[1];
    fprintf(fp,
            "%d,%d,%" PRId64 ",%" PRId64 ",%" PRId64 ",SECDA,%s,%s,%s,1,%" PRId64
            ",%.3f,%.3f,%.3f\n",
            graph, index, node->ne[0], node->ne[1], src0 ? src0->ne[0] : 0,
            ggml_op_name(node->op), src0 ? ggml_type_name(src0->type) : "NONE",
            src1 ? ggml_type_name(src1->type) : "NONE", cycles, (double)cycles,
            (double)time_us, (double)time_us);
  }

  void end_graph() {
    fflush(fp);
    graph++;
  }
};
} // namespace

// ************************************* //
// backend interface
// ************************************* //

static const char *ggml_secda_get_name(ggml_backend_t backend) {
  return "SECDA";

  GGML_UNUSED(backend);
}

static void ggml_secda_free(ggml_backend_t backend) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  secda_planner_backend_free(ctx->planner);
  delete ctx;
  delete backend;
}

// Weights are preloaded by the planner (secda_planner.h): on the first compute
// of each scheduler split pass, or here for a standalone (uid 0) graph.
static ggml_backend_graph_plan_t
ggml_secda_graph_plan_create(ggml_backend_t backend,
                             const struct ggml_cgraph *cgraph) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  return secda_planner_plan_create(backend, ctx->planner, cgraph);
}

static void ggml_secda_graph_plan_free(ggml_backend_t backend,
                                       ggml_backend_graph_plan_t plan) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  secda_planner_plan_free(ctx->planner, plan);
}

// Called by ggml_backend_sched on each SECDA split of a split pass: record it
// (the graph is not modified).
static void ggml_secda_graph_optimize(ggml_backend_t backend,
                                      struct ggml_cgraph *cgraph) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  secda_planner_observe(ctx->planner, cgraph);
}

static enum ggml_status ggml_secda_graph_compute(ggml_backend_t backend,
                                                 struct ggml_cgraph *cgraph) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  secda_planner_resolve(backend, ctx->planner, cgraph);
  secda_graph_stats *stats = secda_graph_stats::get();

  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *node = cgraph->nodes[i];
    const bool timed = stats && !ggml_op_is_empty(node->op);
    const int64_t t0 = timed ? ggml_time_us() : 0;
    const int64_t c0 = timed ? ggml_cycles() : 0;

    switch (node->op) {
    case GGML_OP_MUL_MAT:
      secda_planner_set_layer(ctx->planner, node);
      ggml_secda_mul_mat(ctx, node);
      break;

    case GGML_OP_OUT_PROD: ggml_secda_out_prod(ctx, node); break;

#if defined(BFPP_ACC_V4)
    // Jude: Added
    case GGML_OP_SOFT_MAX:
      ggml_secda_soft_max(ctx, node);
      break;
#endif

    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE: break;

    default:
      GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
    }
    if (timed)
      stats->row(i, node, ggml_cycles() - c0, ggml_time_us() - t0);
  }
  if (stats) stats->end_graph();
  secda_planner_end_compute(ctx->planner);

  return GGML_STATUS_SUCCESS;

  GGML_UNUSED(backend);
}

static struct ggml_backend_i secda_backend_i = {
    /* .get_name                = */ ggml_secda_get_name,
    /* .free                    = */ ggml_secda_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_secda_graph_plan_create,
    /* .graph_plan_free         = */ ggml_secda_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_secda_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ ggml_secda_graph_optimize,
};

static ggml_guid_t ggml_secda_guid(void) {

  // SECDA GUID
  static ggml_guid guid = {0x00, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97,
                           0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d};
  return &guid;
}

ggml_backend_t ggml_backend_secda_init(void) {
  initSECDA_ACC();
  ggml_secda_context *ctx = new ggml_secda_context;
  ggml_backend_t backend = new ggml_backend{
      /* .guid      = */ ggml_secda_guid(),
      /* .interface = */ secda_backend_i,
      /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_secda_reg(), 0),
      /* .context   = */ ctx,
  };

  return backend;
}

bool ggml_backend_is_secda(ggml_backend_t backend) {
  return backend != NULL && ggml_guid_matches(backend->guid, ggml_secda_guid());
}

void ggml_backend_secda_set_n_threads(ggml_backend_t backend_secda,
                                      int n_threads) {
  GGML_ASSERT(ggml_backend_is_secda(backend_secda));

  ggml_secda_context *ctx = (ggml_secda_context *)backend_secda->context;
  ctx->n_threads = n_threads;
}

// Standalone (uid 0) graphs are planned on compute unless this is off; a perf
// harness turns it off around its warm-up (reached through
// ggml_backend_reg_get_proc_address, "ggml_backend_secda_set_auto_plan").
void ggml_backend_secda_set_auto_plan(ggml_backend_t backend_secda,
                                      bool enable) {
  GGML_ASSERT(ggml_backend_is_secda(backend_secda));

  ggml_secda_context *ctx = (ggml_secda_context *)backend_secda->context;
  ctx->planner.auto_plan = enable;
}

// ************************************* //
// device interface
// ************************************* //

static const char *ggml_backend_secda_device_get_name(ggml_backend_dev_t dev) {
  return "SECDA";

  GGML_UNUSED(dev);
}

static const char *
ggml_backend_secda_device_get_description(ggml_backend_dev_t dev) {

  return "SECDA";

  GGML_UNUSED(dev);
}

static void ggml_backend_secda_device_get_memory(ggml_backend_dev_t dev,
                                                 size_t *free, size_t *total) {
  // TODO
  *free = 0;
  *total = 0;

  GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type
ggml_backend_secda_device_get_type(ggml_backend_dev_t dev) {
  return GGML_BACKEND_DEVICE_TYPE_ACCEL;

  GGML_UNUSED(dev);
}

static void
ggml_backend_secda_device_get_props(ggml_backend_dev_t dev,
                                    struct ggml_backend_dev_props *props) {
  props->name = ggml_backend_secda_device_get_name(dev);
  props->description = ggml_backend_secda_device_get_description(dev);
  props->type = ggml_backend_secda_device_get_type(dev);
  ggml_backend_secda_device_get_memory(dev, &props->memory_free,
                                       &props->memory_total);
  props->caps = {
      /* .async                 = */ false,
      /* .host_buffer           = */ false,
      /* .buffer_from_host_ptr  = */ true,
      /* .events                = */ false,
  };
}

static ggml_backend_t
ggml_backend_secda_device_init_backend(ggml_backend_dev_t dev,
                                       const char *params) {
  return ggml_backend_secda_init();

  GGML_UNUSED(dev);
  GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t
ggml_backend_secda_device_get_buffer_type(ggml_backend_dev_t dev) {
  return ggml_backend_cpu_buffer_type();

  GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_secda_device_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size) {
  return ggml_backend_cpu_buffer_from_ptr(ptr, size);

  GGML_UNUSED(dev);
  GGML_UNUSED(max_tensor_size);
}

static bool
ggml_backend_secda_device_supports_op(ggml_backend_dev_t dev,
                                      const struct ggml_tensor *op) {
  switch (op->op) {

  case GGML_OP_MUL_MAT: {
    const struct ggml_tensor *src0 = op->src[0];
    const struct ggml_tensor *src1 = op->src[1];

    bool s0_con = ggml_is_contiguous(src0);
    bool s1_con = ggml_is_contiguous(src1);
    bool supp_q2 = false;
    bool supp_q3 = false;
    bool supp_q4 = false;
    bool supp_q5 = false;
    bool supp_q6 = false;

#ifdef GGML_SECDA_QK2
    // SECDA_COUT << "Support Q2" << std::endl;
    supp_q2 = true;
#endif
#ifdef GGML_SECDA_QK3
    // SECDA_COUT << "Support Q3" << std::endl;
    supp_q3 = true;
#endif
#ifdef GGML_SECDA_QK4
    // SECDA_COUT << "Support Q4" << std::endl;
    supp_q4 = true;
#endif
#ifdef GGML_SECDA_QK5
    // SECDA_COUT << "Support Q5" << std::endl;
    supp_q5 = true;
#endif
#ifdef GGML_SECDA_QK6
    // SECDA_COUT << "Support Q6" << std::endl;
    supp_q6 = true;
#endif

    bool q2 = (src0->type == GGML_TYPE_Q2_K) && (supp_q2);
    bool q3 = (src0->type == GGML_TYPE_Q3_K) && (supp_q3);
    bool q4 = (src0->type == GGML_TYPE_Q4_K) && (supp_q4);
    bool q5 = (src0->type == GGML_TYPE_Q5_K) && (supp_q5);
    bool q6 = (src0->type == GGML_TYPE_Q6_K) && (supp_q6);
    bool s0_type = (q2 || q3 || q4 || q5 || q6);
    // bool s0_type = (q2 || q3);
    // bool s0_type = (q2);
    // bool s0_type = (q3);

    bool s1_type = src1->type == GGML_TYPE_F32;
    bool is_supported = s0_con && s1_con && s1_type && s0_type;
    if (!is_supported) return false;

    // Dimension checks
    // int K = src1->ne[0];
    // int M = src0->ne[0];
    // int N = op->ne[1];

    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t K = src1->ne[0];

    bool dim_ok = dim_check(M, N, K);
    // if (!dim_ok) {
    //   SECDA_COUT << "SECDA: Dimension check failed for M=" << M << ", N=" <<
    //   N
    //             << ", K=" << K << std::endl;
    // } else {
    //   SECDA_COUT << "SECDA: Dimension check passed for M=" << M << ", N=" <<
    //   N
    //             << ", K=" << K << std::endl;
    // }
    is_supported = is_supported && dim_ok;

    // Multiple independent weight slices packed into src0's ne[2]/ne[3]
    // (e.g. MoE-style per-expert weights) are not supported: the preload
    // cache and compute path only ever hold a single 2D weight matrix, so
    // both driver variants silently compute wrong output for these.
    bool s0_batched = (src0->ne[2] > 1) || (src0->ne[3] > 1);
    is_supported = is_supported && !s0_batched;

#if !defined(SECDA_DRIVER_BATCHES)
    // The non-batch-capable driver has no ne2/ne3 loop at all - it only
    // computes a single 2D slice, so broadcasting src0 across multiple
    // src1/dst slices (nr != [1,1], e.g. GQA-style broadcast) silently
    // leaves the other slices uncomputed. driver_batches handles this
    // correctly, so only reject it here for the non-batch driver.
    bool broadcast =
        (src1->ne[2] != src0->ne[2]) || (src1->ne[3] != src0->ne[3]);
    is_supported = is_supported && !broadcast;
#endif

    // return false;
    return is_supported;
  }

#if defined(BFPP_ACC_V4)
  // Jude: Added - the V4 hardware softmax datapath tiles the row (see
  // acc_softmax_unit.sc.h), so unlike MUL_MAT there is no upper bound on
  // ne0/row width here. Gate on dtype/contiguity/mask-type only.
  case GGML_OP_SOFT_MAX: {
    const struct ggml_tensor *src0 = op->src[0]; // logits
    const struct ggml_tensor *src1 = op->src[1]; // mask (optional)
    const struct ggml_tensor *src2 = op->src[2]; // sinks (optional)

    bool ok = ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
              src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
    if (src1) {
      ok = ok && ggml_is_contiguous(src1) &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    }
    if (src2) {
      ok = ok && src2->type == GGML_TYPE_F32;
    }
    // Profitability floor (docs/softmax_plan.md S5.2: "prefer CPU for tiny
    // rows"). Left inert (>=1) for now - Stage C wants small-ne0 correctness
    // cases exercised, not skipped; raise this later from measured
    // crossover data.
    const int64_t SECDA_SOFTMAX_MIN_N = 1;
    ok = ok && src0->ne[0] >= SECDA_SOFTMAX_MIN_N;
    return ok;
  }
#endif

  default: return false;
  }

  GGML_UNUSED(dev);
}

// Jude: Comment
// There is a check which checks if the device supports the buffer type
// Right now, the only buffer type is host buffer, so we can just return true
// But in the future, we might want to make sure the data is move to DMA buffer
static bool
ggml_backend_secda_device_supports_buft(ggml_backend_dev_t dev,
                                        ggml_backend_buffer_type_t buft) {
  return ggml_backend_buft_is_host(buft);

  GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_secda_device_i = {
    /* .get_name             = */ ggml_backend_secda_device_get_name,
    /* .get_description      = */ ggml_backend_secda_device_get_description,
    /* .get_memory           = */ ggml_backend_secda_device_get_memory,
    /* .get_type             = */ ggml_backend_secda_device_get_type,
    /* .get_props            = */ ggml_backend_secda_device_get_props,
    /* .init_backend         = */ ggml_backend_secda_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_secda_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */
    ggml_backend_secda_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_secda_device_supports_op,
    /* .supports_buft        = */ ggml_backend_secda_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ************************************* //
// backend reg interface
// ************************************* //

static const char *ggml_secda_reg_get_name(ggml_backend_reg_t reg) {
  return "SECDA";

  GGML_UNUSED(reg);
}

static size_t ggml_secda_reg_get_device_count(ggml_backend_reg_t reg) {
  return 1;

  GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_secda_reg_get_device(ggml_backend_reg_t reg,
                                                    size_t index) {
  GGML_ASSERT(index == 0);

  static ggml_backend_device ggml_backend_secda_device = {
      /* .iface   = */ ggml_backend_secda_device_i,
      /* .reg     = */ reg,
      /* .context = */ nullptr,
  };

  return &ggml_backend_secda_device;

  GGML_UNUSED(reg);
  GGML_UNUSED(index);
}

static void *ggml_secda_get_proc_address(ggml_backend_reg_t reg,
                                         const char *name) {
  if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
    return (void *)ggml_backend_secda_set_n_threads;
  }
  if (std::strcmp(name, "ggml_backend_secda_set_auto_plan") == 0) {
    return (void *)ggml_backend_secda_set_auto_plan;
  }
  return NULL;

  GGML_UNUSED(reg);
  GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_secda_reg_i = {
    /* .get_name         = */ ggml_secda_reg_get_name,
    /* .get_device_count = */ ggml_secda_reg_get_device_count,
    /* .get_device       = */ ggml_secda_reg_get_device,
    /* .get_proc_address = */ ggml_secda_get_proc_address,
};

ggml_backend_reg_t ggml_backend_secda_reg(void) {
  static struct ggml_backend_reg ggml_secda_reg = {
      /* .api_version = */ GGML_BACKEND_API_VERSION,
      /* .iface   = */ ggml_secda_reg_i,
      /* .context = */ NULL,
  };

  return &ggml_secda_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_secda_reg)

// ************************************* //