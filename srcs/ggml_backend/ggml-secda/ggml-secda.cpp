#include "ggml-secda.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ops_support.h"

#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <vector>



struct ggml_backend_plan_secda {
  int supported_nodes = 0;
  int preloaded_nodes = 0;
  bool planned = false;
  int plan_counter = 0;
  int plan_reused = 0;
  uint64_t graph_uid = 0;

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
      graph_uid = 0;
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

// ************************************* //
// backend interface
// ************************************* //

static const char *ggml_secda_get_name(ggml_backend_t backend) {
  return "SECDA";

  GGML_UNUSED(backend);
}

static void ggml_secda_free(ggml_backend_t backend) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;
  delete ctx;
  delete backend;
}

static ggml_backend_graph_plan_t
ggml_secda_graph_plan_create(ggml_backend_t backend,
                             const struct ggml_cgraph *cgraph) {

  if (secda_plan.planned && secda_plan.graph_uid == cgraph->uid) {
    secda_plan.plan_reused++;
    return &secda_plan;
  }
  // if (secda_plan.planned) return &secda_plan;
  secda_plan.reset();
  secda_plan.graph_uid = cgraph->uid;
  resetPlan_T();
  SECDA_COUT << std::endl;
  SECDA_COUT
      << "================================================================"
      << std::endl;
  SECDA_COUT << "SECDA Graph Plan Create" << std::endl;

  // Code to to preload weights
  // Need to adapt to support different MatMul Quantization types
  int layer = 0;

#ifdef SECDA_LOG
  std::ofstream plans_file("_plans/plans" +
                               std::to_string(secda_plan.plan_counter) + ".csv",
                           std::ios::out);
  plans_file << "plan_count,layer,M,K,N,wgt_type,weight_size,preloaded"
             << std::endl;
#endif

  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *node = cgraph->nodes[i];
    bool node_supported = ggml_backend_supports_op(backend, node);
    if (node_supported) {
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
      secda_plan.supported_nodes++;
      layer++;
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
  return &secda_plan;
}

static void secda_graph_compute_perf_stats_node(struct ggml_tensor *node,
                                                int s_cycles,
                                                int64_t s_time_us) {
  int64_t cycles_cur = ggml_cycles() - s_cycles;
  int64_t time_us_cur = ggml_time_us() - s_time_us;

  node->perf_runs++;
  node->perf_cycles += cycles_cur;
  node->perf_time_us += time_us_cur;
  node->isSECDA = 1;
}

static enum ggml_status ggml_secda_graph_compute(ggml_backend_t backend,
                                                 struct ggml_cgraph *cgraph) {
  ggml_secda_context *ctx = (ggml_secda_context *)backend->context;

  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *node = cgraph->nodes[i];
    int64_t perf_node_start_cycles = ggml_cycles();
    int64_t perf_node_start_time_us = ggml_time_us();

    switch (node->op) {
    case GGML_OP_MUL_MAT: ggml_secda_mul_mat(ctx, node); break;

    case GGML_OP_OUT_PROD: ggml_secda_out_prod(ctx, node); break;

    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE: break;

    default:
      GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
    }
    secda_graph_compute_perf_stats_node(node, perf_node_start_cycles,
                                        perf_node_start_time_us);
  }

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
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_secda_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
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

    // return false;
    return is_supported;
  }

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