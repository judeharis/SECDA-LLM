
#ifndef SECDA_OPS_SUPPORT_H
#define SECDA_OPS_SUPPORT_H

#include <algorithm>

#include "ggml-impl.h"
#include "ggml-secda.h"

// #define GGML_COMMON_DECL_C
#include "ggml-quants.h"

// #define SECDA_LOG
#ifdef SECDA_BACKEND_PRINT
#define SECDA_COUT std::cout
#else
#define SECDA_COUT                                                             \
  if (false) std::cout
#endif

bool dim_check(int M, int N, int K);

void initSECDA_ACC();

void resetPlan_T();

void updatePlan_T(int supported_nodes);

bool modelPlanned_T();

bool preload_weights_alloc(unsigned wgt_size, int layer, int M, int K,
                    const void *wgt, int wgt_type);

void ggml_secda_mul_mat(ggml_secda_context *ctx, struct ggml_tensor *node);

void ggml_secda_out_prod(ggml_secda_context *ctx, struct ggml_tensor *node);

typedef void (*ggml_from_float_to_mat_t)(const float *GGML_RESTRICT x,
                                         void *GGML_RESTRICT y, int64_t nr,
                                         int64_t k, int64_t bs);
typedef void (*ggml_vec_dot_t)(int n, float *GGML_RESTRICT s, size_t bs,
                               const void *GGML_RESTRICT x, size_t bx,
                               const void *GGML_RESTRICT y, size_t by, int nrc);
typedef void (*ggml_gemv_t)(int n, float *GGML_RESTRICT s, size_t bs,
                            const void *GGML_RESTRICT x,
                            const void *GGML_RESTRICT y, int nr, int nc);
typedef void (*ggml_gemm_t)(int n, float *GGML_RESTRICT s, size_t bs,
                            const void *GGML_RESTRICT x,
                            const void *GGML_RESTRICT y, int nr, int nc);

#endif // SECDA_OPS_SUPPORT_H