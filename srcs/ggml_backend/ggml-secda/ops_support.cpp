#include "ops_support.h"

// Jude: Added BFPP_ACC_V4 branch (checked first, matching the CMake priority
// chain) - V4 forks from V3's plain driver/, adding hardware softmax. Like
// V3, V4 also has a driver_batches variant (GQA-broadcast MUL_MAT); softmax
// itself doesn't need a batched variant (ggml_secda_soft_max already handles
// ne02/ne03 via its own row loop), so driver_batches only changes EntryMM.
#if defined(BFPP_ACC_V4) && defined(SECDA_DRIVER_BATCHES)
#include "acc_dels/bfpp_acc/v4/accelerator/driver_batches/acc_driver_connector.h"
#elif defined(BFPP_ACC_V4)
#include "acc_dels/bfpp_acc/v4/accelerator/driver/acc_driver_connector.h"
#elif defined(BFPP_ACC_V3) && defined(SECDA_DRIVER_BATCHES)
#include "acc_dels/bfpp_acc/v3/accelerator/driver_batches/acc_driver_connector.h"
#elif defined(BFPP_ACC_V3)
#include "acc_dels/bfpp_acc/v3/accelerator/driver/acc_driver_connector.h"
#else
#error "ggml-secda needs BFPP_ACC_V3 or BFPP_ACC_V4 (v1/v2 are retired to acc_dels/bfpp_acc/legacy/)"
#endif

#include "ggml-quants.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>


bool preload_weights_alloc(unsigned wgt_size, int layer, int M, int K,
                           const void *wgt, int wgt_type) {
  return preloadWeights(wgt_size, layer, M, K, wgt, wgt_type);
}

bool dim_check(int M, int N, int K) { return checkDim(M, N, K); }

void initSECDA_ACC() { initACC(); }

void resetPlan_T() { resetPlan(); }

void updatePlan_T(int supported_nodes) { updatePlan(supported_nodes); }

bool modelPlanned_T() { return modelPlanned(); }

void ggml_secda_mul_mat(ggml_secda_context *ctx, struct ggml_tensor *dst) {

  auto start_1 = std::chrono::high_resolution_clock::now();
  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];
  GGML_TENSOR_BINARY_OP_LOCALS

  const enum ggml_type type = src0->type;

  const int64_t M = ne0;
  const int64_t N = ne1;
  const int64_t K = ne10;

  int wgt_stride = nb01;
  int out_stride = nb1;
  int wgt_type = 3;
  if (type == GGML_TYPE_Q6_K) wgt_type = 6;
  if (type == GGML_TYPE_Q5_K) wgt_type = 5;
  if (type == GGML_TYPE_Q4_K) wgt_type = 4;
  if (type == GGML_TYPE_Q3_K) wgt_type = 3;
  if (type == GGML_TYPE_Q2_K) wgt_type = 2;

  // llama.cpp's batched graph execution (e.g. llama-perplexity) can produce
  // degenerate zero-token nodes (ne1 == 0). ggml's stride invariant
  // nb[i] <= nb[i+1] assumes ne[i] >= 1, so the asserts and Q8_K packing
  // below would misbehave for these - route straight into EntryMM with N=0
  // instead of skipping it outright. EntryMM still needs to run (it's a
  // no-op on the hardware side for N=0) so its per-node "layer" counter
  // advances/wraps in step with ggml_secda_graph_plan_create's
  // supported_nodes count; skipping the call entirely leaves that counter
  // one node behind for the rest of the run, which corrupts LoadWeights'
  // preloaded-weight framing for a completely unrelated layer on the next
  // real call.
  if (ggml_nelements(dst) == 0) {
#if defined(SECDA_DRIVER_BATCHES)
    EntryMM(src0->data, nullptr, dst->data, M, 0, K, 0, wgt_stride, out_stride,
            wgt_type, (int)ne2, (int)ne3, 0, 0, (int)nb2, (int)nb3);
#else
    EntryMM(src0->data, nullptr, dst->data, M, 0, K, 0, wgt_stride, out_stride,
            wgt_type);
#endif
    updateProfile(std::chrono::high_resolution_clock::now() - start_1);
    return;
  }

  enum ggml_type const vec_dot_type = GGML_TYPE_Q8_K;

  GGML_ASSERT(nb00 == ggml_type_size(type));

  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_K, ne10);

  const size_t desired_wsize =
      ggml_row_size(vec_dot_type, ggml_nelements(src1));

  if (ctx->work_size < desired_wsize) {
    ctx->work_data.reset(new char[desired_wsize]);
    ctx->work_size = desired_wsize;
  }

  void *twdata = ctx->work_data.get();

  // #ifdef GGML_SECDA_ARM
  //   float *inp_pointer = new float[N * K];
  //   if (load) {
  //     // load inputs from csv
  //     std::ifstream inp_file;
  //     inp_file.open("_aData/cgpt/in1.csv");
  //     for (int r = 0; r < N * K; r++) {
  //       inp_file >> inp_pointer[r];
  //     }
  //     inp_file.close();
  //   }
  // #endif

  if (src1->type != GGML_TYPE_Q8_K) {
    char *wdata = (char *)twdata;

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);
    const size_t nbw2 = nbw1 * ne11;
    const size_t nbw3 = nbw2 * ne12;

    assert(desired_wsize >= ne13 * nbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
      for (int64_t i12 = 0; i12 < ne12; ++i12) {
        int64_t i11_processed = 0;
        for (int64_t i11 = i11_processed; i11 < ne11; i11 += 1) {
          // #ifdef GGML_SECDA_ARM
          //           if (load) {
          //             quantize_row_q8_K(
          //                 (float *)((char *)inp_pointer + i13 * nb13 + i12 *
          //                 nb12 +
          //                           i11 * nb11),
          //                 (void *)(wdata + i13 * nbw3 + i12 * nbw2 + i11 *
          //                 nbw1), ne10);
          //             load = false;
          //           } else {
          //             quantize_row_q8_K(
          //                 (float *)((char *)src1->data + i13 * nb13 + i12 *
          //                 nb12 +
          //                           i11 * nb11),
          //                 (void *)(wdata + i13 * nbw3 + i12 * nbw2 + i11 *
          //                 nbw1), ne10);
          //           }
          // #else
          quantize_row_q8_K_ref(
              (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 +
                        i11 * nb11),
              (block_q8_K *)(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
              ne10);
          // #endif
        }
      }
    }
  }

  // #define PRINT_FILE
  // #ifdef PRINT_FILE
  //   std::ofstream myfile2;
  //   myfile2.open("_aData/cgpt/wdata.csv");
  //   int *res_pointer2 = (int *)twdata;
  //   for (int r = 0; r < N * K; r++) {
  //     myfile2 << (int)res_pointer2[r] << std::endl;
  //   }
  //   myfile2.close();

  //   #ifndef GGML_SECDA_ARM
  //     std::ofstream myfile;
  //     myfile.open("_aData/cgpt/in1.csv");
  //     float *res_pointer = (float *)src1->data;
  //     for (int r = 0; r < N * K; r++) {
  //       myfile << (float)res_pointer[r] << std::endl;
  //     }
  //     myfile.close();
  //     exit(0);
  //   #endif
  // #endif

#if defined(SECDA_DRIVER_BATCHES)
  // twdata was packed above as a [ne13][ne12][ne11][row_size] buffer, so its
  // own batch strides mirror dst's ne2/ne3 one-to-one (ggml_mul_mat
  // broadcasts src0 across src1's/dst's ne2/ne3).
  const int64_t inp_stride2 = (int64_t)row_size * ne11;
  const int64_t inp_stride3 = inp_stride2 * ne12;
  EntryMM(src0->data, twdata, dst->data, M, N, K, row_size, wgt_stride,
          out_stride, wgt_type, (int)ne2, (int)ne3, (int)inp_stride2,
          (int)inp_stride3, (int)nb2, (int)nb3);
#else
  EntryMM(src0->data, twdata, dst->data, M, N, K, row_size, wgt_stride,
          out_stride, wgt_type);
#endif
  auto end1 = std::chrono::high_resolution_clock::now();
  auto time1 = end1 - start_1;
  updateProfile(time1);
  // double elapsed_time =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(time1).count();
  // updateProfile(elapsed_time);

  // const int M = ne01;
  // const int K = ne00;
  // const int N = ne11;
  // printf("ggml_compute_forward_mul_mat: M = %ld, K = %ld, N = %ld, qtype = %d\n", M, K, N, wgt_type);
  // const char *type_name = ggml_type_name(src0->type);
  // if (type_name != NULL && strcmp(type_name, "f32") != 0) {
  //   char base_name[512];
  //   char file_counter_chars[16];
  //   snprintf(file_counter_chars, sizeof(file_counter_chars), "%d",
  //            file_counter++);
  //   snprintf(base_name, sizeof(base_name), "%s_%s", file_counter_chars,
  //            type_name != NULL ? type_name : "dst");
  //   char csv_name[512];
  //   snprintf(csv_name, sizeof(csv_name), "results/%s_acc.csv", base_name);

  //   FILE *fp = fopen(csv_name, "w");
  //   if (fp != NULL) {
  //     for (int r = 0; r < M; ++r) {
  //       for (int c = 0; c < N; ++c) {
  //         const float v =
  //             *(const float *)((const char *)dst->data + c * nb1 + r * nb0);
  //         fprintf(fp, c + 1 < N ? "%f," : "%f", v);
  //       }
  //       fputc('\n', fp);
  //     }
  //     fclose(fp);
  //   }
  // }
}

void ggml_secda_out_prod(ggml_secda_context *ctx, struct ggml_tensor *dst) {
  GGML_UNUSED(ctx);
  GGML_UNUSED(dst);
}

// Jude: Added - dispatches a GGML_OP_SOFT_MAX node to the V4 accelerator's
// tiled hardware softmax, one row (i03,i02,i01 slice) at a time. Loop
// structure, mask-broadcast indexing, and ALiBi slope formula are ported
// directly from ggml_compute_forward_soft_max_f32
// (llama.cpp/ggml/src/ggml-cpu/ops.cpp) - the actual per-element math
// (scale/mask/exp/normalize) runs on the accelerator, not here; this
// function only computes the same row pointers/parameters CPU would and
// hands each row to EntrySoftmax.
void ggml_secda_soft_max(ggml_secda_context *ctx, struct ggml_tensor *dst) {
  GGML_UNUSED(ctx);
#if defined(BFPP_ACC_V4)
  const struct ggml_tensor *src0 = dst->src[0]; // logits
  const struct ggml_tensor *src1 = dst->src[1]; // mask (optional)
  const struct ggml_tensor *src2 = dst->src[2]; // sinks (optional)

  GGML_TENSOR_UNARY_OP_LOCALS

  float scale = 1.0f;
  float max_bias = 0.0f;
  memcpy(&scale, (float *)dst->op_params + 0, sizeof(float));
  memcpy(&max_bias, (float *)dst->op_params + 1, sizeof(float));

  const int64_t nb11 = src1 ? src1->nb[1] : 1;
  const int64_t nb12 = src1 ? src1->nb[2] : 1;
  const int64_t nb13 = src1 ? src1->nb[3] : 1;
  const int64_t ne12 = src1 ? src1->ne[2] : 1;
  const int64_t ne13 = src1 ? src1->ne[3] : 1;
  const bool mask_is_f16 = src1 && src1->type == GGML_TYPE_F16;

  const uint32_t n_head = (uint32_t)ne02;
  const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));
  const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

  const bool has_sink = src2 != nullptr;
  const float *sk = has_sink ? (const float *)src2->data : nullptr;

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        const int64_t i11 = i01;
        const int64_t i12 = i02 % ne12;
        const int64_t i13 = i03 % ne13;

        const uint32_t h = (uint32_t)i02;
        const float slope =
            (max_bias > 0.0f)
                ? (h < n_head_log2 ? powf(m0, h + 1)
                                    : powf(m1, 2 * (h - n_head_log2) + 1))
                : 1.0f;

        const float *sp = (const float *)((const char *)src0->data +
                                          i01 * nb01 + i02 * nb02 + i03 * nb03);
        float *dp = (float *)((char *)dst->data + i01 * nb1 + i02 * nb2 +
                              i03 * nb3);
        const void *mp = src1 ? (const void *)((const char *)src1->data +
                                                i11 * nb11 + i12 * nb12 +
                                                i13 * nb13)
                              : nullptr;
        const float sink_value = has_sink ? sk[i02] : 0.0f;

        const bool node_done =
            (i03 == ne03 - 1) && (i02 == ne02 - 1) && (i01 == ne01 - 1);

        EntrySoftmax(sp, mp, mask_is_f16, has_sink, sink_value, dp,
                    (unsigned int)ne00, scale, slope, node_done);
      }
    }
  }
#else
  GGML_UNUSED(dst);
  GGML_ABORT("%s: SECDA SOFT_MAX offload requires BFPP_ACC_V4\n", __func__);
#endif
}
