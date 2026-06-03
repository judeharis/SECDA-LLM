#include "ops_support.h"

#if defined(BFPP_ACC_V3)
#include "acc_dels/bfpp_acc/v3/accelerator/driver/acc_driver_connector.h"
#elif defined(BFPP_ACC_V2)
#include "acc_dels/bfpp_acc/v2/accelerator/driver/acc_driver_connector.h"
#else
#include "acc_dels/bfpp_acc/v1/accelerator/driver/acc_driver_connector.h"
#endif

#include "ggml-quants.h"
#include <chrono>
#include <fstream>
#include <iostream>

bool load = true;

// // placeholder 
// void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
//     quantize_row_q8_K_ref(x, y, k);
// }

bool preload_weights_alloc(unsigned wgt_size, int layer, int M, int K,
                           const void *wgt, int wgt_type) {
  return preloadWeights(wgt_size, layer, M, K, wgt, wgt_type);
}

void initSECDA_ACC() { initACC(); }

void updatePlan_T(int supported_nodes) { updatePlan(supported_nodes); }

bool modelPlanned_T() { return modelPlanned(); }

void ggml_secda_mul_mat(ggml_secda_context *ctx, struct ggml_tensor *dst) {

  auto start_1 = std::chrono::high_resolution_clock::now();
  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];
  GGML_TENSOR_BINARY_OP_LOCALS

  const enum ggml_type type = src0->type;

  enum ggml_type const vec_dot_type = GGML_TYPE_Q8_K;

  GGML_ASSERT(nb00 == ggml_type_size(type));

  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  const int64_t M = ne0;
  const int64_t N = ne1;
  const int64_t K = ne10;

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
              (block_q8_K *)(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1), ne10);
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

  int wgt_stride = nb01;
  int out_stride = nb1;
  int wgt_type = 3;
  if (type == GGML_TYPE_Q6_K) wgt_type = 6;
  if (type == GGML_TYPE_Q5_K) wgt_type = 5;
  if (type == GGML_TYPE_Q4_K) wgt_type = 4;
  if (type == GGML_TYPE_Q3_K) wgt_type = 3;
  if (type == GGML_TYPE_Q2_K) wgt_type = 2;

  EntryMM(src0->data, twdata, dst->data, M, N, K, row_size, wgt_stride,
          out_stride, wgt_type);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto time1 = end1 - start_1;
  updateProfile(time1);
  // double elapsed_time =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(time1).count();
  // updateProfile(elapsed_time);
}

void ggml_secda_out_prod(ggml_secda_context *ctx, struct ggml_tensor *dst) {
  GGML_UNUSED(ctx);
  GGML_UNUSED(dst);
}
