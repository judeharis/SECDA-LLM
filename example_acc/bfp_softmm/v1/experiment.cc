
// hello world c++ program

#include <fstream>
#include <iostream>

#ifdef SYSC
#include "secda-core/secda_integrator/systemc_integrate.h"
#endif

#include "accelerator/driver/driver.h"
#include "bfp_softmax.h"
#include "bfp_vec_dot.h"
#include "secda-core/secda_profiler/profiler.h"

static unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2, dma_addr3};
static unsigned int dma_addrs_in[4] = {dma_in0, dma_in1, dma_in2, dma_in3};
static unsigned int dma_addrs_out[4] = {dma_out0, dma_out1, dma_out2, dma_out3};

#ifdef SYSC
ACCNAME *acc;
struct sysC_sigs *scs;
struct s_mdma *mdma;
#else
int *acc;
struct s_mdma *mdma;
#endif

struct a_ctrl *ctrl;
static h_ctrl *hwc;

struct acc_times a_t;
static struct Profile profile;
// =========================================

#ifdef ACC_PROFILE
#define PLOG(X) X
#define DELLOG(X)
#else
#define PLOG(X)
#define DELLOG(X) X
#endif
using namespace std;

#ifndef PROB_N
#define PROB_N 2
#endif

#ifndef PROB_M
#define PROB_M 4
#endif

#ifndef PROB_K
#define PROB_K 2
#endif

#ifndef PROB_TYPE
#define PROB_TYPE 4
#endif

int main() {
  // ========================================
  // ========================================
  // Initialize the Accelerator

  DELLOG(std::cout << "===========================" << std::endl;);
#ifdef SYSC
  static ACCNAME _acc("ACCNAME");
  static struct sysC_sigs scs1(1);
  static struct a_ctrl ctrl1;
  static struct h_ctrl hwc1;
  static struct s_mdma mdma1(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                             DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
  sysC_init();
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl1.init_sigs(CTRL_Reg_Count);

  sysC_binder(&_acc, &scs1, &ctrl1, &hwc1, &mdma1);
  acc = &_acc;
  scs = &scs1;
  ctrl = &ctrl1;
  hwc = &hwc1;
  mdma = &mdma1;
  DELLOG(std::cout << "Initialised the SystemC Modules" << std::endl;);

#else
  acc = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *acc_ctrl_base = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *acc_hwc_base = getAccBaseAddress<int>(acc_hwc_address, 65536);
  static struct a_ctrl ctrl1(acc_ctrl_base);
  static struct h_ctrl hwc1(acc_hwc_base);
  static struct s_mdma mdma1(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                             DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
  ctrl1.init_sigs(CTRL_Reg_Count);
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl = &ctrl1;
  hwc = &hwc1;
  mdma = &mdma1;
  DELLOG(std::cout << "Initialised the DMA" << std::endl;);
#endif
  DELLOG(std::cout << "ACCNAME Accelerator";);
  DELLOG(std::cout << std::endl;);
  DELLOG(std::cout << "===========================" << std::endl;);
  // ========================================
  // ========================================
  // Data Preparation
  size_t bs = 0;
  size_t bx = 0;
  size_t by = 0;
  int nrc = 1;

  block_q2_K data_block_q2_K[3];
  block_q8_K data_block_q2_q8_K[3];
  block_q3_K data_block_q3_K[3];
  block_q8_K data_block_q3_q8_K[3];
  block_q4_K data_block_q4_K[6];
  block_q8_K data_block_q4_q8_K[6];
  block_q5_K data_block_q5_K[6];
  block_q8_K data_block_q5_q8_K[6];
  block_q6_K data_block_q6_K[3];
  block_q8_K data_block_q6_q8_K[3];

  // Load data from binary files
  std::ifstream file_src0("data/block_q2_K.bin", std::ios::binary);
  std::ifstream file_src1("data/block_q2_q8_K.bin", std::ios::binary);
  std::ifstream file_src2("data/block_q3_K.bin", std::ios::binary);
  std::ifstream file_src3("data/block_q3_q8_K.bin", std::ios::binary);
  std::ifstream file_src4("data/block_q4_K.bin", std::ios::binary);
  std::ifstream file_src5("data/block_q4_q8_K.bin", std::ios::binary);
  std::ifstream file_src6("data/block_q5_K.bin", std::ios::binary);
  std::ifstream file_src7("data/block_q5_q8_K.bin", std::ios::binary);
  std::ifstream file_src8("data/block_q6_K.bin", std::ios::binary);
  std::ifstream file_src9("data/block_q6_q8_K.bin", std::ios::binary);
  if (!file_src0.is_open() || !file_src1.is_open() || !file_src2.is_open() ||
      !file_src3.is_open() || !file_src4.is_open() || !file_src5.is_open() ||
      !file_src6.is_open() || !file_src7.is_open() || !file_src8.is_open() ||
      !file_src9.is_open()) {
    std::cerr << "Error: Could not open data files" << std::endl;
    return 1;
  }

  // Read single block from binary files
  file_src0.read(reinterpret_cast<char *>(data_block_q2_K),
                 sizeof(block_q2_K) * 3);
  file_src1.read(reinterpret_cast<char *>(data_block_q2_q8_K),
                 sizeof(block_q8_K) * 3);
  file_src2.read(reinterpret_cast<char *>(data_block_q3_K),
                 sizeof(block_q3_K) * 3);
  file_src3.read(reinterpret_cast<char *>(data_block_q3_q8_K),
                 sizeof(block_q8_K) * 3);
  file_src4.read(reinterpret_cast<char *>(data_block_q4_K),
                 sizeof(block_q4_K) * 6);
  file_src5.read(reinterpret_cast<char *>(data_block_q4_q8_K),
                 sizeof(block_q8_K) * 6);
  file_src6.read(reinterpret_cast<char *>(data_block_q5_K),
                 sizeof(block_q5_K) * 6);
  file_src7.read(reinterpret_cast<char *>(data_block_q5_q8_K),
                 sizeof(block_q8_K) * 6);
  file_src8.read(reinterpret_cast<char *>(data_block_q6_K),
                 sizeof(block_q6_K) * 3);
  file_src9.read(reinterpret_cast<char *>(data_block_q6_q8_K),
                 sizeof(block_q8_K) * 3);

  file_src0.close();
  file_src1.close();
  file_src2.close();
  file_src3.close();
  file_src4.close();
  file_src5.close();
  file_src6.close();
  file_src7.close();
  file_src8.close();
  file_src9.close();

  // ========================================
  // FPGA Prep
  acc_container drv;

#ifdef SYSC
  drv.scs = scs;
#endif
  drv.profile = &profile;
  drv.acc = acc;
  drv.ctrl = ctrl;
  drv.a_t = &a_t;
  drv.hwc = hwc;
  drv.mdma = mdma;

  // ========================================
  // ========================================
  // Synthetic MatMul Definition
  cout << "init Problem" << endl;

  int synM = PROB_M;
  int synN = PROB_N;
  int synK = 256 * PROB_K;
  float cpuOut[synM * synN];
  int wgt_type = PROB_TYPE;

  cout << "Done CPU Start" << endl;

  if (wgt_type == 2)
    run_synthetic_matmul_q2_K_q8_K(synM, synN, synK, data_block_q2_K,
                                   data_block_q2_q8_K, cpuOut, a_t);
  else if (wgt_type == 3)
    run_synthetic_matmul_q3_K_q8_K(synM, synN, synK, data_block_q3_K,
                                   data_block_q2_q8_K, cpuOut, a_t);
  else if (wgt_type == 4)
    run_synthetic_matmul_q4_K_q8_K(synM, synN, synK, data_block_q4_K,
                                   data_block_q2_q8_K, cpuOut, a_t);
  else if (wgt_type == 5)
    run_synthetic_matmul_q5_K_q8_K(synM, synN, synK, data_block_q5_K,
                                   data_block_q2_q8_K, cpuOut, a_t);
  else if (wgt_type == 6)
    run_synthetic_matmul_q6_K_q8_K(synM, synN, synK, data_block_q6_K,
                                   data_block_q2_q8_K, cpuOut, a_t);

  cout << "Done CPU Computation" << endl;

  // ========================================

  // ========================================

  int M_0 = synM;
  int N_0 = synN;
  int K_0 = synK;
  float acc_out[synM * synN];

  block_q8_K acc_q8_data[N_0 * (K_0 / 256)];
  for (int i = 0; i < N_0 * (K_0 / 256); i++)
    memcpy(&acc_q8_data[i], &(data_block_q2_q8_K[i % 3]), sizeof(block_q8_K));

  block_q8_K qw_data[M_0 * (K_0 / 256)];
  block_q2_K *acc_q2_data = (block_q2_K *)qw_data;
  block_q3_K *acc_q3_data = (block_q3_K *)qw_data;
  block_q4_K *acc_q4_data = (block_q4_K *)qw_data;
  block_q5_K *acc_q5_data = (block_q5_K *)qw_data;
  block_q6_K *acc_q6_data = (block_q6_K *)qw_data;

  if (wgt_type == 6) {
    for (int i = 0; i < M_0 * (K_0 / 256); i++)
      memcpy(&acc_q6_data[i], &(data_block_q6_K[i % 3]), sizeof(block_q6_K));
  } else if (wgt_type == 5) {
    for (int i = 0; i < M_0 * (K_0 / 256); i++)
      memcpy(&acc_q5_data[i], &(data_block_q5_K[i % 3]), sizeof(block_q5_K));
  } else if (wgt_type == 4) {
    for (int i = 0; i < M_0 * (K_0 / 256); i++)
      memcpy(&acc_q4_data[i], &(data_block_q4_K[i % 3]), sizeof(block_q4_K));
  } else if (wgt_type == 3) {
    for (int i = 0; i < M_0 * (K_0 / 256); i++)
      memcpy(&acc_q3_data[i], &(data_block_q3_K[i % 3]), sizeof(block_q3_K));
  } else if (wgt_type == 2) {
    for (int i = 0; i < M_0 * (K_0 / 256); i++)
      memcpy(&acc_q2_data[i], &(data_block_q2_K[i % 3]), sizeof(block_q2_K));
  }

#if (defined(BFPP_QK2) && (PROB_TYPE == 2)) ||                                 \
    (defined(BFPP_QK3) && (PROB_TYPE == 3)) ||                                 \
    (defined(BFPP_QK4) && (PROB_TYPE == 4)) ||                                 \
    (defined(BFPP_QK5) && (PROB_TYPE == 5)) ||                                 \
    (defined(BFPP_QK6) && (PROB_TYPE == 6))
  prf_start(1);
  acc_driver::Entry_Prepare(drv, M_0, N_0, K_0, (char *)acc_q8_data,
                            (char *)qw_data, (char *)acc_out, wgt_type);
  prf_end(1, a_t.fpga_total);
#endif
  // a_t.print();

  // print_results(acc_out, synM, synN, wgt_type, true);

  // #if defined(COMPARE_RESULTS)
  compare_results(cpuOut, acc_out, synM, synN);
  // #endif
  // ========================================

  // ========================================
  // ========================================
  // Softmax Test: fused F32 softmax(logits*scale + mask*ALiBi_slope) with
  // attention-sink correction, ported from llama.cpp's ggml_soft_max_ext.
  // n_head is deliberately not a power of two, so the test exercises both
  // branches of the ALiBi slope formula (heads below vs at/above n_head_log2).
  cout << "init Softmax Problem" << endl;

  const unsigned int SM_N_HEAD = 6;
  const unsigned int SM_ROWS_PER_HEAD = 2;
  const unsigned int SM_N = 32;
  const float SM_SCALE = 0.125f;
  const float SM_MAX_BIAS = 8.0f;

  unsigned int sm_n_head_log2 =
      1u << (unsigned int)floor(log2((double)SM_N_HEAD));
  float sm_m0 = powf(2.0f, -(SM_MAX_BIAS) / sm_n_head_log2);
  float sm_m1 = powf(2.0f, -(SM_MAX_BIAS / 2.0f) / sm_n_head_log2);

  static float sm_logits[SM_N_HEAD][SM_ROWS_PER_HEAD][SM_N];
  static float sm_mask[SM_N_HEAD][SM_N];
  static float sm_sink[SM_N_HEAD];

  for (unsigned int h = 0; h < SM_N_HEAD; h++) {
    sm_sink[h] = ((float)((h * 13 + 3) % 17) - 8.0f) / 4.0f;
    // causal-style mask: mask out the tail of the row, varying by head, but
    // never the whole row (an all-masked row is a legitimate-but-NaN-producing
    // edge case in real GGML too, not something to construct here)
    unsigned int cutoff = 8 + (h % 3) * 6;
    for (unsigned int i = 0; i < SM_N; i++)
      sm_mask[h][i] = (i < cutoff) ? 0.0f : -INFINITY;
    for (unsigned int r = 0; r < SM_ROWS_PER_HEAD; r++)
      for (unsigned int i = 0; i < SM_N; i++)
        sm_logits[h][r][i] =
            ((float)(((h + 1) * (r + 1) * (i * 37 + 11)) % 101) - 50.0f) /
            10.0f;
  }

  cout << "Done Softmax CPU + FPGA" << endl;
  for (unsigned int h = 0; h < SM_N_HEAD; h++) {
    float slope;
    if (SM_MAX_BIAS <= 0.0f) {
      slope = 1.0f;
    } else if (h < sm_n_head_log2) {
      slope = powf(sm_m0, h + 1);
    } else {
      slope = powf(sm_m1, 2 * (h - sm_n_head_log2) + 1);
    }

    for (unsigned int r = 0; r < SM_ROWS_PER_HEAD; r++) {
      float sm_cpu_out[SM_N];
      bfp_soft_max_f32_row(sm_logits[h][r], sm_mask[h], /*mask_is_f16=*/false,
                           /*has_sink=*/true, sm_sink[h], sm_cpu_out, SM_N,
                           SM_SCALE, slope);

      float sm_acc_out[SM_N];
      acc_driver::Entry_Prepare_Softmax(
          drv, sm_logits[h][r], sm_mask[h], nullptr, /*has_sink=*/true,
          sm_sink[h], sm_acc_out, SM_N, SM_SCALE, slope);

      cout << "Softmax head " << h << " row " << r << ": ";
      compare_results(sm_cpu_out, sm_acc_out, SM_N, 1);
    }
  }
  // ========================================

  return 0;
}
