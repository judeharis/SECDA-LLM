#ifndef __ACCNAME_BFP_PROCESSOR_COMPUTE_SC_H__
#define __ACCNAME_BFP_PROCESSOR_COMPUTE_SC_H__

#include "acc_config.sc.h"

float BFPP_UNIT::vec_dot(int wi, int ii) {
  // #pragma HLS inline OFF

  float sumf = 0;
  int8_t wgt[QK_K];
  int32_t aux32b[16];

  bool wm[QK_K];
  sc_uint<8> wscales[16];
  sc_uint<6> wsmins[16];
  sc_int<8> iqs[256];
  sc_uint<2> wqs[QK_K];

#pragma HLS array_partition variable = wgt complete
#pragma HLS array_partition variable = aux32b complete
#pragma HLS array_partition variable = wm complete
#pragma HLS array_partition variable = iqs complete
#pragma HLS array_partition variable = wqs complete
#pragma HLS array_partition variable = wscales complete
#pragma HLS array_partition variable = wsmins complete

  unsigned int wgt_ty = wgt_type.read();

  for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    aux32b[j] = 0;
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll
      wgt[(j * 16) + k] = 0;
    }
  }

// =====================================================================
// Weight type 2 or 3 or 6
// =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)
  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {
    int u = 0;
    int u2 = 128;
    for (int j = 0; j < 128; j += 4) {
#pragma HLS unroll
      wqs[0 + u] = w_qs[wi][j + 0];
      wqs[32 + u] = w_qs[wi][j + 1];
      wqs[64 + u] = w_qs[wi][j + 2];
      wqs[96 + u] = w_qs[wi][j + 3];
      u++;
    }

    for (int j = 128; j < 256; j += 4) {
#pragma HLS unroll
      wqs[0 + u2] = w_qs[wi][j + 0];
      wqs[32 + u2] = w_qs[wi][j + 1];
      wqs[64 + u2] = w_qs[wi][j + 2];
      wqs[96 + u2] = w_qs[wi][j + 3];
      u2++;
    }
  }
#endif

  // =====================================================================
  // Weight type 3 or 5
  // =====================================================================
#if defined(BFPP_QK3) || defined(BFPP_QK5)
  if (wgt_ty == 3 || wgt_ty == 5) {
    int u3 = 0;
    for (int j = 0; j < 256; j += 8) {
#pragma HLS unroll
      wm[0 + u3] = w_hmask[wi][j + 0];
      wm[32 + u3] = w_hmask[wi][j + 1];
      wm[64 + u3] = w_hmask[wi][j + 2];
      wm[96 + u3] = w_hmask[wi][j + 3];
      wm[128 + u3] = w_hmask[wi][j + 4];
      wm[160 + u3] = w_hmask[wi][j + 5];
      wm[192 + u3] = w_hmask[wi][j + 6];
      wm[224 + u3] = w_hmask[wi][j + 7];
      u3++;
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 3 or 5
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)
  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {
    for (int j = 0; j < 16; j++) {
      sc_uint<8> scales = w_scales[wi][j];
      wsmins[j] = scales.range(7, 4);
      if (wgt_ty == 2) wscales[j] = scales.range(3, 0);
      if (wgt_ty == 3) wscales[j] = scales.range(5, 0);
      if (wgt_ty == 6) wscales[j] = scales.range(7, 0);
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK4) || defined(BFPP_QK5)
  if (wgt_ty == 4 || wgt_ty == 5) {
    int m = 0;
    for (int j = 0; j < 8; j++) {
#pragma HLS unroll
      sc_uint<8> scales = w_scales[wi][j];
      wscales[j] = scales.range(6, 0);
      sc_uint<8> mins = w_scales[wi][j + 8];
      wsmins[m++] = mins.range(6, 0);
      wsmins[m++] = mins.range(6, 0);
    }
  }
#endif

  // =====================================================================
  // Load input quants
  // =====================================================================
  for (int j = 0; j < 256; j++) {
#pragma HLS unroll
    iqs[j] = i_qs[ii][j];
  }

  // =====================================================================
  // Prepare weight quants
  // =====================================================================
  for (int j = 0; j < QK_K; j++) {
#pragma HLS unroll
    sc_uint<8> w = 0;
#if defined(BFPP_QK2) || defined(BFPP_QK3)
    if (wgt_ty == 2 || wgt_ty == 3) wgt[j] = wqs[j];
    if (wgt_ty == 3) wgt[j] -= (wm[j] ? 0 : 4);
#endif

#if defined(BFPP_QK4) || defined(BFPP_QK5)
    if (wgt_ty == 4 || wgt_ty == 5) {
      w.range(3, 2) = w_qs2[wi][j];
      w.range(1, 0) = w_qs[wi][j];
      if (wgt_ty == 5) w.range(5, 4) = wm[j];
      wgt[j] = w;
    }
#endif

#if defined(BFPP_QK6)
    if (wgt_ty == 6) {
      w.range(5, 4) = wqs[j];
      w.range(3, 2) = w_qs3[wi][j];
      w.range(1, 0) = w_qs2[wi][j];
      wgt[j] = w - 32;
    }
#endif
  }

  // =====================================================================
  // Super-block scales calculations
  // =====================================================================
  float dall = ggml_compute_fp16_to_fp32(w_d[wi]) * i_d[ii];
  // =====================================================================
  // Weight type 2 or 3 or 5
  // =====================================================================

#if defined(BFPP_QK2) || defined(BFPP_QK4) || defined(BFPP_QK5)
  float dmin = ggml_compute_fp16_to_fp32(w_dmin[wi]) * i_d[ii];
  // Sums Calculation
  int min_sum = 0;
  for (int k = 0; k < 16; ++k) {
#pragma HLS pipeline II = 1
#pragma HLS unroll
    sc_uint<6> wsmin;
    if (wgt_ty == 2 || wgt_ty == 3) wsmin = wsmins[k].range(3, 0);
    if (wgt_ty == 4 || wgt_ty == 5) wsmin = wsmins[k].range(5, 0);
    min_sum += i_bsums[ii][k] * wsmin;
  }
#endif
  // =====================================================================

  // =====================================================================
  // Dot Product Calculation
  // =====================================================================
  // Weight type 2 or 3 or 6
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)

  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {
    for (int k = 0; k < 16; k++) {
#pragma HLS pipeline II = 1
      int sum = 0;
      for (int j = 0; j < 16; j++) {
#pragma HLS unroll factor = 16
        sum += iqs[(k * 16) + j] * wgt[(k * 16) + j];
      }
      aux32b[k] = sum;
    }

    int acc_sum = 0;
    for (int k = 0; k < 16; ++k) {
#pragma HLS pipeline II = 1
#pragma HLS unroll
      int scale = 0;
      sc_int<8> wgt_int_scale;
      wgt_int_scale.range(7, 0) = wscales[k].range(7, 0);
      if (wgt_ty == 6) scale = wgt_int_scale;
      else if (wgt_ty == 2) scale = wscales[k];
      else if (wgt_ty == 3) scale = wscales[k] - 32;
      acc_sum += aux32b[k] * scale;
    }

    float res_all = acc_sum * dall;
    float res = 0;
#if defined(BFPP_QK2)
    float res_q2 = (res_all) - (min_sum * dmin);
    if (wgt_ty == 2) res = res_q2;
#endif

#if defined(BFPP_QK3) || defined(BFPP_QK6)
    if (wgt_ty == 3 || wgt_ty == 6) res = res_all;
#endif
    return res;
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK4) || defined(BFPP_QK5)
  if (wgt_ty == 4 || wgt_ty == 5) {
    int32_t acc_sum = 0;
    for (int k = 0; k < 8; k++) {
#pragma HLS unroll factor = 8
      int32_t isum = 0;
      for (int j = 0; j < 32; j++) {
#pragma HLS unroll factor = 32
        isum += iqs[(k * 32) + j] * wgt[(k * 32) + j];
      }
      int scale = 0;
      scale = wscales[k];
      acc_sum += isum * scale;
    }
    float res = (acc_sum * dall) - (min_sum * dmin);
    return res;
  }
#endif
  // =====================================================================
}

#endif // __ACCNAME_BFP_PROCESSOR_COMPUTE_SC_H__