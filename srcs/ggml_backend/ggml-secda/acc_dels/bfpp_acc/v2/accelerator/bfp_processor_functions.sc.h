#ifndef __ACCNAME_BFP_PROCESSOR_FUNCTIONS_SC_H__
#define __ACCNAME_BFP_PROCESSOR_FUNCTIONS_SC_H__

#include "acc_config.sc.h"
#include <bitset>

void BFPP_UNIT::u96_to_u6x16(sc_biguint<96> in, sc_uint<8> *out) {
  int s = 0;
  for (int i = 0; i < 8; i += 4) {
#pragma HLS loop unroll
    out[(i + 0)].range(3, 0) = in.range((i + 0) * 8 + 3, (i + 0) * 8);
    out[(i + 0)].range(5, 4) = in.range(65 + s, 64 + s);

    out[(i + 1)].range(3, 0) = in.range((i + 1) * 8 + 3, (i + 1) * 8);
    out[(i + 1)].range(5, 4) = in.range(73 + s, 72 + s);

    out[(i + 2)].range(3, 0) = in.range((i + 2) * 8 + 3, (i + 2) * 8);
    out[(i + 2)].range(5, 4) = in.range(81 + s, 80 + s);

    out[(i + 3)].range(3, 0) = in.range((i + 3) * 8 + 3, (i + 3) * 8);
    out[(i + 3)].range(5, 4) = in.range(89 + s, 88 + s);
    s += 2;
  }

  for (int i = 0; i < 8; i += 4) {
#pragma HLS loop unroll
    out[(i + 8)].range(3, 0) = in.range((i + 0) * 8 + 7, (i + 0) * 8 + 4);
    out[(i + 8)].range(5, 4) = in.range(65 + s, 64 + s);

    out[(i + 9)].range(3, 0) = in.range((i + 1) * 8 + 7, (i + 1) * 8 + 4);
    out[(i + 9)].range(5, 4) = in.range(73 + s, 72 + s);

    out[(i + 10)].range(3, 0) = in.range((i + 2) * 8 + 7, (i + 2) * 8 + 4);
    out[(i + 10)].range(5, 4) = in.range(81 + s, 80 + s);

    out[(i + 11)].range(3, 0) = in.range((i + 3) * 8 + 7, (i + 3) * 8 + 4);
    out[(i + 11)].range(5, 4) = in.range(89 + s, 88 + s);
    s += 2;
  }
}

void BFPP_UNIT::u96_to_u6x8x2(sc_biguint<96> in, sc_uint<8> *out) {
  int s = 0;
  for (int i = 0; i < 4; i++) {
#pragma HLS loop unroll
    out[(i + 0)].range(5, 0) = in.range(i * 8 + 5, i * 8);

    out[(i + 4)].range(3, 0) = in.range(i * 8 + 67, i * 8 + 64);
    out[(i + 4)].range(5, 4) = in.range(i * 8 + 7, i * 8 + 6);

    out[(i + 8)].range(5, 0) = in.range(i * 8 + 37, i * 8 + 32);

    out[(i + 12)].range(3, 0) = in.range(i * 8 + 71, i * 8 + 68);
    out[(i + 12)].range(5, 4) = in.range(i * 8 + 39, i * 8 + 38);
  }
}

float BFPP_UNIT::fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

uint32_t BFPP_UNIT::fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

float BFPP_UNIT::ggml_compute_fp16_to_fp32(uint16_t h) {
  // #pragma HLS inline OFF
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));

  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

float BFPP_UNIT::vec_dot_q3q2(int wi, int ii) {
  // #pragma HLS inline OFF

  float sumf = 0;
  int8_t aux8[QK_K];
  int32_t aux32[16][16];
  int32_t aux32b[16];
  float sums[16];

  sc_uint<2> wqs[QK_K];
  bool wm[QK_K];
  sc_uint<8> wscales[16];
  sc_uint<6> wsmins[16];
  sc_int<8> iqs[256];
  int8_t *a = aux8;

  int summs[16];

#pragma HLS array_partition variable = summs complete

#pragma HLS array_partition variable = aux8 complete
#pragma HLS array_partition variable = aux32 complete
#pragma HLS array_partition variable = aux32b complete
#pragma HLS array_partition variable = sums complete
#pragma HLS array_partition variable = wm complete
#pragma HLS array_partition variable = iqs complete
#pragma HLS array_partition variable = wqs complete
#pragma HLS array_partition variable = wscales complete

#pragma HLS array_partition variable = wsmins complete
  unsigned int wgt_ty = wgt_type.read();

  for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    aux32b[j] = 0;
    sums[j] = 0;
    summs[j] = 0;
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll
      aux32[j][k] = 0;
      aux8[(j * 16) + k] = 0;
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
    if (wgt_ty == 2 || wgt_ty == 3) a[j] = wqs[j];
    if (wgt_ty == 3) a[j] -= (wm[j] ? 0 : 4);
#endif

#if defined(BFPP_QK4) || defined(BFPP_QK5)
    if (wgt_ty == 4 || wgt_ty == 5) {
      w.range(3, 2) = w_qs2[wi][j];
      w.range(1, 0) = w_qs[wi][j];
      if (wgt_ty == 5) w.range(5, 4) = wm[j];
      a[j] = w;
    }
#endif

#if defined(BFPP_QK6)
    if (wgt_ty == 6) {
      w.range(5, 4) = wqs[j];
      w.range(3, 2) = w_qs3[wi][j];
      w.range(1, 0) = w_qs2[wi][j];
      a[j] = w - 32;
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
  for (int k = 0; k < 16; ++k) {
#pragma HLS pipeline II = 1
#pragma HLS unroll
    sc_uint<6> wsmin;
    if (wgt_ty == 2 || wgt_ty == 3) wsmin = wsmins[k].range(3, 0);
    if (wgt_ty == 4 || wgt_ty == 5) wsmin = wsmins[k].range(5, 0);
    summs[k] = i_bsums[ii][k] * wsmin;
  }
  summs[0] = summs[0] + summs[1];
  summs[2] = summs[2] + summs[3];
  summs[4] = summs[4] + summs[5];
  summs[6] = summs[6] + summs[7];
  summs[8] = summs[8] + summs[9];
  summs[10] = summs[10] + summs[11];
  summs[12] = summs[12] + summs[13];
  summs[14] = summs[14] + summs[15];
  summs[1] = summs[0] + summs[2];
  summs[3] = summs[4] + summs[6];
  summs[5] = summs[8] + summs[10];
  summs[7] = summs[12] + summs[14];
  summs[0] = summs[1] + summs[3];
  summs[2] = summs[5] + summs[7];
  summs[4] = summs[0] + summs[2];
  summs[0] = summs[4];
#endif
  // =====================================================================

  // =====================================================================
  // Dot Product Calculation
  // =====================================================================
  // Weight type 2 or 3 or 6
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)

  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {
    int32_t acc_sum = 0;
    for (int k = 0; k < 16; k++) {
      int32_t isum = 0;
      for (int j = 0; j < 16; j++) {
#pragma HLS unroll factor = 16
        isum += iqs[(k * 16) + j] * a[(k * 16) + j];
      }
      int scale = 0;
      sc_int<8> wgt_int_scale;
      wgt_int_scale.range(7, 0) = wscales[k].range(7, 0);
      if (wgt_ty == 6) scale = wgt_int_scale;
      else if (wgt_ty == 2) scale = wscales[k];
      else if (wgt_ty == 3) scale = wscales[k] - 32;
      acc_sum += isum * scale;
    }

    float res_all = acc_sum * dall;
    float res = 0;
#if defined(BFPP_QK2)
    float res_q2 = (res_all) - (summs[0] * dmin);
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
        isum += iqs[(k * 32) + j] * a[(k * 32) + j];
      }
      int scale = 0;
      scale = wscales[k];
      acc_sum += isum * scale;
    }
    float res = (acc_sum * dall) - (summs[0] * dmin);
    return res;
  }
#endif
  // =====================================================================
}

// =====================================================================
// =====================================================================
// =====================================================================

void BFPP_UNIT::weight_mapper(int wb) {
  SIGWRITE(wgtlS, 3);
  int wgt_types = wgt_type.read();

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK4)
  if (wgt_type.read() == 4 || wgt_type.read() == 5) {
    sc_uint<32> sscale = wgt_fifo.read().data.to_uint();
    w_d[wb] = sscale.range(15, 0);
    w_dmin[wb] = sscale.range(31, 16);
    sc_uint<32> scale1 = wgt_fifo.read().data.to_uint();
    sc_uint<32> scale2 = wgt_fifo.read().data.to_uint();
    sc_uint<32> scale3 = wgt_fifo.read().data.to_uint();
    sc_biguint<96> scale_com = (scale3, scale2, scale1);
    u96_to_u6x8x2(scale_com, &w_scales[wb][0]);
  }
#endif

  // =====================================================================
  // Weight type 5 or 3
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK3)
  if (wgt_type.read() == 5 || wgt_type.read() == 3) {
    for (int i = 0; i < 256; i += 32) {
#pragma HLS pipeline II = 1
      sc_uint<32> hmask = wgt_fifo.read().data.to_uint();
      for (int k = 0; k < 32; k++) {
#pragma HLS loop unroll
        w_hmask[wb][i + k] = hmask.range(k, k);
      }
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK4)
  if (wgt_type.read() == 4 || wgt_type.read() == 5) {
    for (int k = 0; k < 4; k++) {
#pragma HLS pipeline II = 1
      for (int i = 0; i < 8; i++) {
#pragma HLS pipeline II = 1
        sc_uint<32> qs = wgt_fifo.read().data.to_uint();
        for (int j = 0; j < 4; j++) {
#pragma HLS unroll
          w_qs[wb][k * 64 + i * 4 + j] = qs.range(j * 8 + 1, j * 8 + 0);
          w_qs2[wb][k * 64 + i * 4 + j] = qs.range(j * 8 + 3, j * 8 + 2);
          w_qs[wb][k * 64 + i * 4 + j + 32] = qs.range(j * 8 + 5, j * 8 + 4);
          w_qs2[wb][k * 64 + i * 4 + j + 32] = qs.range(j * 8 + 7, j * 8 + 6);
        }
      }
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 6
  // =====================================================================
#if defined(BFPP_QK6)
  if (wgt_type.read() == 6) {
    for (int k = 0; k < 2; k++) {
#pragma HLS pipeline II = 1
      for (int i = 0; i < 16; i++) {
#pragma HLS pipeline II = 1
        sc_uint<32> qs = wgt_fifo.read().data.to_uint();
        for (int j = 0; j < 4; j++) {
#pragma HLS unroll
          w_qs2[wb][k * 128 + i * 4 + j] = qs.range(j * 8 + 1, j * 8 + 0);
          w_qs3[wb][k * 128 + i * 4 + j] = qs.range(j * 8 + 3, j * 8 + 2);

          w_qs2[wb][k * 128 + i * 4 + j + 64] = qs.range(j * 8 + 5, j * 8 + 4);
          w_qs3[wb][k * 128 + i * 4 + j + 64] = qs.range(j * 8 + 7, j * 8 + 6);
        }
      }
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 2
  // =====================================================================
#if defined(BFPP_QK2)
  if (wgt_type.read() == 2) {
    for (int i = 0; i < 16; i += 4) {
#pragma HLS pipeline II = 1
      sc_uint<32> scales = wgt_fifo.read().data.to_uint();
      w_scales[wb][i + 0] = scales.range(8, 0);
      w_scales[wb][i + 1] = scales.range(16, 8);
      w_scales[wb][i + 2] = scales.range(24, 16);
      w_scales[wb][i + 3] = scales.range(31, 24);
    }
  }
#endif
  // =====================================================================

  SIGWRITE(wgtlS, 4);

  // =====================================================================
  // Weight type 2 or 3 or 6
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)
  if (wgt_type.read() == 2 || wgt_type.read() == 3 || wgt_type.read() == 6) {
    for (int i = 0; i < 64; i += 4) {
#pragma HLS pipeline II = 1
      sc_uint<32> qs = wgt_fifo.read().data.to_uint();
      for (int j = 0; j < 16; j++) {
#pragma HLS unroll
        w_qs[wb][i * 4 + j] = qs.range((j + 1) * 2 - 1, j * 2);
      }
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 6
  // =====================================================================
#if defined(BFPP_QK6)
  if (wgt_type.read() == 6) {
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      sc_uint<32> scales = wgt_fifo.read().data.to_uint();
      for (int j = 0; j < 4; j++) {
#pragma HLS unroll
        w_scales[wb][i * 4 + j] = scales.range(j * 8 + 7, j * 8 + 0);
      }
    }
    sc_uint<32> data = wgt_fifo.read().data.to_uint();
    w_d[wb] = data.range(15, 0);
  }
#endif
  // =====================================================================

  SIGWRITE(wgtlS, 5);

  // =====================================================================
  // Weight type 3
  // =====================================================================
#if defined(BFPP_QK3)
  if (wgt_type.read() == 3) {
    sc_uint<32> scale1 = wgt_fifo.read().data.to_uint();
    sc_uint<32> scale2 = wgt_fifo.read().data.to_uint();
    sc_uint<32> scale3 = wgt_fifo.read().data.to_uint();
    sc_biguint<96> scale_com = (scale3, scale2, scale1);
    u96_to_u6x16(scale_com, &w_scales[wb][0]);
    sc_uint<32> data = wgt_fifo.read().data.to_uint();
    w_d[wb] = data.range(15, 0);
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 2
  // =====================================================================
#if defined(BFPP_QK2)
  if (wgt_type.read() == 2) {
    sc_uint<32> sscale = wgt_fifo.read().data.to_uint();
    w_d[wb] = sscale.range(15, 0);
    w_dmin[wb] = sscale.range(31, 16);
  }
#endif
  // =====================================================================

  DWAIT(1);
}

#endif // __ACCNAME_BFP_PROCESSOR_FUNCTIONS_SC_H__