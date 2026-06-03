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
    out[(i + 0)].range(7, 6) = 0;

    out[(i + 1)].range(3, 0) = in.range((i + 1) * 8 + 3, (i + 1) * 8);
    out[(i + 1)].range(5, 4) = in.range(73 + s, 72 + s);
    out[(i + 1)].range(7, 6) = 0;

    out[(i + 2)].range(3, 0) = in.range((i + 2) * 8 + 3, (i + 2) * 8);
    out[(i + 2)].range(5, 4) = in.range(81 + s, 80 + s);
    out[(i + 2)].range(7, 6) = 0;

    out[(i + 3)].range(3, 0) = in.range((i + 3) * 8 + 3, (i + 3) * 8);
    out[(i + 3)].range(5, 4) = in.range(89 + s, 88 + s);
    out[(i + 3)].range(7, 6) = 0;
    s += 2;
  }

  for (int i = 0; i < 8; i += 4) {
#pragma HLS loop unroll
    out[(i + 8)].range(3, 0) = in.range((i + 0) * 8 + 7, (i + 0) * 8 + 4);
    out[(i + 8)].range(5, 4) = in.range(65 + s, 64 + s);
    out[(i + 8)].range(7, 6) = 0;

    out[(i + 9)].range(3, 0) = in.range((i + 1) * 8 + 7, (i + 1) * 8 + 4);
    out[(i + 9)].range(5, 4) = in.range(73 + s, 72 + s);
    out[(i + 9)].range(7, 6) = 0;

    out[(i + 10)].range(3, 0) = in.range((i + 2) * 8 + 7, (i + 2) * 8 + 4);
    out[(i + 10)].range(5, 4) = in.range(81 + s, 80 + s);
    out[(i + 10)].range(7, 6) = 0;

    out[(i + 11)].range(3, 0) = in.range((i + 3) * 8 + 7, (i + 3) * 8 + 4);
    out[(i + 11)].range(5, 4) = in.range(89 + s, 88 + s);
    out[(i + 11)].range(7, 6) = 0;
    s += 2;
  }
}

void BFPP_UNIT::u96_to_u6x8x2(sc_biguint<96> in, sc_uint<8> *out) {
  int s = 0;
  for (int i = 0; i < 4; i++) {
#pragma HLS loop unroll
    out[(i + 0)].range(5, 0) = in.range(i * 8 + 5, i * 8);
    out[(i + 0)].range(7, 6) = 0;


    out[(i + 4)].range(3, 0) = in.range(i * 8 + 67, i * 8 + 64);
    out[(i + 4)].range(5, 4) = in.range(i * 8 + 7, i * 8 + 6);
    out[(i + 4)].range(7, 6) = 0;


    out[(i + 8)].range(5, 0) = in.range(i * 8 + 37, i * 8 + 32);
    out[(i + 8)].range(7, 6) = 0;

    out[(i + 12)].range(3, 0) = in.range(i * 8 + 71, i * 8 + 68);
    out[(i + 12)].range(5, 4) = in.range(i * 8 + 39, i * 8 + 38);
    out[(i + 12)].range(7, 6) = 0;
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

// =====================================================================
// =====================================================================
// =====================================================================

void BFPP_UNIT::weight_mapper4x(int wb, unsigned int l1, unsigned int l2,
                                unsigned int l3, unsigned int l4) {
  // #pragma HLS inline off
  // SIGWRITE(wgtlS, 3);
  int wgt_ty = wgt_type.read();
  bool run_A = wb < l1;
  bool run_B = wb + 1 < l2;
  bool run_C = wb + 2 < l3;
  bool run_D = wb + 3 < l4;

  sc_uint<2> temp_wqsA[QK_K];
  sc_uint<2> temp_wqsB[QK_K];
  sc_uint<2> temp_wqsC[QK_K];
  sc_uint<2> temp_wqsD[QK_K];
#pragma HLS array_partition variable = temp_wqsA complete
#pragma HLS array_partition variable = temp_wqsB complete
#pragma HLS array_partition variable = temp_wqsC complete
#pragma HLS array_partition variable = temp_wqsD complete

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK4)
  if (wgt_ty == 4 || wgt_ty == 5) {
    sc_uint<32> sscaleA;
    sc_uint<32> sscaleB;
    sc_uint<32> sscaleC;
    sc_uint<32> sscaleD;
    sc_uint<32> scale1A;
    sc_uint<32> scale1B;
    sc_uint<32> scale1C;
    sc_uint<32> scale1D;
    sc_uint<32> scale2A;
    sc_uint<32> scale2B;
    sc_uint<32> scale2C;
    sc_uint<32> scale2D;
    sc_uint<32> scale3A;
    sc_uint<32> scale3B;
    sc_uint<32> scale3C;
    sc_uint<32> scale3D;
    sc_biguint<96> scale_comA;
    sc_biguint<96> scale_comB;
    sc_biguint<96> scale_comC;
    sc_biguint<96> scale_comD;
    if (run_A) sscaleA = wgt_fifo1.read().data.to_uint();
    if (run_B) sscaleB = wgt_fifo2.read().data.to_uint();
    if (run_C) sscaleC = wgt_fifo3.read().data.to_uint();
    if (run_D) sscaleD = wgt_fifo4.read().data.to_uint();

    if (run_A) w_d[wb + 0] = sscaleA.range(15, 0);
    if (run_B) w_d[wb + 1] = sscaleB.range(15, 0);
    if (run_C) w_d[wb + 2] = sscaleC.range(15, 0);
    if (run_D) w_d[wb + 3] = sscaleD.range(15, 0);

    if (run_A) w_dmin[wb + 0] = sscaleA.range(31, 16);
    if (run_B) w_dmin[wb + 1] = sscaleB.range(31, 16);
    if (run_C) w_dmin[wb + 2] = sscaleC.range(31, 16);
    if (run_D) w_dmin[wb + 3] = sscaleD.range(31, 16);
    if (run_A) scale1A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale1B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale1C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale1D = wgt_fifo4.read().data.to_uint();
    if (run_A) scale2A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale2B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale2C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale2D = wgt_fifo4.read().data.to_uint();
    if (run_A) scale3A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale3B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale3C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale3D = wgt_fifo4.read().data.to_uint();
    if (run_A) scale_comA = (scale3A, scale2A, scale1A);
    if (run_B) scale_comB = (scale3B, scale2B, scale1B);
    if (run_C) scale_comC = (scale3C, scale2C, scale1C);
    if (run_D) scale_comD = (scale3D, scale2D, scale1D);
    if (run_A) u96_to_u6x8x2(scale_comA, &(w_scales[wb + 0][0]));
    if (run_B) u96_to_u6x8x2(scale_comB, &(w_scales[wb + 1][0]));
    if (run_C) u96_to_u6x8x2(scale_comC, &(w_scales[wb + 2][0]));
    if (run_D) u96_to_u6x8x2(scale_comD, &(w_scales[wb + 3][0]));
  }
#endif

  // =====================================================================
  // Weight type 5 or 3
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK3)
  if (wgt_ty == 5 || wgt_ty == 3) {
  loop_A:
    for (int i = 0; i < 8; i++) {
#pragma HLS pipeline II = 1
      sc_uint<32> hmaskA;
      sc_uint<32> hmaskB;
      sc_uint<32> hmaskC;
      sc_uint<32> hmaskD;
      if (run_A) hmaskA = wgt_fifo1.read().data.to_uint();
      if (run_B) hmaskB = wgt_fifo2.read().data.to_uint();
      if (run_C) hmaskC = wgt_fifo3.read().data.to_uint();
      if (run_D) hmaskD = wgt_fifo4.read().data.to_uint();
      for (int k = 0; k < 2; k++) {
        for (int p = 0; p < 16; p++) {
#pragma HLS loop unroll
          if (run_A)
            w_hmask[wb + 0][i * 2 + k][p] =
                hmaskA.range(k * 16 + p, k * 16 + p);
          if (run_B)
            w_hmask[wb + 1][i * 2 + k][p] =
                hmaskB.range(k * 16 + p, k * 16 + p);
          if (run_C)
            w_hmask[wb + 2][i * 2 + k][p] =
                hmaskC.range(k * 16 + p, k * 16 + p);
          if (run_D)
            w_hmask[wb + 3][i * 2 + k][p] =
                hmaskD.range(k * 16 + p, k * 16 + p);
        }
      }
    }
  }
#endif
  // =====================================================================

  // =====================================================================
  // Weight type 4 or 5
  // =====================================================================
#if defined(BFPP_QK5) || defined(BFPP_QK4)
  if (wgt_ty == 4 || wgt_ty == 5) {
  loop_B:
    for (int k = 0; k < 4; k++) {
#pragma HLS pipeline II = 1
      for (int i = 0; i < 2; i++) {
        for (int p = 0; p < 4; p++) {

#pragma HLS pipeline II = 1
          sc_uint<32> qsA;
          sc_uint<32> qsB;
          sc_uint<32> qsC;
          sc_uint<32> qsD;
          if (run_A) qsA = wgt_fifo1.read().data.to_uint();
          if (run_B) qsB = wgt_fifo2.read().data.to_uint();
          if (run_C) qsC = wgt_fifo3.read().data.to_uint();
          if (run_D) qsD = wgt_fifo4.read().data.to_uint();

          for (int j = 0; j < 4; j++) {
            // #pragma HLS unroll
            if (run_A)
              w_qs[wb + 0][k * 4 + i][p * 4 + j] =
                  qsA.range(j * 8 + 1, j * 8 + 0);
            if (run_A)
              w_qs2[wb + 0][k * 4 + i][p * 4 + j] =
                  qsA.range(j * 8 + 3, j * 8 + 2);
            if (run_A)
              w_qs[wb + 0][k * 4 + i + 2][p * 4 + j] =
                  qsA.range(j * 8 + 5, j * 8 + 4);
            if (run_A)
              w_qs2[wb + 0][k * 4 + i + 2][p * 4 + j] =
                  qsA.range(j * 8 + 7, j * 8 + 6);

            if (run_B)
              w_qs[wb + 1][k * 4 + i][p * 4 + j] =
                  qsB.range(j * 8 + 1, j * 8 + 0);
            if (run_B)
              w_qs2[wb + 1][k * 4 + i][p * 4 + j] =
                  qsB.range(j * 8 + 3, j * 8 + 2);
            if (run_B)
              w_qs[wb + 1][k * 4 + i + 2][p * 4 + j] =
                  qsB.range(j * 8 + 5, j * 8 + 4);
            if (run_B)
              w_qs2[wb + 1][k * 4 + i + 2][p * 4 + j] =
                  qsB.range(j * 8 + 7, j * 8 + 6);

            if (run_C)
              w_qs[wb + 2][k * 4 + i][p * 4 + j] =
                  qsC.range(j * 8 + 1, j * 8 + 0);
            if (run_C)
              w_qs2[wb + 2][k * 4 + i][p * 4 + j] =
                  qsC.range(j * 8 + 3, j * 8 + 2);
            if (run_C)
              w_qs[wb + 2][k * 4 + i + 2][p * 4 + j] =
                  qsC.range(j * 8 + 5, j * 8 + 4);
            if (run_C)
              w_qs2[wb + 2][k * 4 + i + 2][p * 4 + j] =
                  qsC.range(j * 8 + 7, j * 8 + 6);
            if (run_D)
              w_qs[wb + 3][k * 4 + i][p * 4 + j] =
                  qsD.range(j * 8 + 1, j * 8 + 0);
            if (run_D)
              w_qs2[wb + 3][k * 4 + i][p * 4 + j] =
                  qsD.range(j * 8 + 3, j * 8 + 2);
            if (run_D)
              w_qs[wb + 3][k * 4 + i + 2][p * 4 + j] =
                  qsD.range(j * 8 + 5, j * 8 + 4);
            if (run_D)
              w_qs2[wb + 3][k * 4 + i + 2][p * 4 + j] =
                  qsD.range(j * 8 + 7, j * 8 + 6);
          }
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
  if (wgt_ty == 6) {
  loop_C:
    for (int k = 0; k < 2; k++) {
#pragma HLS pipeline II = 1
      for (int i = 0; i < 4; i++) {
        for (int p = 0; p < 4; p++) {

#pragma HLS pipeline II = 1
          sc_uint<32> qsA;
          sc_uint<32> qsB;
          sc_uint<32> qsC;
          sc_uint<32> qsD;
          if (run_A) qsA = wgt_fifo1.read().data.to_uint();
          if (run_B) qsB = wgt_fifo2.read().data.to_uint();
          if (run_C) qsC = wgt_fifo3.read().data.to_uint();
          if (run_D) qsD = wgt_fifo4.read().data.to_uint();

          for (int j = 0; j < 4; j++) {
            // #pragma HLS unroll
            if (run_A)
              w_qs2[wb + 0][k * 8 + i][p * 4 + j] =
                  qsA.range(j * 8 + 1, j * 8 + 0);
            if (run_A)
              w_qs3[wb + 0][k * 8 + i][p * 4 + j] =
                  qsA.range(j * 8 + 3, j * 8 + 2);
            if (run_A)
              w_qs2[wb + 0][k * 8 + i + 4][p * 4 + j] =
                  qsA.range(j * 8 + 5, j * 8 + 4);
            if (run_A)
              w_qs3[wb + 0][k * 8 + i + 4][p * 4 + j] =
                  qsA.range(j * 8 + 7, j * 8 + 6);

            if (run_B)
              w_qs2[wb + 1][k * 8 + i][p * 4 + j] =
                  qsB.range(j * 8 + 1, j * 8 + 0);
            if (run_B)
              w_qs3[wb + 1][k * 8 + i][p * 4 + j] =
                  qsB.range(j * 8 + 3, j * 8 + 2);
            if (run_B)
              w_qs2[wb + 1][k * 8 + i + 4][p * 4 + j] =
                  qsB.range(j * 8 + 5, j * 8 + 4);
            if (run_B)
              w_qs3[wb + 1][k * 8 + i + 4][p * 4 + j] =
                  qsB.range(j * 8 + 7, j * 8 + 6);
            if (run_C)
              w_qs2[wb + 2][k * 8 + i][p * 4 + j] =
                  qsC.range(j * 8 + 1, j * 8 + 0);
            if (run_C)
              w_qs3[wb + 2][k * 8 + i][p * 4 + j] =
                  qsC.range(j * 8 + 3, j * 8 + 2);
            if (run_C)
              w_qs2[wb + 2][k * 8 + i + 4][p * 4 + j] =
                  qsC.range(j * 8 + 5, j * 8 + 4);
            if (run_C)
              w_qs3[wb + 2][k * 8 + i + 4][p * 4 + j] =
                  qsC.range(j * 8 + 7, j * 8 + 6);
            if (run_D)
              w_qs2[wb + 3][k * 8 + i][p * 4 + j] =
                  qsD.range(j * 8 + 1, j * 8 + 0);
            if (run_D)
              w_qs3[wb + 3][k * 8 + i][p * 4 + j] =
                  qsD.range(j * 8 + 3, j * 8 + 2);
            if (run_D)
              w_qs2[wb + 3][k * 8 + i + 4][p * 4 + j] =
                  qsD.range(j * 8 + 5, j * 8 + 4);
            if (run_D)
              w_qs3[wb + 3][k * 8 + i + 4][p * 4 + j] =
                  qsD.range(j * 8 + 7, j * 8 + 6);
          }
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
  if (wgt_ty == 2) {
  loop_D:
    for (int i = 0; i < 16; i += 4) {
#pragma HLS pipeline II = 1
      sc_uint<32> scalesA;
      sc_uint<32> scalesB;
      sc_uint<32> scalesC;
      sc_uint<32> scalesD;
      if (run_A) scalesA = wgt_fifo1.read().data.to_uint();
      if (run_B) scalesB = wgt_fifo2.read().data.to_uint();
      if (run_C) scalesC = wgt_fifo3.read().data.to_uint();
      if (run_D) scalesD = wgt_fifo4.read().data.to_uint();

      if (run_A) w_scales[wb][i + 0] = scalesA.range(8, 0);
      if (run_A) w_scales[wb][i + 1] = scalesA.range(16, 8);
      if (run_A) w_scales[wb][i + 2] = scalesA.range(24, 16);
      if (run_A) w_scales[wb][i + 3] = scalesA.range(31, 24);
      if (run_B) w_scales[wb + 1][i + 0] = scalesB.range(8, 0);
      if (run_B) w_scales[wb + 1][i + 1] = scalesB.range(16, 8);
      if (run_B) w_scales[wb + 1][i + 2] = scalesB.range(24, 16);
      if (run_B) w_scales[wb + 1][i + 3] = scalesB.range(31, 24);
      if (run_C) w_scales[wb + 2][i + 0] = scalesC.range(8, 0);
      if (run_C) w_scales[wb + 2][i + 1] = scalesC.range(16, 8);
      if (run_C) w_scales[wb + 2][i + 2] = scalesC.range(24, 16);
      if (run_C) w_scales[wb + 2][i + 3] = scalesC.range(31, 24);
      if (run_D) w_scales[wb + 3][i + 0] = scalesD.range(8, 0);
      if (run_D) w_scales[wb + 3][i + 1] = scalesD.range(16, 8);
      if (run_D) w_scales[wb + 3][i + 2] = scalesD.range(24, 16);
      if (run_D) w_scales[wb + 3][i + 3] = scalesD.range(31, 24);
    }
  }
#endif
  // =====================================================================

  // SIGWRITE(wgtlS, 4);

  // =====================================================================
  // Weight type 2 or 3 or 6
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)

  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {

  loop_E:
    for (int i = 0; i < 16; i++) {
#pragma HLS pipeline II = 1
      sc_uint<32> qsA;
      sc_uint<32> qsB;
      sc_uint<32> qsC;
      sc_uint<32> qsD;
      if (run_A) qsA = wgt_fifo1.read().data.to_uint();
      if (run_B) qsB = wgt_fifo2.read().data.to_uint();
      if (run_C) qsC = wgt_fifo3.read().data.to_uint();
      if (run_D) qsD = wgt_fifo4.read().data.to_uint();
      for (int j = 0; j < 16; j++) {
#pragma HLS unroll
        if (run_A) temp_wqsA[i * 16 + j] = qsA.range(j * 2 + 1, j * 2);
        if (run_B) temp_wqsB[i * 16 + j] = qsB.range(j * 2 + 1, j * 2);
        if (run_C) temp_wqsC[i * 16 + j] = qsC.range(j * 2 + 1, j * 2);
        if (run_D) temp_wqsD[i * 16 + j] = qsD.range(j * 2 + 1, j * 2);
        // if (run_A) w_qs[wb + 0][i][j] = qsA.range((j + 1) * 2 - 1, j * 2);
        // if (run_B) w_qs[wb + 1][i][j] = qsB.range((j + 1) * 2 - 1, j * 2);
        // if (run_C) w_qs[wb + 2][i][j] = qsC.range((j + 1) * 2 - 1, j * 2);
        // if (run_D) w_qs[wb + 3][i][j] = qsD.range((j + 1) * 2 - 1, j * 2);
      }
    }

  loop_F:
    for (int i = 0; i < 2; i++) {
#pragma HLS pipeline II = 1
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
          for (int p = 0; p < 16; p++) {
#pragma HLS unroll
            if (run_A)
              w_qs[wb + 0][i * 8 + j * 2 + k][p] =
                  temp_wqsA[i * 128 + j + k * 64 + p * 4];
            if (run_B)
              w_qs[wb + 1][i * 8 + j * 2 + k][p] =
                  temp_wqsB[i * 128 + j + k * 64 + p * 4];
            if (run_C)
              w_qs[wb + 2][i * 8 + j * 2 + k][p] =
                  temp_wqsC[i * 128 + j + k * 64 + p * 4];
            if (run_D)
              w_qs[wb + 3][i * 8 + j * 2 + k][p] =
                  temp_wqsD[i * 128 + j + k * 64 + p * 4];
          }
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
  if (wgt_ty == 6) {
  loop_G:
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      sc_uint<32> scalesA;
      sc_uint<32> scalesB;
      sc_uint<32> scalesC;
      sc_uint<32> scalesD;
      if (run_A) scalesA = wgt_fifo1.read().data.to_uint();
      if (run_B) scalesB = wgt_fifo2.read().data.to_uint();
      if (run_C) scalesC = wgt_fifo3.read().data.to_uint();
      if (run_D) scalesD = wgt_fifo4.read().data.to_uint();

      if (run_A) {
        w_scales[wb + 0][i * 4 + 0] = scalesA.range(0 * 8 + 7, 0 * 8);
        w_scales[wb + 0][i * 4 + 1] = scalesA.range(1 * 8 + 7, 1 * 8);
        w_scales[wb + 0][i * 4 + 2] = scalesA.range(2 * 8 + 7, 2 * 8);
        w_scales[wb + 0][i * 4 + 3] = scalesA.range(3 * 8 + 7, 3 * 8);
      }
      if (run_B) {
        w_scales[wb + 1][i * 4 + 0] = scalesB.range(0 * 8 + 7, 0 * 8);
        w_scales[wb + 1][i * 4 + 1] = scalesB.range(1 * 8 + 7, 1 * 8);
        w_scales[wb + 1][i * 4 + 2] = scalesB.range(2 * 8 + 7, 2 * 8);
        w_scales[wb + 1][i * 4 + 3] = scalesB.range(3 * 8 + 7, 3 * 8);
      }
      if (run_C) {
        w_scales[wb + 2][i * 4 + 0] = scalesC.range(0 * 8 + 7, 0 * 8);
        w_scales[wb + 2][i * 4 + 1] = scalesC.range(1 * 8 + 7, 1 * 8);
        w_scales[wb + 2][i * 4 + 2] = scalesC.range(2 * 8 + 7, 2 * 8);
        w_scales[wb + 2][i * 4 + 3] = scalesC.range(3 * 8 + 7, 3 * 8);
      }
      if (run_D) {
        w_scales[wb + 3][i * 4 + 0] = scalesD.range(0 * 8 + 7, 0 * 8);
        w_scales[wb + 3][i * 4 + 1] = scalesD.range(1 * 8 + 7, 1 * 8);
        w_scales[wb + 3][i * 4 + 2] = scalesD.range(2 * 8 + 7, 2 * 8);
        w_scales[wb + 3][i * 4 + 3] = scalesD.range(3 * 8 + 7, 3 * 8);
      }
    }
    sc_uint<32> dataA;
    sc_uint<32> dataB;
    sc_uint<32> dataC;
    sc_uint<32> dataD;
    if (run_A) dataA = wgt_fifo1.read().data.to_uint();
    if (run_B) dataB = wgt_fifo2.read().data.to_uint();
    if (run_C) dataC = wgt_fifo3.read().data.to_uint();
    if (run_D) dataD = wgt_fifo4.read().data.to_uint();
    if (run_A) w_d[wb] = dataA.range(15, 0);
    if (run_B) w_d[wb + 1] = dataB.range(15, 0);
    if (run_C) w_d[wb + 2] = dataC.range(15, 0);
    if (run_D) w_d[wb + 3] = dataD.range(15, 0);
  }
#endif
// =====================================================================

// SIGWRITE(wgtlS, 5);

// =====================================================================
// Weight type 3
// =====================================================================
#if defined(BFPP_QK3)
  if (wgt_ty == 3) {
    sc_uint<32> scale1A;
    sc_uint<32> scale1B;
    sc_uint<32> scale1C;
    sc_uint<32> scale1D;
    sc_uint<32> scale2A;
    sc_uint<32> scale2B;
    sc_uint<32> scale2C;
    sc_uint<32> scale2D;
    sc_uint<32> scale3A;
    sc_uint<32> scale3B;
    sc_uint<32> scale3C;
    sc_uint<32> scale3D;
    sc_biguint<96> scale_comA;
    sc_biguint<96> scale_comB;
    sc_biguint<96> scale_comC;
    sc_biguint<96> scale_comD;
    sc_uint<32> dataA;
    sc_uint<32> dataB;
    sc_uint<32> dataC;
    sc_uint<32> dataD;
    if (run_A) scale1A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale1B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale1C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale1D = wgt_fifo4.read().data.to_uint();
    if (run_A) scale2A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale2B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale2C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale2D = wgt_fifo4.read().data.to_uint();
    if (run_A) scale3A = wgt_fifo1.read().data.to_uint();
    if (run_B) scale3B = wgt_fifo2.read().data.to_uint();
    if (run_C) scale3C = wgt_fifo3.read().data.to_uint();
    if (run_D) scale3D = wgt_fifo4.read().data.to_uint();

    if (run_A) scale_comA = (scale3A, scale2A, scale1A);
    if (run_B) scale_comB = (scale3B, scale2B, scale1B);
    if (run_C) scale_comC = (scale3C, scale2C, scale1C);
    if (run_D) scale_comD = (scale3D, scale2D, scale1D);
    if (run_A) u96_to_u6x16(scale_comA, &w_scales[wb][0]);
    if (run_B) u96_to_u6x16(scale_comB, &w_scales[wb + 1][0]);
    if (run_C) u96_to_u6x16(scale_comC, &w_scales[wb + 2][0]);
    if (run_D) u96_to_u6x16(scale_comD, &w_scales[wb + 3][0]);

    if (run_A) dataA = wgt_fifo1.read().data.to_uint();
    if (run_B) dataB = wgt_fifo2.read().data.to_uint();
    if (run_C) dataC = wgt_fifo3.read().data.to_uint();
    if (run_D) dataD = wgt_fifo4.read().data.to_uint();
    if (run_A) w_d[wb] = dataA.range(15, 0);
    if (run_B) w_d[wb + 1] = dataB.range(15, 0);
    if (run_C) w_d[wb + 2] = dataC.range(15, 0);
    if (run_D) w_d[wb + 3] = dataD.range(15, 0);
  }
#endif
// =====================================================================

// =====================================================================
// Weight type 2
// =====================================================================
#if defined(BFPP_QK2)
  if (wgt_ty == 2) {
    sc_uint<32> sscaleA;
    sc_uint<32> sscaleB;
    sc_uint<32> sscaleC;
    sc_uint<32> sscaleD;
    if (run_A) sscaleA = wgt_fifo1.read().data.to_uint();
    if (run_B) sscaleB = wgt_fifo2.read().data.to_uint();
    if (run_C) sscaleC = wgt_fifo3.read().data.to_uint();
    if (run_D) sscaleD = wgt_fifo4.read().data.to_uint();
    if (run_A) w_d[wb] = sscaleA.range(15, 0);
    if (run_A) w_dmin[wb] = sscaleA.range(31, 16);
    if (run_B) w_d[wb + 1] = sscaleB.range(15, 0);
    if (run_B) w_dmin[wb + 1] = sscaleB.range(31, 16);
    if (run_C) w_d[wb + 2] = sscaleC.range(15, 0);
    if (run_C) w_dmin[wb + 2] = sscaleC.range(31, 16);
    if (run_D) w_d[wb + 3] = sscaleD.range(15, 0);
    if (run_D) w_dmin[wb + 3] = sscaleD.range(31, 16);
  }
#endif
  // =====================================================================

  DWAIT(1);
}

#endif // __ACCNAME_BFP_PROCESSOR_FUNCTIONS_SC_H__