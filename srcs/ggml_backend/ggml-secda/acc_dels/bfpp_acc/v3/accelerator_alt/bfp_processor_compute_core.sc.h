#ifndef __ACCNAME_BFP_PROCESSOR_COMPUTE_CORE_SC_H__
#define __ACCNAME_BFP_PROCESSOR_COMPUTE_CORE_SC_H__

#include "acc_config.sc.h"

float BFPP_UNIT::vec_dot() {
  // #pragma HLS inline off
  float sumf = 0;
  int32_t aux32b[16];
#pragma HLS array_partition variable = aux32b complete

  unsigned int wgt_ty = wgt_type.read();

  for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    aux32b[j] = 0;
  }

  // *********************************************************************
  // SB Dall
  // *********************************************************************
  float dall = ggml_compute_fp16_to_fp32(wd) * id;
  // *********************************************************************

  // *********************************************************************
  // SB Dmin
  // *********************************************************************
  // =====================================================================
  // Weight type 2 or 3 or 5
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK4) || defined(BFPP_QK5)
  float dmin = 0;
  int min_sum = 0;
  if (wgt_ty == 2 || wgt_ty == 4 || wgt_ty == 5) {
    dmin = ggml_compute_fp16_to_fp32(wdmin) * id;
    // Sums Calculation
    min_sum = 0;
    for (int k = 0; k < 16; ++k) {
#pragma HLS unroll
      sc_uint<6> wsmin;
      if (wgt_ty == 2) wsmin = wsmins[k].range(3, 0);
      if (wgt_ty == 4 || wgt_ty == 5) wsmin = wsmins[k].range(5, 0);
      min_sum += ibsums[k] * wsmin;
    }
  }
#endif
  // =====================================================================
  // *********************************************************************

  // *********************************************************************
  // Dot Product
  // *********************************************************************
  // Weight type 2 or 3 or 6
  // =====================================================================
#if defined(BFPP_QK2) || defined(BFPP_QK3) || defined(BFPP_QK6)

  if (wgt_ty == 2 || wgt_ty == 3 || wgt_ty == 6) {
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor = 16
      int sum = 0;
      for (int j = 0; j < 16; j++) {
#pragma HLS unroll factor = 16
        sum += iqs[(k * 16) + j] * wgt[(k * 16) + j];
      }
      aux32b[k] = sum;
    }

    int acc_sum = 0;
    for (int k = 0; k < 16; ++k) {
#pragma HLS unroll factor = 16
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
  // *********************************************************************
}

void BFPP_UNIT::ComputeCore() {
  Shake_Reset_S(ComputeCore);
  wait();
  while (1) {
    Shake_Wait(ComputeCore);
    float f1 = vec_dot();
    temp_fifo_out.write(f1);
    wait();
    Shake_Done(ComputeCore);
    wait();
  }
}

#endif // __ACCNAME_BFP_PROCESSOR_COMPUTE_CORE_SC_H__