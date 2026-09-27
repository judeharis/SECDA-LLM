#ifndef __ACCNAME_BFP_PROCESSOR_COMPUTE_LOAD_SC_H__
#define __ACCNAME_BFP_PROCESSOR_COMPUTE_LOAD_SC_H__

#include "acc_config.sc.h"

float BFPP_UNIT::compute_load(int wi, int ii) {
  // #pragma HLS inline off
  unsigned int wgt_ty = wgt_type.read();

  for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll
      wgt[(j * 16) + k] = 0;
    }
  }

  // *********************************************************************
  // Wgt_Load
  // *********************************************************************

  for (int j = 0; j < 16; j++) {
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor = 16
      wqs[j * 16 + k] = w_qs[wi][j][k];
      iqs[j * 16 + k] = i_qs[ii][j][k];
    }
  }

  // =====================================================================
  // Weight type 4 or 5 or 6
  // =====================================================================
#if defined(BFPP_QK4) || defined(BFPP_QK5) || defined(BFPP_QK6)
  if (wgt_ty == 4 || wgt_ty == 5 || wgt_ty == 6) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor = 16
        wqs2[j * 16 + k] = w_qs2[wi][j][k];
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
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor = 16
        wqs3[j * 16 + k] = w_qs3[wi][j][k];
      }
    }
  }
#endif

  // =====================================================================
  // Weight type 3 or 5
  // =====================================================================
#if defined(BFPP_QK3) || defined(BFPP_QK5)
  if (wgt_ty == 3 || wgt_ty == 5) {
    int u3 = 0;
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k += 8) {
#pragma HLS unroll factor = 2
        wm[0 + u3] = w_hmask[wi][j][k + 0];
        wm[32 + u3] = w_hmask[wi][j][k + 1];
        wm[64 + u3] = w_hmask[wi][j][k + 2];
        wm[96 + u3] = w_hmask[wi][j][k + 3];
        wm[128 + u3] = w_hmask[wi][j][k + 4];
        wm[160 + u3] = w_hmask[wi][j][k + 5];
        wm[192 + u3] = w_hmask[wi][j][k + 6];
        wm[224 + u3] = w_hmask[wi][j][k + 7];
        u3++;
      }
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
#pragma HLS unroll factor = 16
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
      wscales[j] = scales.range(5, 0);
      sc_uint<8> mins = w_scales[wi][j + 8];
      wsmins[m++] = mins.range(5, 0);
      wsmins[m++] = mins.range(5, 0);
    }
  }
#endif

  //*********************************************************************

  // *********************************************************************
  // Inp_Load
  // *********************************************************************

  for (int j = 0; j < 16; j++) {
#pragma HLS unroll factor = 16
    ibsums[j] = i_bsums[ii][j];
  }

  // *********************************************************************

  // *********************************************************************
  // Wgt_Prep
  // *********************************************************************
  for (int j = 0; j < QK_K; j++) {
#pragma HLS unroll
    sc_uint<8> w = 0;
#if defined(BFPP_QK2) || defined(BFPP_QK3)
    if (wgt_ty == 2 || wgt_ty == 3) wgt[j] = wqs[j];
    if (wgt_ty == 3) wgt[j] -= (wm[j] ? 0 : 4);
#endif

#if defined(BFPP_QK4) || defined(BFPP_QK5)
    if (wgt_ty == 4 || wgt_ty == 5) {
      w.range(3, 2) = wqs2[j];
      w.range(1, 0) = wqs[j];
      if (wgt_ty == 5) w.range(5, 4) = wm[j];
      wgt[j] = w;
    }
#endif

#if defined(BFPP_QK6)
    if (wgt_ty == 6) {
      w.range(5, 4) = wqs[j];
      w.range(3, 2) = wqs3[j];
      w.range(1, 0) = wqs2[j];
      wgt[j] = w - 32;
    }
#endif
  }
  id = i_d[ii];
  wd = w_d[wi];
  wdmin = w_dmin[wi];
  return vec_dot();
  // *********************************************************************
}

#endif // __ACCNAME_BFP_PROCESSOR_COMPUTE_LOAD_SC_H__