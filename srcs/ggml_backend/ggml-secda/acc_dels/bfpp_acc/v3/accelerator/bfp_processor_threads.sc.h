#ifndef __ACCNAME_BFP_PROCESSOR_THREADS_SC_H__
#define __ACCNAME_BFP_PROCESSOR_THREADS_SC_H__

#include "acc_config.sc.h"

void BFPP_UNIT::LoadWeights() {
  SIGWRITE(LoadWeightS, 0);
  wait();
  while (1) {
    // Load weights
    SIGWRITE(LoadWeightS, 1);
    while (!bfpp_free.read()) wait();
    unsigned int wbs_l1 = wgt_fifo1.read().data.to_uint();
    unsigned int wbs_l2 = wgt_fifo2.read().data.to_uint();
    unsigned int wbs_l3 = wgt_fifo3.read().data.to_uint();
    unsigned int wbs_l4 = wgt_fifo4.read().data.to_uint();
    SIGWRITE(LoadWeightS, 2);
    for (int wb = 0; wb < wbs_l1; wb += 4) {
#pragma HLS loop_tripcount min = 8 max = 8
      weight_mapper4x(wb, wbs_l1, wbs_l2, wbs_l3, wbs_l4);
      DWAIT(1);
    }
    DWAIT(1);
  }
}

void BFPP_UNIT::LoadInputs() {
  SIGWRITE(LoadInputS, 0);
  wait();
  while (1) {
    // Load inputs
    SIGWRITE(LoadInputS, 1);
    while (!bfpp_free.read()) wait();

    unsigned int ibs_l = inp_fifo.read().data.to_uint();
    SIGWRITE(LoadInputS, 2);
    for (int ib = 0; ib < ibs_l; ib++) {
      SIGWRITE(LoadInputS, 3);
      // Fix: Weird code, why array of 2?
      int delta2[2];
      delta2[0] = inp_fifo.read().data;
      i_d[ib] = *(float *)delta2;
      SIGWRITE(LoadInputS, 4);
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j += 4) {
#pragma HLS pipeline II = 1
          sc_uint<32> qs_y = inp_fifo.read().data.to_uint();
          i_qs[ib][i][j + 0] = qs_y.range(7, 0);
          i_qs[ib][i][j + 1] = qs_y.range(15, 8);
          i_qs[ib][i][j + 2] = qs_y.range(23, 16);
          i_qs[ib][i][j + 3] = qs_y.range(31, 24);
        }
      }
      SIGWRITE(LoadInputS, 5);
      DWAIT(1);
      for (int i = 0; i < 16; i += 2) {
#pragma HLS pipeline II = 1
        sc_uint<32> bsums = inp_fifo.read().data.to_uint();
        i_bsums[ib][i] = bsums.range(15, 0);
        i_bsums[ib][i + 1] = bsums.range(31, 16);
      }
      SIGWRITE(LoadInputS, 6);
      DWAIT(1);
    }
    DWAIT(1);
  }
}

void BFPP_UNIT::Compute() {
  bfpp_ready.write(1);
  bfpp_free.write(1);
  ADATA last = {5000, 1};
  ADATA d1 = {0, 0};
  SIGWRITE(ComputeS, 0);
  Shake_Reset_M(ComputeLoad);
  Shake_Reset_M(ComputeCore);
  wait();
  while (1) {
    SIGWRITE(ComputeS, 1);
    while (!bfpp_start.read() || LoadInputS.read() != 1 ||
           LoadWeightS.read() != 1)
      wait();

    bfpp_ready.write(0);
    bfpp_free.write(0);
    while (bfpp_start.read()) wait();
    unsigned int mstep = wb_idx.read();
    unsigned int nstep = ib_idx.read();
    unsigned int kb_l = kb.read();
    wait();
    int m_idx = 0;
    int n_idx = 0;

    SIGWRITE(ComputeS, 2);
    for (int n = 0; n < nstep; n++) {
      for (int m = 0; m < mstep; m++) {
        float acc_sumf = 0;
        for (int k = 0; k < kb_l; k++) {
#pragma HLS pipeline II = 1
          float f1 = compute_load(m_idx++, n_idx + k);
          acc_sumf += f1;
          wait();
        }
        int *fout = reinterpret_cast<int *>(&acc_sumf);
        d1.data = fout[0];
        d1.tlast = (n == nstep - 1 && m == mstep - 1);
        dout1.write(d1);
      }
      n_idx += kb_l;
      m_idx = 0;
    }
    bfpp_ready.write(1);
    bfpp_free.write(1);
    SIGWRITE(ComputeS, 3);
    wait();
  }
}

#endif // __ACCNAME_BFPP_THREADS_SC_H__
