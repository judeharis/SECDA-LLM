#ifndef __ACCNAME_BFP_PROCESSOR_THREADS_SC_H__
#define __ACCNAME_BFP_PROCESSOR_THREADS_SC_H__

#include "acc_config.sc.h"

void BFPP_UNIT::LoadWeights() {
  sc_uint<2> wqs[QK_K];
  bool wm[QK_K];
#pragma HLS array_partition variable = wqs complete
#pragma HLS array_partition variable = wm complete

  SIGWRITE(wgtlS, 0);
  wait();
  while (1) {
    // Load weights
    SIGWRITE(wgtlS, 1);
    while (!bfpp_free.read()) wait();

    unsigned int wbs_l = wgt_fifo.read().data.to_uint();
    SIGWRITE(wgtlS, 2);
    for (int wb = 0; wb < wbs_l; wb++) {
#pragma HLS loop_tripcount min = 8 max = 8
      weight_mapper(wb);
      DWAIT(1);
    }
    DWAIT(1);
  }
}


void BFPP_UNIT::LoadInputs() {
  SIGWRITE(inplS, 0);
  wait();
  while (1) {
    // Load inputs
    SIGWRITE(inplS, 1);
    while (!bfpp_free.read()) wait();

    unsigned int ibs_l = inp_fifo.read().data.to_uint();
    SIGWRITE(inplS, 2);
    for (int ib = 0; ib < ibs_l; ib++) {
      SIGWRITE(inplS, 3);
      // Fix: Weird code, why array of 2?
      int delta2[2];
      delta2[0] = inp_fifo.read().data;
      i_d[ib] = *(float *)delta2;
      SIGWRITE(inplS, 4);
      for (int i = 0; i < 256; i += 4) {
#pragma HLS pipeline II = 1
        sc_uint<32> qs_y = inp_fifo.read().data.to_uint();
        i_qs[ib][i] = qs_y.range(7, 0);
        i_qs[ib][i + 1] = qs_y.range(15, 8);
        i_qs[ib][i + 2] = qs_y.range(23, 16);
        i_qs[ib][i + 3] = qs_y.range(31, 24);
      }
      SIGWRITE(inplS, 5);
      DWAIT(1);
      for (int i = 0; i < 16; i += 2) {
#pragma HLS pipeline II = 1
        sc_uint<32> bsums = inp_fifo.read().data.to_uint();
        i_bsums[ib][i] = bsums.range(15, 0);
        i_bsums[ib][i + 1] = bsums.range(31, 16);
      }
      SIGWRITE(inplS, 6);
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
  SIGWRITE(computeS, 0);
  wait();
  while (1) {
    // SIGWRITE(computeS, 1);
    while (!bfpp_start.read() || wgtlS.read() != 1 || inplS.read() != 1)
    wait(); bfpp_ready.write(0); bfpp_free.write(0); while
    (bfpp_start.read()) wait(); unsigned int mstep = wb_idx.read(); unsigned
    int nstep = ib_idx.read(); unsigned int kb_l = kb.read(); wait(); int
    m_idx = 0; int n_idx = 0;

    // SIGWRITE(computeS, 2);
    for (int n = 0; n < nstep; n++) {
      for (int m = 0; m < mstep; m++) {

        float acc_sumf = 0;
        for (int k = 0; k < kb_l; k++) {
#pragma HLS pipeline II = 1
          // Compute
          float f1 = 0;
          f1 = vec_dot_q3q2(m_idx++, n_idx + k);
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
    // SIGWRITE(computeS, 3);
    wait();
  }
}

#endif // __ACCNAME_BFPP_THREADS_SC_H__
