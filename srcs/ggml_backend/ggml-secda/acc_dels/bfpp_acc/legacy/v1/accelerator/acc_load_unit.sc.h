#ifndef __ACCNAME_LOAD_UNIT_SC_H__
#define __ACCNAME_LOAD_UNIT_SC_H__

void ACCNAME::Weight_Transfer() {
  HWC_SIG(Weight_Transfer, 0);
  wait();
  while (1) {
    HWC_SIG(Weight_Transfer, 1);
    ADATA d;
    d = wgt_fifo.read();
    HWC_SIG(Weight_Transfer, 2);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      vars.wgt_write(d, i);
    }
  }
}

void ACCNAME::Load_Unit() {
  ADATA d = {0, 0};

  data_load.write(0);
  SIGWRITE(loadS, 0);
  HWC_SIG(Load_Unit, 0);

  wait();
  while (1) {
    SIGWRITE(loadS, 1);
    HWC_SIG(Load_Unit, 1);

    while (!data_load.read()) wait();
    SIGWRITE(loadS, 2);
    HWC_SIG(Load_Unit, 2);

    opcode op = op_src;
    wait();

    if (op.load_inp) {
      inp_packet ip = inp_packet(&din1);
      SIGWRITE(loadS, 3);
      HWC_SIG(Load_Unit, 3);

      unsigned int i = 0;
      d.data = ip.knb;
      for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
        vars.inp_write(d, i);
      }
      for (int kn = 0; kn < ip.knb; kn++) {
        for (int s = 0; s < INP_BLCK; s++) {
#pragma HLS pipeline
          sc_uint<32> val = din1.read().data.to_uint();
          d.data = val;
          for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
            vars.inp_write(d, i);
          }
        }
      }
    }
    wait();

    if (op.load_wgt) {
      wgt_packet wp = wgt_packet(&din1);
      SIGWRITE(loadS, 4);
      HWC_SIG(Load_Unit, 4);

      unsigned int i = 0;
      d.data = wp.kmb;
      wgt_fifo.write(d);
      for (int km = 0; km < wp.kmb; km++) {
        for (int s = 0; s < wp.wgt_blck; s++) {
#pragma HLS pipeline
          sc_uint<32> val = din1.read().data.to_uint();
          d.data = val;
          wgt_fifo.write(d);
        }
      }
    }
    wait();

    data_load.write(false);
    DWAIT();
  }
}

#endif // __ACCNAME_LOAD_UNIT_SC_H__