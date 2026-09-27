#ifndef __ACCNAME_LOAD_UNIT_SC_H__
#define __ACCNAME_LOAD_UNIT_SC_H__

void ACCNAME::Weight_Transfer_A() {
  HWC_SIG(Weight_Transfer_A, 0);
  wait();
  while (1) {
    ADATA d;
    d = glb_wf_1.read();
    HWC_SIG(Weight_Transfer_A, 1);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      vars.wgt1_write(d, i);
    }
  }
}

void ACCNAME::Weight_Transfer_B() {
  HWC_SIG(Weight_Transfer_B, 0);
  wait();
  while (1) {
    ADATA d;
    d = glb_wf_2.read();
    HWC_SIG(Weight_Transfer_B, 1);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      vars.wgt2_write(d, i);
    }
  }
}

void ACCNAME::Weight_Transfer_C() {
  HWC_SIG(Weight_Transfer_C, 0);
  wait();
  while (1) {
    ADATA d;
    d = glb_wf_3.read();
    HWC_SIG(Weight_Transfer_C, 1);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      vars.wgt3_write(d, i);
    }
  }
}

void ACCNAME::Weight_Transfer_D() {
  HWC_SIG(Weight_Transfer_D, 0);
  wait();
  while (1) {
    ADATA d;
    d = glb_wf_4.read();
    HWC_SIG(Weight_Transfer_D, 1);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      vars.wgt4_write(d, i);
    }
  }
}

void ACCNAME::WeightLoader_A() {
  HWC_SIG(WeightLoader_A, 0);
  wait();
  while (1) {
    ADATA d;

    while (!weightloader_A.read()) wait();
    unsigned int len = weightloader_A_len.read();
    for (int s = 0; s < len; s++) {
#pragma HLS pipeline
      sc_uint<32> val = din1.read().data.to_uint();
      d.data = val;
      glb_wf_1.write(d);
    }
    weightloader_A.write(false);
    wait();
  }
}

void ACCNAME::WeightLoader_B() {
  HWC_SIG(WeightLoader_B, 0);
  wait();
  while (1) {
    ADATA d;

    while (!weightloader_B.read()) wait();
    unsigned int len = weightloader_B_len.read();
    for (int s = 0; s < len; s++) {
#pragma HLS pipeline
      sc_uint<32> val = din2.read().data.to_uint();
      d.data = val;
      glb_wf_2.write(d);
    }
    weightloader_B.write(false);
    wait();
  }
}

void ACCNAME::WeightLoader_C() {
  HWC_SIG(WeightLoader_C, 0);
  wait();
  while (1) {
    ADATA d;

    while (!weightloader_C.read()) wait();
    unsigned int len = weightloader_C_len.read();
    for (int s = 0; s < len; s++) {
#pragma HLS pipeline
      sc_uint<32> val = din3.read().data.to_uint();
      d.data = val;
      glb_wf_3.write(d);
    }
    weightloader_C.write(false);
    wait();
  }
}

void ACCNAME::WeightLoader_D() {
  HWC_SIG(WeightLoader_D, 0);
  wait();
  while (1) {
    ADATA d;

    while (!weightloader_D.read()) wait();
    unsigned int len = weightloader_D_len.read();
    for (int s = 0; s < len; s++) {
#pragma HLS pipeline
      sc_uint<32> val = din4.read().data.to_uint();
      d.data = val;
      glb_wf_4.write(d);
    }
    weightloader_D.write(false);
    wait();
  }
}

void ACCNAME::Load_Unit() {
  ADATA d = {0, 0};
  ADATA d2 = {0, 0};

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

      d.data = wp.kmb;
      glb_wf_1.write(d);
      glb_wf_2.write(d);
      glb_wf_3.write(d);
      glb_wf_4.write(d);

      unsigned int i = 0;
      weightloader_A_len.write(wp.a_size);
      weightloader_B_len.write(wp.b_size);
      weightloader_C_len.write(wp.c_size);
      weightloader_D_len.write(wp.d_size);
      wait();
      weightloader_A.write(true);
      weightloader_B.write(true);
      weightloader_C.write(true);
      weightloader_D.write(true);
      wait();
      while (weightloader_A.read() || weightloader_B.read() ||
             weightloader_C.read() || weightloader_D.read())
        wait();
    }
    wait();

    if (op.compute && op.store && op.load_inp && op.load_wgt) {
      data_load.write(true);
      dout2.write(d);
      dout3.write(d);
      dout4.write(d);
    }

    data_load.write(false);
    DWAIT();
  }
}

#endif // __ACCNAME_LOAD_UNIT_SC_H__