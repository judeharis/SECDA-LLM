#ifndef __ACCNAME_CONTROL_UNIT_SC_H__
#define __ACCNAME_CONTROL_UNIT_SC_H__

void ACCNAME::Control_Unit() {
  // clang-format off
#ifdef __SYNTHESIS__
  vars.vars_0.wgtlSS.write(0);
  vars.vars_0.inplSS.write(0);
#endif

  done.write(0);
  bool started = start.read();
  // clang-format on

  SIGWRITE(inS, 0);
  HWC_SIG(Control_Unit, 0);
  wait();
  while (1) {
    SIGWRITE(inS, 1);
    HWC_SIG(Control_Unit, 1);
    op_src = opcode(din1.read().data.to_uint());
    opcode op = op_src;
    SIGWRITE(inS, 2);
    HWC_SIG(Control_Unit, 2);

    wait();
    if (op_src.config) {
      SIGWRITE(inS, 3);
      HWC_SIG(Control_Unit, 3);
      while (schedule.read()) wait();
      kb = din1.read().data.to_uint();
      M = din1.read().data.to_uint();
      N = din1.read().data.to_uint();
      int wgt_t = din1.read().data.to_uint();
      wgt_type = wgt_t;
      for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
        vars.wgt_type_write(wgt_type, i);
      }
      wait();
    }
    wait();

    if (op_src.load_inp || op_src.load_wgt) {
      SIGWRITE(inS, 4);
      HWC_SIG(Control_Unit, 4);
      data_load.write(true);
      wait();
      while (data_load.read()) wait();
    }

    wait();
    if (op_src.compute) {
      while (schedule.read()) wait();
      kmb = din1.read().data.to_uint();
      knb = din1.read().data.to_uint();
      mstep = din1.read().data.to_uint();
      nstep = din1.read().data.to_uint();

      SIGWRITE(inS, 5);
      HWC_SIG(Control_Unit, 5);
      schedule.write(1);
      wait();
    }
    wait();

    // Jude: Added - dispatch one softmax tile/pass round. Header shape
    // mirrors OPCODE_CONFIG's 4-word read; word[0] (softmax_n) doubles as
    // tile_len for the tiled two-pass protocol (see acc_softmax_unit.sc.h).
    if (op_src.softmax) {
      while (do_softmax.read()) wait();
      softmax_n.write(din1.read().data.to_uint());
      softmax_scale_bits.write(din1.read().data.to_uint());
      softmax_slope_bits.write(din1.read().data.to_uint());
      softmax_flags.write(din1.read().data.to_uint());
      softmax_sink_bits.write(din1.read().data.to_uint());

      SIGWRITE(inS, 7);
      HWC_SIG(Control_Unit, 7);
      do_softmax.write(true);
      wait();
      while (do_softmax.read()) wait();
    }

    SIGWRITE(inS, 6);
    HWC_SIG(Control_Unit, 6);
    wait();
  }
}

#endif // __ACCNAME_CONTROL_UNIT_SC_H__
