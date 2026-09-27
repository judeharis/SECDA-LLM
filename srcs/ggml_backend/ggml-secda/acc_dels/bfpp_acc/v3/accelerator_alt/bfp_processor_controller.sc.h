#ifndef __ACCNAME_BFP_PROCESSOR_CONTROLLER_SC_H__
#define __ACCNAME_BFP_PROCESSOR_CONTROLLER_SC_H__


void ACCNAME::bfpp_check_ready() {
  for (int i = 0; i < BFPP_COUNT; i++) {
    while (!vars.bfpp_ready_read(i)) wait();
  }
}

void ACCNAME::init_bfpp() {
#pragma HLS inline OFF
  for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
    vars[i].load_inp.write(false);
    vars[i].load_wgt.write(false);
    vars[i].bfpp_start.write(false);
    vars[i].kb.write(0);
    vars[i].wbs.write(0);
    vars[i].ibs.write(0);
    vars[i].wb_idx.write(0);
    vars[i].ib_idx.write(0);
  }
  DWAIT();
}

#endif // __ACCNAME_BFP_PROCESSOR_CONTROLLER_SC_H__