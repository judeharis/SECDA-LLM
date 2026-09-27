#ifndef __ACCNAME_SCHEDULER_SC_H__
#define __ACCNAME_SCHEDULER_SC_H__

void ACCNAME::wait_bfpp_ready() { bfpp_check_ready(); }

void ACCNAME::bfpp_data_transfer() {
  input_transfer.write(1);
  weight_transfer.write(1);
  DWAIT();
  while (weight_transfer.read()) wait();
  while (input_transfer.read()) wait();
}

void ACCNAME::start_bfpp() {
  for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
    vars.bfpp_start_write(1, i);
    vars.wb_idx_write(mstep, i);
    vars.ib_idx_write(nstep, i);
    vars.kb_write(kb, i);
  }
  SIGWRITE(schS, 31);
  HWC_SIG(Scheduler, 31);
  wait();
  for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
    while (vars.bfpp_ready_read(i)) wait();
  }
  SIGWRITE(schS, 32);
  HWC_SIG(Scheduler, 32);
  wait();
  for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
    vars.bfpp_start_write(0, i);
  }
}

void ACCNAME::Scheduler() {

  SIGWRITE(schS, 0);
  HWC_SIG(Scheduler, 0);
  schedule.write(0);
  init_bfpp();
  wait();
  while (1) {
    SIGWRITE(schS, 1);
    HWC_SIG(Scheduler, 1);
    while (!schedule.read()) wait();

    SIGWRITE(schS, 2);
    HWC_SIG(Scheduler, 2);
    wait_bfpp_ready();
    wait();

    SIGWRITE(schS, 3);
    HWC_SIG(Scheduler, 3);
    start_bfpp();
    wait();

    SIGWRITE(schS, 4);
    HWC_SIG(Scheduler, 4);
    schedule.write(0);
    wait();
  }
}

#endif // __ACCNAME_SCHEDULER_SC_H__
