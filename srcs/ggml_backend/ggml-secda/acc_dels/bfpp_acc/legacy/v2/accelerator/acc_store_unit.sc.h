#ifndef __ACCNAME_STORE_UNIT_SC_H__
#define __ACCNAME_STORE_UNIT_SC_H__

void ACCNAME::Store_Unit() {

  SIGWRITE(outS, 0);
  HWC_SIG(Store_Unit, 0);
  
  int data[BFPP_COUNT];
  ADATA d = {0, 0};
  wait();
  while (1) {
    SIGWRITE(outS, 1);
    HWC_SIG(Store_Unit, 1);
    for (int i = 0; i < BFPP_COUNT; i++) {
#pragma HLS unroll
      d = vars.dout_read(i);
      dout1.write(d);
    }
    DWAIT();
  }
}

#endif // __ACCNAME_STORE_UNIT_SC_H__