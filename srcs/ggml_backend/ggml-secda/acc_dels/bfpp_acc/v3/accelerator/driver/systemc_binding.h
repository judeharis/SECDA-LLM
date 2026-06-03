#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#ifdef SYSC

#include "../acc.sc.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to VM SystemC definition
// This contains all the correct port/signal bindings to instantiate the VM
// accelerator
struct sysC_sigs {
  int id;
  Clock_Reset_Define;
  sc_fifo<ADATA> dout1;
  sc_fifo<ADATA> din1;
  sc_fifo<ADATA> dout2;
  sc_fifo<ADATA> din2;
  sc_fifo<ADATA> dout3;
  sc_fifo<ADATA> din3;
  sc_fifo<ADATA> dout4;
  sc_fifo<ADATA> din4;

  sysC_sigs(int id_)
      : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800),
        dout2("dout2_fifo", 563840), din2("din2_fifo", 554800),
        dout3("dout3_fifo", 563840), din3("din3_fifo", 554800),
        dout4("dout4_fifo", 563840), din4("din4_fifo", 554800) {
    sc_clock clk_clock("ClkClock", 1, SC_NS);
    id = id_;
  }
};

void sysC_binder(ACCNAME *acc, sysC_sigs *scs, a_ctrl *ctrl, h_ctrl *hwc,
                 s_mdma *mdma) {

  Clock_Reset_Bind(acc, scs);
  Clock_Reset_Bind(ctrl->ctrl, scs);
  Clock_Reset_Bind(hwc->hwc_resetter, scs);

  CTRL_Bind_CtrlSignals(acc, ctrl);
  CTRL_Bind_RegSignals(inSS);
  CTRL_Bind_RegSignals(loadSS);
  CTRL_Bind_RegSignals(outSS);
  CTRL_Bind_RegSignals(schSS);
  CTRL_Bind_RegSignals(wgttSS);
  CTRL_Bind_RegSignals(inptSS);

  acc->dout1(scs->dout1);
  acc->din1(scs->din1);
  acc->dout2(scs->dout2);
  acc->din2(scs->din2);
  acc->dout3(scs->dout3);
  acc->din3(scs->din3);
  acc->dout4(scs->dout4);
  acc->din4(scs->din4);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_clock);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  mdma->dmas[0].dmad->din1(scs->din1);
  mdma->dmas[1].dmad->dout1(scs->dout2);
  mdma->dmas[1].dmad->din1(scs->din2);
  mdma->dmas[2].dmad->dout1(scs->dout3);
  mdma->dmas[2].dmad->din1(scs->din3);
  mdma->dmas[3].dmad->dout1(scs->dout4);
  mdma->dmas[3].dmad->din1(scs->din4);

  HWC_Bind_Reset;
  HWC_Bind_Signals(Control_Unit);
  HWC_Bind_Signals(Load_Unit);
  HWC_Bind_Signals(Store_Unit);
  HWC_Bind_Signals(Scheduler);
  HWC_Bind_Signals(Weight_Transfer_A);
  HWC_Bind_Signals(Weight_Transfer_B);
  HWC_Bind_Signals(Weight_Transfer_C);
  HWC_Bind_Signals(Weight_Transfer_D);
  HWC_Bind_Signals(WeightLoader_A);
  HWC_Bind_Signals(WeightLoader_B);
  HWC_Bind_Signals(WeightLoader_C);
  HWC_Bind_Signals(WeightLoader_D);
  HWC_Bind_Signals(X1_Compute);
  HWC_Bind_Signals(X1_LoadWeight);
  HWC_Bind_Signals(X1_LoadInput);
}
#endif // SYSC

#endif // SYSTEMC_BINDING