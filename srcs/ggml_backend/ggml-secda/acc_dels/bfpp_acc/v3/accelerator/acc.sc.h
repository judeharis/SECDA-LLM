#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include "bfp_processor_top.sc.h"

// #include "hwc.sc.h"
#include <systemc.h>

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  // ================================================= //
  // Global ports
  // ================================================= //

  // Control ports
  CTRL_Define_Ports;

  // Data ports

  sc_fifo_in<ADATA> din1;
  sc_fifo_out<ADATA> dout1;

  sc_fifo_in<ADATA> din2;
  sc_fifo_out<ADATA> dout2;

  sc_fifo_in<ADATA> din3;
  sc_fifo_out<ADATA> dout3;

  sc_fifo_in<ADATA> din4;
  sc_fifo_out<ADATA> dout4;

  sc_out<unsigned int> inSS;
  sc_out<unsigned int> loadSS;
  sc_out<unsigned int> outSS;
  sc_out<unsigned int> schSS;
  sc_out<unsigned int> wgttSS;
  sc_out<unsigned int> inptSS;

  // ================================================= //
  // Global variables
  // ================================================= //

  sc_signal<int> kb;       // Number of super-blocks per K
  sc_signal<int> M;        // M dim size
  sc_signal<int> N;        // N dim size
  sc_signal<int> mstep;    // M dim mstep
  sc_signal<int> nstep;    // N dim nstep
  sc_signal<int> kmb;      // current mstep * kb
  sc_signal<int> knb;      // current nstep * kb
  sc_signal<int> wgt_blck; // weight block size in number of elements

  sc_signal<int> weightloader_A_len;
  sc_signal<int> weightloader_B_len;
  sc_signal<int> weightloader_C_len;
  sc_signal<int> weightloader_D_len;

  int wgt_type; // weight type
  opcode op_src;

  // ================================================= //
  // Global buffers
  // ================================================= //
  // unsigned long long input[INPUT_SIZE];
  // unsigned long long weight[INPUT_SIZE];
  // unsigned long long output[INPUT_SIZE];

  // ================================================= //
  // Global fifos
  // ================================================= //
  // sc_fifo<sc_uint<64>> input_fifo;
  sc_fifo<ADATA> glb_wf_1;
  sc_fifo<ADATA> glb_wf_2;
  sc_fifo<ADATA> glb_wf_3;
  sc_fifo<ADATA> glb_wf_4;

  // ================================================= //
  // Global signals
  // ================================================= //
  DEFINE_SC_SIGNAL(bool, data_load);
  DEFINE_SC_SIGNAL(bool, start_compute);
  DEFINE_SC_SIGNAL(bool, schedule);
  DEFINE_SC_SIGNAL(bool, weight_transfer);
  DEFINE_SC_SIGNAL(bool, input_transfer);

  DEFINE_SC_SIGNAL(bool, weightloader_A);
  DEFINE_SC_SIGNAL(bool, weightloader_B);
  DEFINE_SC_SIGNAL(bool, weightloader_C);
  DEFINE_SC_SIGNAL(bool, weightloader_D);

  DEFINE_SC_SIGNAL(int, inS);
  DEFINE_SC_SIGNAL(int, loadS);
  DEFINE_SC_SIGNAL(int, outS);
  DEFINE_SC_SIGNAL(int, schS);
  DEFINE_SC_SIGNAL(int, wgttS);
  DEFINE_SC_SIGNAL(int, inptS);

  // ================================================= //
  // Submodules
  // ================================================= //

  struct var_array vars;

  // ================================================= //
  // Functions
  // ================================================= //

  unsigned int qtypeToSize(unsigned int qtype);
  void wait_bfpp_ready();
  void bfpp_check_ready();
  void start_weight_transfer();
  void start_input_transfer();
  void bfpp_data_transfer();
  void init_bfpp();
  void start_bfpp();

  // ================================================= //
  // Profiling variable
  // ================================================= //

#ifndef __SYNTHESIS__
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles};
#endif
  // ================================================= //
  // HWC
  // ================================================= //

  HWC_Reset;
  HWC_CTHREAD(Control_Unit)
  HWC_CTHREAD(Load_Unit)
  HWC_CTHREAD(Store_Unit)
  HWC_CTHREAD(Scheduler)
  HWC_CTHREAD(Weight_Transfer_A)
  HWC_CTHREAD(Weight_Transfer_B)
  HWC_CTHREAD(Weight_Transfer_C)
  HWC_CTHREAD(Weight_Transfer_D)
  HWC_CTHREAD(WeightLoader_A)
  HWC_CTHREAD(WeightLoader_B)
  HWC_CTHREAD(WeightLoader_C)
  HWC_CTHREAD(WeightLoader_D)
  HWC_CTHREADSub(X1_Compute, vars.vars_0.ComputeSS);
  HWC_CTHREADSub(X1_LoadWeight, vars.vars_0.LoadWeightSS);
  HWC_CTHREADSub(X1_LoadInput, vars.vars_0.LoadInputSS);

  void HW_MAIN() {
    wait();
    while (true) {
      {
#pragma HLS LATENCY max = 0 min = 0
#pragma HLS protocol fixed
        HWC_Logic(Control_Unit);
        HWC_Logic(Load_Unit);
        HWC_Logic(Store_Unit);
        HWC_Logic(Scheduler);
        HWC_Logic(Weight_Transfer_A);
        HWC_Logic(Weight_Transfer_B);
        HWC_Logic(Weight_Transfer_C);
        HWC_Logic(Weight_Transfer_D);
        HWC_Logic(WeightLoader_A);
        HWC_Logic(WeightLoader_B);
        HWC_Logic(WeightLoader_C);
        HWC_Logic(WeightLoader_D);
        HWC_Logic(X1_Compute);
        HWC_Logic(X1_LoadWeight);
        HWC_Logic(X1_LoadInput);
        DWAIT();
      }
    }
  }
  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(name_), glb_wf_1("glb_wf_1", 4096 * 1),
        glb_wf_2("glb_wf_2", 4096 * 1), glb_wf_3("glb_wf_3", 4096 * 1),
        glb_wf_4("glb_wf_4", 4096 * 1) {

    // Connect PE ports
    vars.init(clock, reset);

    SC_CTHREAD(Control_Unit, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Load_Unit, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Store_Unit, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Weight_Transfer_A, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Weight_Transfer_B, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Weight_Transfer_C, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Weight_Transfer_D, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WeightLoader_A, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WeightLoader_B, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WeightLoader_C, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WeightLoader_D, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(X1_Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(X1_LoadWeight, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(X1_LoadInput, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(HW_MAIN, clock);
    reset_signal_is(reset, true);

    // clang-format off
CTRL_PragGroup;
CTRL_Prag(inSS);
CTRL_Prag(loadSS);
CTRL_Prag(outSS);
CTRL_Prag(schSS);
CTRL_Prag(wgttSS);
CTRL_Prag(inptSS);

AXI4S_In_Prag1(din1);
AXI4S_In_Prag2(din2);
AXI4S_In_Prag3(din3);
AXI4S_In_Prag4(din4);

AXI4S_Out_Prag1(dout1);
AXI4S_Out_Prag2(dout2);
AXI4S_Out_Prag3(dout3);
AXI4S_Out_Prag4(dout4);

HWC_PragReset;
HWC_PragGroup(Control_Unit)
HWC_PragGroup(Load_Unit)
HWC_PragGroup(Store_Unit)
HWC_PragGroup(Scheduler);
HWC_PragGroup(Weight_Transfer_A);
HWC_PragGroup(Weight_Transfer_B);
HWC_PragGroup(Weight_Transfer_C);
HWC_PragGroup(Weight_Transfer_D);
HWC_PragGroup(WeightLoader_A);
HWC_PragGroup(WeightLoader_B);
HWC_PragGroup(WeightLoader_C);
HWC_PragGroup(WeightLoader_D);
HWC_PragGroup(X1_Compute);
HWC_PragGroup(X1_LoadWeight);
HWC_PragGroup(X1_LoadInput);
    // clang-format on
  }
};

#endif