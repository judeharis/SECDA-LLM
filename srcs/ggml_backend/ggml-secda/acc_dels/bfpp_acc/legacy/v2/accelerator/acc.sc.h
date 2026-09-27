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

  sc_out<unsigned int> inSS;
  sc_out<unsigned int> loadSS;
  sc_out<unsigned int> outSS;
  sc_out<unsigned int> schSS;
  sc_out<unsigned int> wgttSS;
  sc_out<unsigned int> inptSS;

  // ================================================= //
  // Global variables
  // ================================================= //

  sc_signal<int> kb;    // Number of super-blocks per K
  sc_signal<int> M;     // M dim size
  sc_signal<int> N;     // N dim size
  sc_signal<int> mstep; // M dim mstep
  sc_signal<int> nstep; // N dim nstep
  sc_signal<int> kmb;   // current mstep * kb
  sc_signal<int> knb;   // current nstep * kb
  int wgt_type;         // weight type
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
  sc_fifo<ADATA> wgt_fifo;

  // ================================================= //
  // Global signals
  // ================================================= //
  DEFINE_SC_SIGNAL(bool, data_load);
  DEFINE_SC_SIGNAL(bool, start_compute);
  DEFINE_SC_SIGNAL(bool, schedule);
  DEFINE_SC_SIGNAL(bool, weight_transfer);
  DEFINE_SC_SIGNAL(bool, input_transfer);

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
  HWC_CTHREAD(Weight_Transfer)

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
        HWC_Logic(Weight_Transfer);
        DWAIT();
      }
    }
  }
  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(name_), wgt_fifo("wgt_fifo", 4096 * 6) {

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

    SC_CTHREAD(Weight_Transfer, clock);
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

CTRL_Prag(vars.vars_0.computeSS);
CTRL_Prag(vars.vars_0.wgtlSS);
CTRL_Prag(vars.vars_0.inplSS);

AXI4S_In_Prag(din1);
AXI4S_Out_Prag(dout1);

HWC_PragReset;
HWC_PragGroup(Control_Unit)
HWC_PragGroup(Load_Unit)
HWC_PragGroup(Store_Unit)
HWC_PragGroup(Scheduler);
HWC_PragGroup(Weight_Transfer);
    // clang-format on
  }
};

#endif