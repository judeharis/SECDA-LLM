#ifndef __ACCNAME_BFPP_SC_H__
#define __ACCNAME_BFPP_SC_H__

#include "acc_config.sc.h"

SC_MODULE(BFPP_UNIT) {
  // IO ports
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<bool> load_inp;
  sc_in<bool> load_wgt;
  sc_in<bool> bfpp_start;
  sc_out<bool> bfpp_ready;

  sc_in<unsigned int> kb;       // blocks per K dim
  sc_in<unsigned int> wbs;      // weight blocks
  sc_in<unsigned int> ibs;      // input blocks
  sc_in<unsigned int> wb_idx;   // compute wb index
  sc_in<unsigned int> ib_idx;   // compute ib index
  sc_in<unsigned int> wgt_type; // weight type

  // FIFOs
  sc_fifo_in<ADATA> wgt_fifo1;
  sc_fifo_in<ADATA> wgt_fifo2;
  sc_fifo_in<ADATA> wgt_fifo3;
  sc_fifo_in<ADATA> wgt_fifo4;

  sc_fifo_in<ADATA> inp_fifo;
  sc_fifo_out<ADATA> dout1;

  // Status Signals
  sc_signal<bool> bfpp_free;

  // Memory
  // Weight SUP_KMB super-blocks (3-bit quantization)

#if defined(BFPP_QK3) || defined(BFPP_QK5)
  bool w_hmask[SUP_KMB][QK_K]; // quants - high bit
#endif

#if defined(BFPP_QK6)
  sc_uint<2> w_qs3[SUP_KMB][QK_K]; // quants - low - high - high 2 bits
#endif

#if defined(BFPP_QK4) || defined(BFPP_QK5) || defined(BFPP_QK6)
  sc_uint<2> w_qs2[SUP_KMB][QK_K]; // quants - low - high  2 bits
#endif

  sc_uint<2> w_qs[SUP_KMB][QK_K];          // quants - low - low 2 bits
  sc_uint<8> w_scales[SUP_KMB][QK_K / 16]; // scales, quantized with 6-8 bits
  uint16_t w_d[SUP_KMB];                   // super-block scale
  uint16_t w_dmin[SUP_KMB];                // super-block mins

  // Input SUP_KNB super-blocks
  float i_d[SUP_KNB];              // super-block scale
  sc_int<8> i_qs[SUP_KNB][QK_K];   // quants
  sc_int<16> i_bsums[SUP_KNB][16]; // input block sums

  // Debug
  sc_signal<unsigned int> computeS;
  sc_out<unsigned int> computeSS;

  sc_signal<unsigned int> wgtlS;
  sc_out<unsigned int> wgtlSS;

  sc_signal<unsigned int> inplS;
  sc_out<unsigned int> inplSS;

  // Functions
  float vec_dot(int, int);
  float vec_dot_unroll_2(int wi, int ii, int ii2);
  float ggml_compute_fp16_to_fp32(uint16_t h);
  uint32_t fp32_to_bits(float f);
  float fp32_from_bits(uint32_t w);
  void u96_to_u6x16(sc_biguint<96> in, sc_uint<8> * out);
  void u96_to_u6x8x2(sc_biguint<96> in, sc_uint<8> * out);
  // void weight_mapper(int wb);

  void weight_mapper(int wb, sc_fifo_in<ADATA> &wgt_fifo);
  void weight_mapper2(int wb);
  void weight_mapper2x(int wb, unsigned int wbs_l1, unsigned int wbs_l2);
  void weight_mapper4x(int wb, unsigned int wbs_l1, unsigned int wbs_l2,
                       unsigned int wbs_l3, unsigned int wbs_l4);

  // Modules
  void LoadWeights();
  void LoadInputs();
  void Compute();

  // Constructors
  void init(sc_in<bool> & clock, sc_in<bool> & reset, BFPP_vars & vars) {
    this->clock(clock);
    this->reset(reset);

    this->load_inp(vars.load_inp);
    this->load_wgt(vars.load_wgt);

    this->bfpp_start(vars.bfpp_start);
    this->bfpp_ready(vars.bfpp_ready);

    this->kb(vars.kb);
    this->wbs(vars.wbs);
    this->ibs(vars.ibs);
    this->wb_idx(vars.wb_idx);
    this->ib_idx(vars.ib_idx);
    this->wgt_type(vars.wgt_type);

    this->wgt_fifo1(vars.wgt_fifo1);
    this->wgt_fifo2(vars.wgt_fifo2);
    this->wgt_fifo3(vars.wgt_fifo3);
    this->wgt_fifo4(vars.wgt_fifo4);
    this->inp_fifo(vars.inp_fifo);
    this->dout1(vars.dout1);

    this->wgtlSS(vars.wgtlSS);
    this->inplSS(vars.inplSS);
    this->computeSS(vars.computeSS);
  }

  SC_HAS_PROCESS(BFPP_UNIT);

  BFPP_UNIT(sc_module_name name_) : sc_module(name_) {

    SC_CTHREAD(LoadWeights, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(LoadInputs, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

#if defined(BFPP_QK3) || defined(BFPP_QK5)
#pragma HLS array_partition variable = w_hmask dim = 1 cyclic factor = 2
#pragma HLS array_partition variable = w_hmask dim = 2 cyclic factor = 16

#endif

#if defined(BFPP_QK4) || defined(BFPP_QK5) || defined(BFPP_QK6)
#pragma HLS array_partition variable = w_qs2 dim = 1 cyclic factor = 2
#pragma HLS array_partition variable = w_qs2 dim = 2 cyclic factor = 16
#pragma HLS RESOURCE variable = w_qs2 core = RAM_T2P_BRAM
#endif

#if defined(BFPP_QK6)
#pragma HLS array_partition variable = w_qs3 dim = 1 cyclic factor = 2
#pragma HLS array_partition variable = w_qs3 dim = 2 cyclic factor = 16
#pragma HLS RESOURCE variable = w_qs3 core = RAM_T2P_BRAM
#endif

#pragma HLS array_partition variable = w_qs dim = 1 cyclic factor = 2
#pragma HLS array_partition variable = w_qs dim = 2 cyclic factor = 16
#pragma HLS array_partition variable = w_scales dim = 1 cyclic factor = 2
#pragma HLS array_partition variable = w_scales dim = 2 cyclic factor = 16
#pragma HLS array_partition variable = w_d dim = 1 cyclic factor = 4
#pragma HLS array_partition variable = w_dmin dim = 1 cyclic factor = 4

#pragma HLS RESOURCE variable = w_qs core = RAM_T2P_URAM
#pragma HLS RESOURCE variable = w_scales core = RAM_T2P_BRAM
#pragma HLS RESOURCE variable = w_d core = RAM_T2P_BRAM
#pragma HLS RESOURCE variable = w_dmin core = RAM_T2P_BRAM

#pragma HLS array_partition variable = i_bsums dim = 2 cyclic factor = 16
#pragma HLS array_partition variable = i_qs dim = 2 cyclic factor = 16
#pragma HLS array_partition variable = i_d dim = 1 cyclic factor = 16
  }
};

struct var_array {
  BFPP_vars vars_0;
  BFPP_UNIT P0;

#ifndef __SYNTHESIS__
  var_array() : vars_0(16, 0), P0("P0") {}
#else
  var_array() : vars_0(16), P0("P0") {}
#endif

  // Status/Control Access
  void load_inp_write(bool data, int index) { vars_0.load_inp.write(data); }

  void load_wgt_write(bool data, int index) { vars_0.load_wgt.write(data); }

  void bfpp_start_write(bool data, int index) { vars_0.bfpp_start.write(data); }

  bool bfpp_ready_read(int index) { return vars_0.bfpp_ready.read(); }

  void kb_write(unsigned int data, int index) { vars_0.kb.write(data); }

  void wbs_write(unsigned int data, int index) { vars_0.wbs.write(data); }

  void ibs_write(unsigned int data, int index) { vars_0.ibs.write(data); }

  void wb_idx_write(unsigned int data, int index) { vars_0.wb_idx.write(data); }

  void ib_idx_write(unsigned int data, int index) { vars_0.ib_idx.write(data); }

  void wgt_type_write(unsigned int data, int index) {
    vars_0.wgt_type.write(data);
  }

  // FIFO Access
  void wgt1_write(ADATA data, int index) { vars_0.wgt_fifo1.write(data); }
  void wgt2_write(ADATA data, int index) { vars_0.wgt_fifo2.write(data); }
  void wgt3_write(ADATA data, int index) { vars_0.wgt_fifo3.write(data); }
  void wgt4_write(ADATA data, int index) { vars_0.wgt_fifo4.write(data); }

  void inp_write(ADATA data, int index) { vars_0.inp_fifo.write(data); }

  ADATA dout_read(int index) { return vars_0.dout1.read(); }

  // Module Access
  BFPP_vars &operator[](int index) { return vars_0; }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    P0.init(clock, reset, vars_0);
  }
};

#endif // __ACCNAME_BFPP_SC_H__
