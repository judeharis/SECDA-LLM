#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#include <inttypes.h>
#define ACCNAME BFP_Acc
#define SUBMODULENAME SBVP_UNIT

//==============================================================================
// Hardware Constants
//==============================================================================
// Define any Hardware specific constants for the accelerator
// These constants will be accessible in the driver
// These constants will be used to generate the hardware

//==============================================================================
// Address mapping for the accelerator and DMA
//==============================================================================
#ifdef KRIA
// KRIA
// Pre-Defined Address for Accelerator
#define acc_ctrl_address 0xA0000000
#define acc_hwc_address 0xA0020000

#define acc_address 0x00A0000000
#define dma_addr0 0xA0010000
#define dma_in0 0x3A000000
#define dma_out0 0x38000000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x0000000037400000
#define DMA_RANGE_END 0x00000000773FFFFF
#define DMA_RANGE_OFFSET 0xC00000         // 1.5MB
#define DMA_RANGE_SIZE 0x40000000 // 1GB


#define DMA_IN_BUF_SIZE 0x3C000000 // 960 MB
#define DMA_OUT_BUF_SIZE 0x1000000 // 16 MB
#define CMA_BUF_TOTAL_SIZE (DMA_IN_BUF_SIZE + DMA_OUT_BUF_SIZE) // 976 MB
#define DMA_INP_SIZE 0x800000
#define DMA_WGT_SIZE (DMA_IN_BUF_SIZE - DMA_INP_SIZE) // 960 MB - 8 MB = 952 MB
#else
// Z1
// Pre-Defined Address for Accelerator
#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_in0 0x18000000
#define dma_out0 0x1C000000

#define DMA_IN_BUF_SIZE 0x7000000
#define DMA_OUT_BUF_SIZE 0x0500000
#define DMA_INP_SIZE 0x100000
#define DMA_WGT_SIZE (DMA_IN_BUF_SIZE - DMA_INP_SIZE)
#define DMA_RANGE_START 0x18000000
#define DMA_RANGE_END 0x1fffffff
#define DMA_RANGE_SIZE 0x8000000
#endif // KRIA

// AXIMM Constants
#ifdef KRIA
#define MM_BL 0x100000 // 1MB
#define in_addr 0x38000000
#define out_addr 0x39000000
#else
// Z1
#define MM_BL 0x100000 // 1MB
#define in_addr 0x18000000
#define out_addr 0x19000000
#endif

//==============================================================================
// Data types
//==============================================================================
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_TYPE sc_uint
#define s_mdma multi_dma<AXI_DWIDTH, 0>
#define mm_buf mm_buffer<unsigned long long>
#define mm_buf_float mm_buffer<float>

#define a_ctrl acc_ctrl<int>
#define h_ctrl hwc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

// MMIO Registers
#define HWC_Monitor_Count 5
#define CTRL_Reg_Count 9

// Number of PEs
#define BFPP_COUNT 1

// Quantization Varaints Supported
#define BFPP_QK2
#define BFPP_QK3
// #define BFPP_QK4
// #define BFPP_QK5
// #define BFPP_QK6

#define Q2_X64 11 // 10.5
#define Q3_X64 14 // 13.75
#define Q4_X64 18 // 18
#define Q5_X64 22 // 22
#define Q6_X64 27 // 26.25
#define Q8_X64 37 // 36.5

#define Q2_X32 21 // 21
#define Q3_X32 28 // 27.5
#define Q4_X32 36 // 36
#define Q5_X32 44 // 44
#define Q6_X32 53 // 52.5
#define Q8_X32 73 // 73

#define Q2_X8 84
#define Q3_X8 110
#define Q4_X8 144
#define Q5_X8 176
#define Q6_X8 210
#define Q8_X8 292

// Opcodes
#define OPCODE_LOAD_INP 0x1
#define OPCODE_LOAD_WGT 0x2
#define OPCODE_COMPUTE 0x4
#define OPCODE_STORE 0x8
#define OPCODE_CONFIG 0x10

// HW Parameters
#define I8_32 4
#define I16_32 2
#define QK_K 256
#define N_val 4096
// #define INP_BLCK 65
#define INP_BLCK 73
#define WGT_BLCK 28
// #define WGT_BLCK 21

#ifdef KRIA
#define uF 512
#define SUP_KMB 1024
#define SUP_KNB 512
#else
#define uF 512
#define SUP_KMB 1024
#define SUP_KNB 512
#endif

typedef struct {
  float d;                  // delta
  int8_t qs[QK_K];          // quants
  int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} acc_block_q8_K;

typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} acc_block_q6_K;

typedef struct {
  union {
    struct {
      uint16_t d;    // super-block scale for quantized scales
      uint16_t dmin; // super-block scale for quantized mins
    } sbs;
    uint32_t dm;
  };
  uint8_t scales[12];   // scales and mins, quantized with 6 bits
  uint8_t qh[QK_K / 8]; // quants, high bit
  uint8_t qs[QK_K / 2]; // quants, low 4 bits
} acc_block_q5_K;

typedef struct {
  union {
    struct {
      uint16_t d;    // super-block scale for quantized scales
      uint16_t dmin; // super-block scale for quantized mins
    } sbs;
    uint32_t dm;
  };
  uint8_t scales[12];   // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2]; // 4--bit quants
} acc_block_q4_K;

typedef struct {
  uint8_t hmask[QK_K / 8]; // quants - high bit
  uint8_t qs[QK_K / 4];    // quants - low 2 bits
  uint8_t scales[12];      // scales, quantized with 6 bits
  uint16_t d;              // super-block scale
} acc_block_q3_K;

typedef struct {
  uint8_t scales[QK_K / 16]; // 4 bit scales and 4 bits mins
  uint8_t qs[QK_K / 4];      // quants
  union {
    struct {
      uint16_t d;    // super-block scale for quantized scales
      uint16_t dmin; // super-block scale for quantized mins
    } sbs;
    uint32_t dm;
  };
} acc_block_q2_K;

//==============================================================================
// SystemC Specfic SIM/HW Configurations
//==============================================================================
#if defined(SYSC) || defined(__SYNTHESIS__)
#include <systemc.h>

#ifndef __SYNTHESIS__
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_profiler/profiler.h"
#define DWAIT(x) wait(x)

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else // !VERBOSE_ACC
#define ALOG(x)
#endif

typedef _BDATA<AXI_DWIDTH, AXI_TYPE> ADATA;

#else // __SYNTHESIS__
#include "sysc_types.h"
#define ALOG(x)

struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _NDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

typedef _NDATA ADATA;
#endif

//==============================================================================
// HW Structs
//==============================================================================

struct opcode {
  unsigned int packet;
  bool load_inp;
  bool load_wgt;
  bool compute;
  bool store;
  bool config;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    load_inp = _packet.range(0, 0);
    load_wgt = _packet.range(1, 1);
    compute = _packet.range(2, 2);
    store = _packet.range(3, 3);
    config = _packet.range(4, 4);
  }

  opcode() {
    packet = 0;
    load_inp = false;
    load_wgt = false;
    compute = false;
    store = false;
    config = false;
  }
};

struct wgt_packet {
  unsigned int kmb;
  unsigned int wgt_blck;
  unsigned int k_size;
  unsigned int m_size;
  unsigned int m_block;
  unsigned int m_block_rem;


  
  wgt_packet(sc_fifo_in<ADATA> *din) {
    ALOG("WGT_PACKET");
    ALOG("Time: " << sc_time_stamp());
    kmb = din->read().data;
    wgt_blck = din->read().data;
  }
};

struct inp_packet {
  unsigned int knb;
  inp_packet(sc_fifo_in<ADATA> *din) {
    ALOG("INP_PACKET");
    ALOG("Time: " << sc_time_stamp());
    knb = din->read().data;
  }
};

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================

struct BFPP_vars {
  // Signal declarations for the hardware submodule
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> load_inp;
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> bfpp_start;
  sc_signal<bool, SC_MANY_WRITERS> bfpp_ready;
  sc_signal<unsigned int, SC_MANY_WRITERS> kb;
  sc_signal<unsigned int, SC_MANY_WRITERS> wbs;
  sc_signal<unsigned int, SC_MANY_WRITERS> ibs;
  sc_signal<unsigned int, SC_MANY_WRITERS> wb_idx;
  sc_signal<unsigned int, SC_MANY_WRITERS> ib_idx;
  sc_signal<unsigned int, SC_MANY_WRITERS> wgt_type;
#else
  sc_signal<bool> load_inp;
  sc_signal<bool> load_wgt;
  sc_signal<bool> bfpp_start;
  sc_signal<bool> bfpp_ready;
  sc_signal<unsigned int> kb;
  sc_signal<unsigned int> wbs;
  sc_signal<unsigned int> ibs;
  sc_signal<unsigned int> wb_idx;
  sc_signal<unsigned int> ib_idx;
  sc_signal<unsigned int> wgt_type;
#endif

  // Declare I/O ports and fifos for the hardware submodule
  sc_fifo<ADATA> wgt_fifo;
  sc_fifo<ADATA> inp_fifo;
  sc_fifo<ADATA> dout1;
  sc_out<unsigned int> computeSS;
  sc_out<unsigned int> wgtlSS;
  sc_out<unsigned int> inplSS;

#ifndef __SYNTHESIS__
  BFPP_vars(int size, int sid)
      : INITSIGPORT(load_inp, sid), INITSIGPORT(load_wgt, sid),
        INITSIGPORT(bfpp_start, sid), INITSIGPORT(bfpp_ready, sid),
        INITSIGPORT(kb, sid), INITSIGPORT(wbs, sid), INITSIGPORT(ibs, sid),
        INITSIGPORT(wb_idx, sid), INITSIGPORT(ib_idx, sid),
        INITSIGPORT(wgt_type, sid), INITSIGPORT(computeSS, sid),
        INITSIGPORT(wgtlSS, sid), INITSIGPORT(inplSS, sid),
        wgt_fifo("wgt_fifo", size), inp_fifo("inp_fifo", size),
        dout1("dout1", size) {}

#else
  BFPP_vars(int size)
      : load_inp("load_inp"), load_wgt("load_wgt"), bfpp_start("bfpp_start"),
        bfpp_ready("bfpp_ready"), kb("kb"), wbs("wbs"), ibs("ibs"),
        wb_idx("wb_idx"), ib_idx("ib_idx"), wgt_type("wgt_type"),
        computeSS("computeSS"), wgtlSS("wgtlSS"), inplSS("inplSS"),
        wgt_fifo(size), inp_fifo(size), dout1(size) {

// Define any HLS pragma for ports/fifos declared here
#pragma HLS resource variable = wgt_fifo core = FIFO_SRL
#pragma HLS resource variable = inp_fifo core = FIFO_SRL
#pragma HLS resource variable = dout1 core = FIFO_SRL
  }
#endif
};

//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H