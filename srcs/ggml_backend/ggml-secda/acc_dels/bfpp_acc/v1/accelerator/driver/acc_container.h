#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <cassert>
#include <cmath>
#include <iomanip>
#include <vector>

#ifdef SYSC
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "secda_tools/secda_utils/acc_helpers.h"
#include "secda_tools/secda_utils/multi_threading.h"
#include "secda_tools/secda_utils/utils.h"

// #define ACC_PRELOAD
#ifdef ACC_PRELOAD
#define LAYER_PREALLOC 500
#else
#define LAYER_PREALLOC 0
#endif

// #define PRINT_FILE
// #define ACC_NEON
#ifdef ACC_NEON

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
// #include <arm_neon.h>
#include "arm_neon.h"
#else
#include "NEON_2_SSE.h"
#endif

#endif

using namespace std;
using namespace std::chrono;
#define TSCALE microseconds
#define TSCAST duration_cast<nanoseconds>

struct acc_times {
  duration_ns driver_total;
  duration_ns fpga_total;
  duration_ns fpga_wgt_send;
  duration_ns fpga_inp_send;
  duration_ns fpga_compute;
  duration_ns fpga_inp_pack;
  duration_ns fpga_wgt_pack;
  duration_ns fpga_wgt_pack_opt;
  duration_ns fpga_compute_cycles;
  duration_ns fpga_weight_transfer_cycles;
  duration_ns esti_weight_transfer_cycles;
  duration_ns layer_total;

  void print() {
#ifdef ACC_PROFILE
    cerr << "================================================" << endl;
    prf_out(TSCALE, driver_total);
    prf_out(TSCALE, fpga_total);
    prf_out(TSCALE, fpga_wgt_send);
    prf_out(TSCALE, fpga_inp_send);
    prf_out(TSCALE, fpga_compute);
    prf_out(TSCALE, fpga_inp_pack);
    prf_out(TSCALE, fpga_wgt_pack);
    prf_out(TSCALE, fpga_wgt_pack_opt);
    prf_out(TSCALE, fpga_compute_cycles);
    prf_out(TSCALE, fpga_weight_transfer_cycles);
    prf_out(TSCALE, esti_weight_transfer_cycles);
    prf_out(TSCALE, layer_total);

    cerr << "================================================" << endl;
#endif
  }
  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    time_t now = time(0);
    char *dt = ctime(&now);
    dt[strlen(dt) - 1] = '\0';
    file << "driver_total,fpga_wgt_send,fpga_inp_send,fpga_"
            "compute,fpga_inp_pack,fpga_wgt_pack,fpga_wgt_pack_opt,"
            "fpga_compute_cycles,fpga_weight_transfer_cycles,esti_weight_transfer_cycles,layer_total"
         << endl;
    // file << dt << ",";
    prf_file_out_x(TSCALE, driver_total, file);
    // prf_file_out_x(TSCALE, fpga_total, file);
    prf_file_out_x(TSCALE, fpga_wgt_send, file);
    prf_file_out_x(TSCALE, fpga_inp_send, file);
    prf_file_out_x(TSCALE, fpga_compute, file);
    prf_file_out_x(TSCALE, fpga_inp_pack, file);
    prf_file_out_x(TSCALE, fpga_wgt_pack, file);
    prf_file_out_x(TSCALE, fpga_wgt_pack_opt, file);
    prf_file_out_x(TSCALE, fpga_compute_cycles, file);
    prf_file_out_x(TSCALE, fpga_weight_transfer_cycles, file);
    prf_file_out_x(TSCALE, esti_weight_transfer_cycles, file);
    prf_file_out_l(TSCALE, layer_total, file);
    file << endl;
    file.close();
#endif
  }

  ~acc_times() {
#ifdef ACC_PROFILE
    cerr << "================================================" << endl;
    cerr << endl;
    prf_out(TSCALE, driver_total);
    // prf_out(TSCALE, fpga_total);
    prf_out(TSCALE, fpga_wgt_send);
    prf_out(TSCALE, fpga_inp_send);
    prf_out(TSCALE, fpga_compute);
    prf_out(TSCALE, fpga_inp_pack);
    prf_out(TSCALE, fpga_wgt_pack);
    prf_out(TSCALE, fpga_wgt_pack_opt);
    prf_out(TSCALE, fpga_compute_cycles);
    prf_out(TSCALE, fpga_weight_transfer_cycles);
    prf_out(TSCALE, esti_weight_transfer_cycles);
    prf_out(TSCALE, layer_total);
    cerr << endl;
    cerr << "================================================" << endl;
    save_prf();
#endif
  }
};

struct model_info {
  int supported_nodes = 0;
  vector<char *> wgt_ptrs;
  vector<unsigned int> wgt_sizes;
  vector<unsigned int> dim_M;
  vector<unsigned int> dim_K;
  bool planned = false;
};

struct layer_details {
  int layer = 0;
  bool profile = false;

  unsigned int alloced_layers = 0;
  unsigned int last_alloced_layer = 0;
  bool alloc_allowed = true;

  int layer_offsets[500];
  bool layer_alloced[500];
  std::vector<int> alloced_buffer_ids;
  unsigned int curr_offset = 0;

  bool alloc_layer(int layer, unsigned int wgt_size) {
    if (!alloc_allowed || (curr_offset + wgt_size > DMA_WGT_SIZE) ||
        alloced_layers >= LAYER_PREALLOC) {
      alloc_allowed = false;
      return false;
    }
    layer_offsets[layer] = curr_offset;
    layer_alloced[layer] = true;
    alloced_layers++;
    return true;
  }

  char *get_buffer_from_layer(int layer, int *DMA_W) {
    int offset = layer_offsets[layer];
    if (offset == -1) {
      cout << "Layer: " << layer << " was not allocated" << endl;
      return NULL;
    }
    char *buffer = reinterpret_cast<char *>(DMA_W);
    return buffer + offset;
  }
};

struct alloced_layers {
  unsigned int layer;
  unsigned int wgt_size;
  unsigned int M;
  unsigned int K;
  const void *wgt;
};

struct acc_container {
// Hardware
#ifdef SYSC
  ACCNAME *acc;
  struct sysC_sigs *scs;
#else
  int *acc;
#endif

  struct a_ctrl *ctrl;
  struct h_ctrl *hwc;
  struct s_mdma *mdma;
  Profile *profile;
  MultiThreadContext *mt_context;

  // Accelerator Specific Parameters
  // Data
  int M;
  int N;
  int K;
  int inp_stride;
  int wgt_stride;
  int out_stride;

  char *inp;
  char *wgt;
  char *out;

  int wgt_type;

  // DMA Buffers
  int *DMA_I;
  int *DMA_W;

  // Debugging
  struct layer_details t;
  struct model_info m;
  struct acc_times *a_t;
  int n = 0;
  int wgt_blk = 0;
  int inp_blk = 0;
};

#endif // ACC_CONTAINER