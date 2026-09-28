#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <cassert>
#include <cmath>
#include <iomanip>
#include <map>
#include <vector>

#ifdef SYSC
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "secda-core/axi_support/v6/axi_api.h"
#include "secda-core/secda_profiler/secda_profile_writer.h"
#include "secda-core/secda_profiler/profiler.h"
#include "secda-core/secda_utils/acc_helpers.h"
#include "secda-core/secda_utils/utils.h"


// #define TOG2(X) threadsafe_cout(X)
#define TOG2(X)

#define STR(X) std::to_string(X)
#define HEX(X) std::hex << X << std::dec

// #define DRIVER_PRINT
#ifdef DRIVER_PRINT
#define DRIVER_COUT std::cout
#else
#define DRIVER_COUT                                                             \
  if (false) std::cout
#endif

#ifdef ACC_PRELOAD
#define LAYER_PREALLOC true
#else
#define LAYER_PREALLOC false
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

  // Hardware counters summed over every accelerator call (they are reset per
  // call), for secda_profile.json; see add_hwc().
  std::vector<HwcSample> hwc_totals;

  void add_hwc(const std::vector<HwcSample> &samples) {
    if (hwc_totals.size() < samples.size()) hwc_totals.resize(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
      HwcSample &t = hwc_totals[i];
      t.index = samples[i].index;
      t.name = samples[i].name;
      t.cycles += samples[i].cycles;
      t.state = samples[i].state;
      t.target_state = samples[i].target_state;
    }
  }

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
            "fpga_compute_cycles,fpga_weight_transfer_cycles,esti_weight_"
            "transfer_cycles,layer_total"
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

    // The same numbers as prf.csv, plus the named hardware counters, in
    // SECDA-Core's secda_profile.json schema ($SECDA_PROFILE_PATH overrides
    // the path).
    auto us = [](duration_ns d) { return (long long)duration_cast<TSCALE>(d).count(); };
    write_secda_profile_us("secda_profile.json",
                           {{"driver_total", us(driver_total)},
                            {"fpga_wgt_send", us(fpga_wgt_send)},
                            {"fpga_inp_send", us(fpga_inp_send)},
                            {"fpga_compute", us(fpga_compute)},
                            {"fpga_inp_pack", us(fpga_inp_pack)},
                            {"fpga_wgt_pack", us(fpga_wgt_pack)},
                            {"fpga_wgt_pack_opt", us(fpga_wgt_pack_opt)},
                            {"fpga_compute_cycles", us(fpga_compute_cycles)},
                            {"fpga_weight_transfer_cycles", us(fpga_weight_transfer_cycles)},
                            {"esti_weight_transfer_cycles", us(esti_weight_transfer_cycles)},
                            {"layer_total", us(layer_total)}},
                           hwc_totals);
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
  bool alloc_allowed;

  unsigned int curr_offset = 0;
  vector<bool> layer_preloaded;
  int dma_size_list[DMA_COUNT];
  uint32_t curr_offsets[DMA_COUNT];

  vector<map<int, tuple<uint32_t, uint32_t>>> tile_offset_map_A;
  vector<map<int, tuple<uint32_t, uint32_t>>> tile_offset_map_B;
  vector<map<int, tuple<uint32_t, uint32_t>>> tile_offset_map_C;
  vector<map<int, tuple<uint32_t, uint32_t>>> tile_offset_map_D;

  layer_details() {
    alloc_allowed = LAYER_PREALLOC;
    layer_preloaded.resize(500, false);
    for (int i = 0; i < DMA_COUNT; i++) {
      dma_size_list[i] = 0;
      curr_offsets[i] = 0;
    }
    tile_offset_map_A.clear();
    tile_offset_map_B.clear();
    tile_offset_map_C.clear();
    tile_offset_map_D.clear();
  }

  void reset() {
    alloc_allowed = LAYER_PREALLOC;
    alloced_layers = 0;
    curr_offset = 0;
    // assign, not resize: resize keeps the previous plan's flags set.
    layer_preloaded.assign(500, false);
    for (int i = 0; i < DMA_COUNT; i++) {
      dma_size_list[i] = 0;
      curr_offsets[i] = 0;
    }
    tile_offset_map_A.clear();
    tile_offset_map_B.clear();
    tile_offset_map_C.clear();
    tile_offset_map_D.clear();
  }

  // Layers are the planner's MUL_MAT ordinals (set per call with setLayer);
  // -1 is a MUL_MAT the plan doesn't cover.
  bool is_preloaded(int l) const {
    return l >= 0 && (size_t)l < layer_preloaded.size() && layer_preloaded[l];
  }

  // Tile maps are indexed by layer, like layer_preloaded.
  void set_tile_maps(int l, const map<int, tuple<uint32_t, uint32_t>> &a,
                     const map<int, tuple<uint32_t, uint32_t>> &b,
                     const map<int, tuple<uint32_t, uint32_t>> &c,
                     const map<int, tuple<uint32_t, uint32_t>> &d) {
    if (tile_offset_map_A.size() <= (size_t)l) {
      tile_offset_map_A.resize(l + 1);
      tile_offset_map_B.resize(l + 1);
      tile_offset_map_C.resize(l + 1);
      tile_offset_map_D.resize(l + 1);
    }
    tile_offset_map_A[l] = a;
    tile_offset_map_B[l] = b;
    tile_offset_map_C[l] = c;
    tile_offset_map_D[l] = d;
  }

  bool alloc_layer(int layer, unsigned int K, unsigned int M,
                   unsigned int wgt_type) {
    if (layer < 0) return false;
    int kb = K / 256;
    int number_of_blocks = (M * kb);
    // Might need to change to tile based allocation and not layer based allocation
    splitfinder(dma_size_list, DMA_COUNT, number_of_blocks);
    int bytes_per_block = 0;
    if (wgt_type == 6) bytes_per_block = sizeof(acc_block_q6_K);
    if (wgt_type == 5) bytes_per_block = sizeof(acc_block_q5_K);
    if (wgt_type == 4) bytes_per_block = sizeof(acc_block_q4_K);
    if (wgt_type == 3) bytes_per_block = sizeof(acc_block_q3_K);
    if (wgt_type == 2) bytes_per_block = sizeof(acc_block_q2_K);

    for (int i = 0; i < DMA_COUNT; i++) {
      dma_size_list[i] = dma_size_list[i] * bytes_per_block;
      if (dma_size_list[i] + curr_offsets[i] > DMA_WGT_SIZE)
        alloc_allowed = false;
    }
    if (!alloc_allowed || !LAYER_PREALLOC) {
      alloc_allowed = false;
      return false;
    }

    if (layer_preloaded.size() <= layer) layer_preloaded.resize(layer + 1, false);
    layer_preloaded[layer] = true;
    alloced_layers++;
    return true;
  }
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
  int *DMA_I1;
  int *DMA_I2;
  int *DMA_I3;
  int *DMA_I4;

  int *DMA_W1;
  int *DMA_W2;
  int *DMA_W3;
  int *DMA_W4;

  // Debugging
  struct layer_details t;
  struct model_info m;
  struct acc_times *a_t;
  int n = 0; // Number of graphs executed
  int wgt_blk = 0;
  int inp_blk = 0;
};

#endif // ACC_CONTAINER