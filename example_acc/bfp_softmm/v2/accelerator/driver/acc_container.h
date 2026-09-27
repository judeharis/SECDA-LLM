#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <cassert>
#include <iomanip>
#include <vector>

#ifdef SYSC
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "secda-core/axi_support/v5/axi_api_v5.h"
#include "secda-core/secda_profiler/profiler.h"
#include "secda-core/secda_utils/multi_threading.h"
#include "secda-core/secda_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
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
  duration_ns fpga_compute_cycles;
  duration_ns fpga_weight_transfer_cycles;
  duration_ns cpu_q2_time;
  duration_ns cpu_q3_time;
  duration_ns cpu_q4_time;
  duration_ns cpu_q5_time;
  duration_ns cpu_q6_time;

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
    prf_out(TSCALE, fpga_compute_cycles);
    prf_out(TSCALE, fpga_weight_transfer_cycles);
    prf_out(TSCALE, cpu_q2_time);
    prf_out(TSCALE, cpu_q3_time);
    prf_out(TSCALE, cpu_q4_time);
    prf_out(TSCALE, cpu_q5_time);
    prf_out(TSCALE, cpu_q6_time);
    cerr << "================================================" << endl;
#endif
  }
  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    time_t now = time(0);
    char *dt = ctime(&now);
    dt[strlen(dt) - 1] = '\0';
    file << "driver_total,fpga_total,fpga_wgt_send,fpga_inp_send,fpga_"
            "compute,fpga_inp_pack,fpga_wgt_pack,fpga_compute_cycles,fpga_"
            "weight_transfer_cycles,"
            "cpu_q2_time,cpu_q3_time,cpu_q4_time,cpu_q5_time,cpu_q6_time"
         << endl;
    // file << dt << ",";
    prf_file_out_x(TSCALE, driver_total, file);
    prf_file_out_x(TSCALE, fpga_total, file);
    prf_file_out_x(TSCALE, fpga_wgt_send, file);
    prf_file_out_x(TSCALE, fpga_inp_send, file);
    prf_file_out_x(TSCALE, fpga_compute, file);
    prf_file_out_x(TSCALE, fpga_inp_pack, file);
    prf_file_out_x(TSCALE, fpga_wgt_pack, file);
    prf_file_out_x(TSCALE, fpga_compute_cycles, file);
    prf_file_out_x(TSCALE, fpga_weight_transfer_cycles, file);
    prf_file_out_x(TSCALE, cpu_q2_time, file);
    prf_file_out_x(TSCALE, cpu_q3_time, file);
    prf_file_out_x(TSCALE, cpu_q4_time, file);
    prf_file_out_x(TSCALE, cpu_q5_time, file);
    prf_file_out_l(TSCALE, cpu_q6_time, file);
    file << endl;
    file.close();
#endif
  }

  ~acc_times() {
#ifdef ACC_PROFILE
    cerr << "================================================" << endl;
    cerr << endl;
    prf_out(TSCALE, driver_total);
    prf_out(TSCALE, fpga_total);
    prf_out(TSCALE, fpga_wgt_send);
    prf_out(TSCALE, fpga_inp_send);
    prf_out(TSCALE, fpga_compute);
    prf_out(TSCALE, fpga_inp_pack);
    prf_out(TSCALE, fpga_wgt_pack);
    prf_out(TSCALE, fpga_compute_cycles);
    prf_out(TSCALE, fpga_weight_transfer_cycles);
    prf_out(TSCALE, cpu_q2_time);
    prf_out(TSCALE, cpu_q3_time);
    prf_out(TSCALE, cpu_q4_time);
    prf_out(TSCALE, cpu_q5_time);
    prf_out(TSCALE, cpu_q6_time);
    cerr << endl;
    cerr << "================================================" << endl;
    save_prf();
#endif
  }
};

struct offload_details {
  int layer = 0;
  int count = 0;
  bool profile = false;
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

  // Debugging
  struct offload_details t;
  struct acc_times *a_t;
  int n = 0;
  int wgt_blk = 0;
  int inp_blk = 0;
};

#endif // ACC_CONTAINER