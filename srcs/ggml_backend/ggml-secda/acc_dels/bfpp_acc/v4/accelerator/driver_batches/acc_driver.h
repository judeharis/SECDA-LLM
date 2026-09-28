#ifndef ACC_DRIVER_H
#define ACC_DRIVER_H

#include "acc_container.h"
#include "acc_driver_mt.h"
// Jude: Added - acc_driver_softmax.h (EntrySoftmax) is intentionally NOT
// included here: it calls advance_layer(), defined further down in this
// file's bfpp_acc namespace block, so it must be included afterwards (see
// acc_driver_connector.cc) rather than nested inside this still-open
// namespace block.

// #define DLOG(X) X
#define DLOG(X)

namespace bfpp_acc {

unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2, dma_addr3};
struct acc_times *a_t;
static class Profile profile;

struct del_params dparams;
struct acc_container *drv;

#ifdef SYSC
ACCNAME *acc;
struct sysC_sigs *scs;
struct s_mdma *mdma;
#else
int *acc;
struct s_mdma *mdma;
#endif

struct a_ctrl *ctrl;
struct h_ctrl *hwc;

#ifdef SYSC
#define THREADS 1
#else
#define THREADS 1
#endif

// Jude: Added - shared "which supported node am I on" bookkeeping, used by
// both EntryMM's tail (MUL_MAT) and EntrySoftmax's tail (SOFT_MAX, see
// acc_driver_softmax.h). Every node ggml_secda_graph_plan_create counts as
// supported must reach a call that runs this exactly once, regardless of op
// type, or dparams.layer desyncs from drv->m.supported_nodes - the same
// class of bug the ggml_nelements(dst)==0 fix addressed for degenerate
// MUL_MAT nodes.
static void advance_layer() {
  if (dparams.layer + 1 == drv->m.supported_nodes) {
    drv->n++;
    dparams.layer = 0;
    return;
  }
  dparams.layer++;
}

void MM(acc_container *drv) {
  int N = drv->N;
  int M = drv->M;
  int K = drv->K;
  int wgt_type = drv->wgt_type;
  int inp_stride = drv->inp_stride;
  int wgt_stride = drv->wgt_stride;
  int out_stride = drv->out_stride;

  int64_t nrc = 1;
  acc_block_q8_K *inp_block = (acc_block_q8_K *)drv->inp;
  char *wgt_block = (char *)drv->wgt;

  // Check if the input can be stored completely in the accelerator
  // Check if the weight can be stored completely in the accelerator
  // Check if the output can be stored completely in the accelerator

  // Depending on the size of the input, weight and output, we can decide how
  // many iterations we need to run the accelerator and which dataflow to use

  // kb is the number of blocks in the K dimension
  int kb = K / QK_K;

  // tile_kmb is the number of blocks in the K dimension that can be stored in
  // the weight buffer
  uint32_t tile_kmb = roundDown(min((SUP_KMB), M * kb), kb);

  // tile_knb is the number of blocks in the K dimension that can be stored in
  // the input buffer
  uint32_t tile_knb = roundDown(min((SUP_KNB), N * kb), kb);
  uint32_t tile_m = tile_kmb / kb;
  uint32_t tile_n = tile_knb / kb;

  if ((N * kb) > SUP_KNB)
    cerr << "Input data exceeds SBVP input buffer" << endl;
  if (tile_m <= 1)
    cerr << "Weight block x depth exceeds SBVP weight buffer" << endl;

  assert((N * kb) <= SUP_KNB && "Input data exceeds SBVP input buffer");
  assert(tile_m >= 1 && "Weight block x depth exceeds SBVP weight buffer");

  int *DMA_I = drv->mdma->dmas[0].dma_get_inbuffer();
  int *DMA_O = drv->mdma->dmas[0].dma_get_outbuffer();

  int ld = 0;
  uint32_t op = OPCODE_LOAD_INP | OPCODE_CONFIG;
  DMA_I[ld++] = op;
  DMA_I[ld++] = kb;
  DMA_I[ld++] = M;
  DMA_I[ld++] = N;
  DMA_I[ld++] = wgt_type;

  // cout << "OP: " << (int)op << endl;
  // cout << "kb: " << kb << endl;
  // cout << "M: " << M << endl;
  // cout << "N: " << N << endl;
  // cout << "WGT_TYPE: " << drv->wgt_type << endl;

  // Input Packet
  DMA_I[ld++] = kb * N;
  prf_start(5);
  for (int64_t n = 0; n < N; n++) {
    for (int k = 0; k < kb; k++) { // Blocks per K dim
      memcpy(&DMA_I[ld], &inp_block[n * kb + k], INP_BLCK * 4);
      ld += INP_BLCK;
    }
  }

#ifdef PRINT_FILE
#ifndef GGML_SECDA_ARM
  ofstream myfile;
  myfile.open("_aData/cgpt/inp/inp_vm_" + std::to_string(drv->inp_blk++) + "_" +
              std::to_string(drv->t.layer) + "_acc_norm.csv");
  int *res_pointer = (int *)DMA_I;
  for (int r = 0; r < ld; r++) {
    myfile << (int)res_pointer[r] << endl;
  }
  myfile.close();
#endif
#endif

  // #ifdef GGML_SECDA_ARM
  //   // load input
  //   std::ifstream inp_file;
  //   inp_file.open("_aData/cgpt/inp/inp_vm_" + std::to_string(drv->inp_blk) +
  //                 "_" + std::to_string(drv->t.layer) + "_acc_norm.csv");
  //   for (int r = 0; r < ld; r++) {
  //     inp_file >> DMA_I[r];
  //   }
  //   inp_file.close();
  //   cout << "Input Loaded" << endl;
  //   ofstream myfile;
  //   myfile.open("_aData/cgpt/inp/inp_vm_" + std::to_string(drv->inp_blk++) +
  //   "_" +
  //               std::to_string(drv->t.layer) + "_acc_afte.csv");
  //   int *res_pointer = (int *)DMA_I;
  //   for (int r = 0; r < ld; r++) {
  //     myfile << (int)res_pointer[r] << endl;
  //   }
  //   myfile.close();
  // #endif

  prf_end(5, drv->a_t->fpga_inp_pack);
  DLOG(cout << "Input Send Start" << endl);
  prf_start(0);
  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[0].dma_wait_send();
  prf_end(0, drv->a_t->fpga_inp_send);
  DLOG(cout << "Input Send End" << endl);

  for (uint32_t n = 0; n < N; n += tile_n) {
    uint32_t nstep = std::min(tile_n, tile_n - n);
    for (uint32_t m = 0; m < M; m += tile_m) {
      uint32_t mstep = std::min(tile_m, M - m);
      LoadWeights(drv, m, mstep, nstep, kb, wgt_block);
      StoreOutputs(drv, m, mstep, nstep, n, out_stride);
    }
  }

#ifdef DELEGATE_VERBOSE
  int compute_cycles = drv->hwc->get_cycle_count(3);
  int weight_transfer_cycles = drv->hwc->get_cycle_count(1);
  unsigned long long compute_time_ns = (compute_cycles * 5);
  unsigned long long weight_transfer_time_ns = (weight_transfer_cycles * 5);
  float send_speed = drv->mdma->dmas[0].get_send_bandwidth();
  float recv_speed = drv->mdma->dmas[0].get_recv_bandwidth();
  bool preloaded = drv->t.layer_preloaded[drv->t.layer];
  drv->mdma->dmas[0].profile_reset();
  std::ofstream file("layers.csv", std::ios::app);
  file << dparams.layer << ", " << M << ", " << N << ", " << K << ", "
       << inp_stride << ", " << wgt_stride << ", " << out_stride << ", "
       << wgt_type << ", " << (preloaded ? "preloaded" : "not_preloaded")
       << ", " << tile_m << ", " << tile_n << ", " << tile_kmb << ", "
       << tile_knb << ", " << (compute_time_ns) << ", "
       << (weight_transfer_time_ns) << ", " << send_speed << ", " << recv_speed
       << endl;
  file.close();
#endif
}

static bool DimCheck(int M, int N, int K) {

  // kb is the number of blocks in the K dimension
  int kb = K / QK_K;

  // tile_kmb is the number of blocks in the K dimension that can be stored in
  // the weight buffer
  uint32_t tile_kmb = roundDown(min((SUP_KMB), M * kb), kb);

  // tile_knb is the number of blocks in the K dimension that can be stored in
  // the input buffer
  uint32_t tile_knb = roundDown(min((SUP_KNB), N * kb), kb);
  uint32_t tile_m = tile_kmb / kb;
  uint32_t tile_n = tile_knb / kb;

  if ((N * kb) > SUP_KNB) {
    // cerr << "Input data exceeds SBVP input buffer" << endl;
    return false;
  }
  if (tile_m <= 1) {
    // cerr << "Weight block x depth exceeds SBVP weight buffer" << endl;
    return false;
  }
  // assert((N * kb) <= SUP_KNB && "Input data exceeds SBVP input buffer");
  // assert(tile_m >= 1 && "Weight block x depth exceeds SBVP weight buffer");

  return true;
}

// m = wgt_rows, n = inp_cols, k = depth
//
// ne2/ne3 are the destination's batch/head extents beyond the M x N slab
// (ggml_tensor::ne[2]/ne[3]); inp_stride2/3 and out_stride2/3 are the
// matching byte strides (ggml_tensor::nb[2]/nb[3]). The weight (wgt) is
// broadcast across every (b2, b3) slice, mirroring ggml_mul_mat's own
// broadcast of src0 across src1's ne2/ne3 - only the input/output pointers
// move between slices. A zero extent anywhere (N == 0, ne2 == 0, ne3 == 0)
// is a legitimate no-op: the loops below simply don't execute, matching
// what a plain C loop over zero elements would do.
static void EntryMM(const void *wgt, const void *inp, void *out, int M, int N,
                    int K, int inp_stride, int wgt_stride, int out_stride,
                    int wgt_type, int ne2, int ne3, int inp_stride2,
                    int inp_stride3, int out_stride2, int out_stride3) {

  // Accelerator Specific Parameters
  drv->t.layer = dparams.layer;
  drv->M = M;
  drv->N = N;
  drv->K = K;
  drv->wgt = (char *)wgt;
  drv->inp_stride = inp_stride;
  drv->wgt_stride = wgt_stride;
  drv->out_stride = out_stride;
  drv->wgt_type = wgt_type;
  drv->a_t = a_t;
  bool preloaded = drv->t.layer_preloaded[drv->t.layer];

  cout << endl << "===========================" << endl;
  cout << "BFPP_ACC || Pre-ACC Info" << endl;
  cout << "Layer: " << dparams.layer << (preloaded ? " (Preloaded)" : "")
       << endl;
  cout << "M: " << M << endl;
  cout << "N: " << N << endl;
  cout << "K: " << K << endl;
  cout << "inp_stride: " << inp_stride << endl;
  cout << "wgt_stride: " << wgt_stride << endl;
  cout << "out_stride: " << out_stride << endl;
  cout << "wgt_type: " << wgt_type << endl;
  cout << "===========================" << endl;

  drv->hwc->reset_hwc(); // Reset HWC
  prf_start(1);          // Start profiling the driver

  for (int b3 = 0; b3 < ne3; b3++) {
    for (int b2 = 0; b2 < ne2; b2++) {
      drv->inp = (char *)inp + (size_t)b3 * inp_stride3 +
                 (size_t)b2 * inp_stride2;
      drv->out = (char *)out + (size_t)b3 * out_stride3 +
                 (size_t)b2 * out_stride2;
      if (N > 0) bfpp_acc::MM(drv);
    }
  }
  SYSC_ON(drv->profile->saveProfile(drv->acc->profiling_vars));
  prf_end(1, a_t->driver_total); // Stop profiling the driver

  int compute_cycles = drv->hwc->get_cycle_count(3);
  int weight_transfer_cycles = drv->hwc->get_cycle_count(1);
  drv->a_t->fpga_compute_cycles += duration_ns(compute_cycles * 5);
  drv->a_t->fpga_weight_transfer_cycles +=
      duration_ns(weight_transfer_cycles * 5);
#ifdef ACC_PROFILE
  drv->a_t->add_hwc(drv->hwc->hwc_samples());
#endif

  #ifdef PRINT_FILE
    char buf[256];
    sprintf(buf, "_aData/cgpt2/out_%d_acc.csv", dparams.layer);
    FILE *f = fopen(buf, "w");
    float *res_pointer = (float *)out;
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        fprintf(f, "%f,", (float)res_pointer[r * N + c]);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  #endif

  // Jude: Added - shared with EntrySoftmax's tail, see advance_layer().
  advance_layer();
}

} // namespace bfpp_acc

#endif // ACC_DRIVER_H
