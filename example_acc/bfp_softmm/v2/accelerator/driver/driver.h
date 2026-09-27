#ifndef ACC_DRIVER
#define ACC_DRIVER

#include "acc_container.h"
#include <algorithm>
#include <cstring>

#define DLOG(X)
#define DLOG2(X)
#define STR(X) std::to_string(X)
#define HEX(X) std::hex << X << std::dec

namespace acc_driver {

void LoadWeights_Inference(acc_container *drv, int m, int mstep, int nstep,
                           int kb, char *wgt_block) {

  int *DMA_I1 = drv->mdma->dmas[0].dma_get_inbuffer();
  int *DMA_I2 = drv->mdma->dmas[1].dma_get_inbuffer();
  int *DMA_I3 = drv->mdma->dmas[2].dma_get_inbuffer();
  int *DMA_I4 = drv->mdma->dmas[3].dma_get_inbuffer();
  uint32_t wgt_blck = 0;
  acc_block_q6_K *wgt_blk_6 = (acc_block_q6_K *)wgt_block;
  acc_block_q5_K *wgt_blk_5 = (acc_block_q5_K *)wgt_block;
  acc_block_q4_K *wgt_blk_4 = (acc_block_q4_K *)wgt_block;
  acc_block_q3_K *wgt_blk_3 = (acc_block_q3_K *)wgt_block;
  acc_block_q2_K *wgt_blk_2 = (acc_block_q2_K *)wgt_block;

  if (drv->wgt_type == 6) wgt_blck = Q6_X32;
  if (drv->wgt_type == 5) wgt_blck = Q5_X32;
  if (drv->wgt_type == 4) wgt_blck = Q4_X32;
  if (drv->wgt_type == 3) wgt_blck = Q3_X32;
  if (drv->wgt_type == 2) wgt_blck = Q2_X32;

  int curr_dma = 0;
  int lenA = 0;
  int lenB = 0;
  int lenC = 0;
  int lenD = 0;
  int lenA_adr = 0;
  int lenB_adr = 0;
  int lenC_adr = 0;
  int lenD_adr = 0;
  uint32_t op = OPCODE_LOAD_WGT, ld1 = 0, ld2 = 0, ld3 = 0, ld4 = 0;
  DMA_I1[ld1++] = op;
  DMA_I1[ld1++] = mstep * kb;
  DMA_I1[ld1++] = wgt_blck;
  lenA_adr = ld1++;
  lenB_adr = ld1++;
  lenC_adr = ld1++;
  lenD_adr = ld1++;

  prf_start(6);
  uint32_t *ld_p = &ld1;
  int *buf_p = DMA_I1;
  for (int ms = m; ms < (m + mstep); ms++) {
    for (int k = 0; k < kb; k++) {
      int *wgt_blk_6_p = (int *)&wgt_blk_6[ms * kb + k];
      int *wgt_blk_5_p = (int *)&wgt_blk_5[ms * kb + k];
      int *wgt_blk_4_p = (int *)&wgt_blk_4[ms * kb + k];
      int *wgt_blk_3_p = (int *)&wgt_blk_3[ms * kb + k];
      int *wgt_blk_2_p = (int *)&wgt_blk_2[ms * kb + k];
      if (drv->wgt_type == 6) memcpy(&buf_p[*ld_p], wgt_blk_6_p, wgt_blck * 4);
      if (drv->wgt_type == 5) memcpy(&buf_p[*ld_p], wgt_blk_5_p, wgt_blck * 4);
      if (drv->wgt_type == 4) memcpy(&buf_p[*ld_p], wgt_blk_4_p, wgt_blck * 4);
      if (drv->wgt_type == 3) memcpy(&buf_p[*ld_p], wgt_blk_3_p, wgt_blck * 4);
      if (drv->wgt_type == 2) memcpy(&buf_p[*ld_p], wgt_blk_2_p, wgt_blck * 4);
      *ld_p += wgt_blck;
      if (curr_dma == 0) {
        ld_p = &ld2;
        buf_p = DMA_I2;
        curr_dma = 1;
        lenA += wgt_blck;
      } else if (curr_dma == 1) {
        ld_p = &ld3;
        buf_p = DMA_I3;
        curr_dma = 2;
        lenB += wgt_blck;
      } else if (curr_dma == 2) {
        ld_p = &ld4;
        buf_p = DMA_I4;
        curr_dma = 3;
        lenC += wgt_blck;
      } else if (curr_dma == 3) {
        ld_p = &ld1;
        buf_p = DMA_I1;
        curr_dma = 0;
        lenD += wgt_blck;
      }
    }
  }
  DMA_I1[lenA_adr] = lenA;
  DMA_I1[lenB_adr] = lenB;
  DMA_I1[lenC_adr] = lenC;
  DMA_I1[lenD_adr] = lenD;

  op = OPCODE_COMPUTE;
  DMA_I1[ld1++] = op;
  DMA_I1[ld1++] = mstep * kb;
  DMA_I1[ld1++] = nstep * kb;
  DMA_I1[ld1++] = mstep;
  DMA_I1[ld1++] = nstep;
  prf_end(6, drv->a_t->fpga_wgt_pack);

  DLOG(cout << "Weight Send Start" << endl);
  prf_start(1);
  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld1);
  drv->mdma->dmas[1].dma_change_start(0);
  drv->mdma->dmas[1].dma_start_send(ld2);
  drv->mdma->dmas[2].dma_change_start(0);
  drv->mdma->dmas[2].dma_start_send(ld3);
  drv->mdma->dmas[3].dma_change_start(0);
  drv->mdma->dmas[3].dma_start_send(ld4);
  drv->mdma->multi_dma_wait_send();
  prf_end(1, drv->a_t->fpga_wgt_send);
  DLOG(cout << "Weight Send End" << endl);
}

void LoadWeights(acc_container *drv, int m, int mstep, int nstep, int kb,
                 char *wgt_block) {
  // if (drv->t.layer_alloced[drv->t.layer]) {
  //   LoadWeights_Preloaded(drv, m, mstep, nstep, kb, wgt_block);
  // } else {
  //   LoadWeights_Inference(drv, m, mstep, nstep, kb, wgt_block);
  // }
  LoadWeights_Inference(drv, m, mstep, nstep, kb, wgt_block);
}

void StoreOutputs(acc_container *drv, int m, int mstep, int nstep, int n,
                  int out_stride) {
  int *DMA_O = drv->mdma->dmas[0].dma_get_outbuffer();
  DLOG(cout << "Compute Start" << endl);
  prf_start(2);
  drv->mdma->dmas[0].dma_start_recv(nstep * mstep);
  drv->mdma->dmas[0].dma_wait_recv();
  prf_end(2, drv->a_t->fpga_compute);
  DLOG(cout << "Compute End" << endl);
  int out_idx = 0;
  for (uint32_t ns = n; ns < (n + nstep); ns++) {
    float *dst_col = (float *)(drv->out + (ns * out_stride));
    memcpy(&dst_col[m], &DMA_O[out_idx], mstep * sizeof(float));
    out_idx += mstep;
  }
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

  // Input Packet
  DMA_I[ld++] = kb * N;
  prf_start(5);
  for (int64_t n = 0; n < N; n++) {
    for (int k = 0; k < kb; k++) { // Blocks per K dim
      memcpy(&DMA_I[ld], &inp_block[n * kb + k], INP_BLCK * 4);
      ld += INP_BLCK;
    }
  }
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
      DLOG(cout << "LoadWeights" << endl);

      LoadWeights(drv, m, mstep, nstep, kb, wgt_block);
      StoreOutputs(drv, m, mstep, nstep, n, out_stride);
    }
  }
}

void Entry(acc_container &drv) {
  // cout << "===========================" << endl;
  // cout << "BFPP || Pre-ACC Info" << endl;
  // cout << "Layer: " << drv.t.layer << endl;
  // cout << "WGT_TYPE: " << drv.wgt_type << endl;
  // cout << "M: " << drv.M << endl;
  // cout << "N: " << drv.N << endl;
  // cout << "K: " << drv.K << endl;
  // cout << "inp_stride: " << drv.inp_stride << endl;
  // cout << "wgt_stride: " << drv.wgt_stride << endl;
  // cout << "out_stride: " << drv.out_stride << endl;
  // std::cout << "===========================" << std::endl;
  prf_start(1); // Start profiling the driver
  MM(&drv);
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.a_t->driver_total); // Stop profiling the driver
}

void Entry_Prepare(acc_container &drv, int M, int N, int K, char *inp,
                   char *wgt, char *out, int wgt_type) {

  unsigned int wgt_size = 0;
  unsigned int inp_size = Q8_X8;
  if (wgt_type == 6) wgt_size = Q6_X8;
  if (wgt_type == 5) wgt_size = Q5_X8;
  if (wgt_type == 4) wgt_size = Q4_X8;
  if (wgt_type == 3) wgt_size = Q3_X8;
  if (wgt_type == 2) wgt_size = Q2_X8;

  // drv setup
  drv.t.layer = wgt_type;
  drv.inp = inp;
  drv.wgt = wgt;
  drv.out = out;
  drv.M = M;
  drv.N = N;
  drv.K = K;
  drv.inp_stride = inp_size * K / 256;
  drv.wgt_stride = wgt_size * K / 256;
  drv.out_stride = M * 4;
  drv.wgt_type = wgt_type;

  drv.hwc->set_target_state(0, 1);  // Control_Unit
  drv.hwc->set_target_state(1, 4);  // Load_Unit
  drv.hwc->set_target_state(2, 1);  // Store_Unit
  drv.hwc->set_target_state(3, 31); // Scheduler
  drv.hwc->set_target_state(4, 1);  // Weight_Transfer
  drv.hwc->set_target_state(12, 2);  // HWC_X1_Compute
  drv.hwc->reset_hwc();             // Reset HWC

  Entry(drv);
  drv.hwc->print_hwc_map(false);
  int compute_cycles = drv.hwc->get_cycle_count(12);
  int weight_transfer_cycles = drv.hwc->get_cycle_count(1);
  drv.a_t->fpga_compute_cycles = duration_ns(compute_cycles * 5);
  drv.a_t->fpga_weight_transfer_cycles =
      duration_ns(weight_transfer_cycles * 5);
}

// v2: flags bit layout, matches acc_softmax_unit.sc.h / Control_Unit's
// softmax branch.
enum SoftmaxFlagBits {
  SM_HAS_MASK = 0x1,
  SM_MASK_F16 = 0x2,
  SM_HAS_SINK = 0x4,
  SM_IS_FIRST_TILE = 0x8,
  SM_IS_LAST_TILE_PASS1 = 0x10,
  SM_IS_PASS2 = 0x20,
};

// Sends one tile/pass round: [OPCODE_SOFTMAX, tile_len, scale_bits,
// slope_bits, flags, sink_bits, logits[tile_len], mask[tile_len] if
// present]. `off` indexes into the row's full logits/mask arrays (the
// caller passes row-relative pointers already offset by `off` for
// logits/mask, this only affects the wire framing).
static void SendSoftmaxTile(acc_container *drv, unsigned int tile_len,
                            float scale, float slope, unsigned int flags,
                            float sink_value, const float *logits_tile,
                            const float *mask_tile_f32,
                            const uint16_t *mask_tile_f16) {
  int *DMA_I = drv->mdma->dmas[0].dma_get_inbuffer();
  int ld = 0;
  DMA_I[ld++] = OPCODE_SOFTMAX;
  DMA_I[ld++] = (int)tile_len;
  memcpy(&DMA_I[ld++], &scale, sizeof(float));
  memcpy(&DMA_I[ld++], &slope, sizeof(float));
  DMA_I[ld++] = (int)flags;
  memcpy(&DMA_I[ld++], &sink_value, sizeof(float));

  memcpy(&DMA_I[ld], logits_tile, tile_len * sizeof(float));
  ld += tile_len;
  if (flags & SM_HAS_MASK) {
    for (unsigned int i = 0; i < tile_len; i++)
      DMA_I[ld + i] = (flags & SM_MASK_F16) ? (int)mask_tile_f16[i]
                                            : ((const int *)mask_tile_f32)[i];
    ld += tile_len;
  }

  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[0].dma_wait_send();
}

// v2: tiled/streaming softmax - no row-width cap (v1's single-packet design
// capped rows at SOFTMAX_MAX_N since the whole row had to fit on-chip). Two
// passes over SOFTMAX_TILE_N-sized tiles: pass 1 streams tiles to build the
// hardware's running max/sum (online-softmax rescale, no output), pass 2
// re-streams the same tiles now that max/sum are finalized and reads back
// the normalized output. `slope` is the host-precomputed ALiBi slope for
// this row's head (1.0f if ALiBi is inactive) - see bfp_softmax.h for the
// matching CPU reference and the slope formula. Exactly one of
// mask_row_f32/mask_row_f16 may be non-null (or both null for no mask).
// Safe for in-place callers: pass 1 never writes host memory, and pass 2
// reads-then-writes each tile before advancing.
void Softmax(acc_container *drv, const float *in_row,
             const float *mask_row_f32, const uint16_t *mask_row_f16,
             bool has_sink, float sink_value, float *out_row,
             unsigned int n, float scale, float slope) {
  bool has_mask = mask_row_f32 != nullptr || mask_row_f16 != nullptr;
  bool mask_f16 = mask_row_f16 != nullptr;
  const unsigned int base_flags = (has_mask ? SM_HAS_MASK : 0u) |
                                  (mask_f16 ? SM_MASK_F16 : 0u) |
                                  (has_sink ? SM_HAS_SINK : 0u);

  // Pass 1: stats (online-softmax max/sum accumulation, no output).
  for (unsigned int off = 0; off < n; off += SOFTMAX_TILE_N) {
    unsigned int tile_len = std::min((unsigned int)SOFTMAX_TILE_N, n - off);
    unsigned int flags = base_flags;
    if (off == 0) flags |= SM_IS_FIRST_TILE;
    if (off + tile_len >= n) flags |= SM_IS_LAST_TILE_PASS1;
    SendSoftmaxTile(drv, tile_len, scale, slope, flags, sink_value,
                    in_row + off, mask_row_f32 ? mask_row_f32 + off : nullptr,
                    mask_row_f16 ? mask_row_f16 + off : nullptr);
  }

  // Pass 2: output (running max/recip already finalized from pass 1).
  for (unsigned int off = 0; off < n; off += SOFTMAX_TILE_N) {
    unsigned int tile_len = std::min((unsigned int)SOFTMAX_TILE_N, n - off);
    unsigned int flags = base_flags | SM_IS_PASS2;
    SendSoftmaxTile(drv, tile_len, scale, slope, flags, sink_value,
                    in_row + off, mask_row_f32 ? mask_row_f32 + off : nullptr,
                    mask_row_f16 ? mask_row_f16 + off : nullptr);

    drv->mdma->dmas[0].dma_start_recv(tile_len);
    drv->mdma->dmas[0].dma_wait_recv();
    int *DMA_O = drv->mdma->dmas[0].dma_get_outbuffer();
    memcpy(out_row + off, DMA_O, tile_len * sizeof(float));
  }
}

void Entry_Prepare_Softmax(acc_container &drv, const float *in_row,
                           const float *mask_row_f32,
                           const uint16_t *mask_row_f16, bool has_sink,
                           float sink_value, float *out_row, unsigned int n,
                           float scale, float slope) {
  drv.hwc->set_target_state(13, 1); // Softmax_Unit
  drv.hwc->reset_hwc();
  Softmax(&drv, in_row, mask_row_f32, mask_row_f16, has_sink, sink_value,
          out_row, n, scale, slope);
  drv.hwc->print_hwc_map(false);
}

} // namespace acc_driver

#endif // ACC_DRIVER