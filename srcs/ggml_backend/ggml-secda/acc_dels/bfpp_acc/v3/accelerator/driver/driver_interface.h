#ifndef DRIVER_INTERFACE_H
#define DRIVER_INTERFACE_H

#include "acc_driver.h"

namespace bfpp_acc {

void updatePlan(int supported_nodes) {
  drv->m.supported_nodes = supported_nodes;
  drv->m.planned = true;
}

void updateProfile(std::chrono::nanoseconds time) {
  drv->a_t->layer_total += time;
}

bool modelPlanned() { return drv->m.planned; }

static bool preloadWeights(unsigned wgt_size, int layer, int M, int K,
                           const void *wgt, int wgt_type) {
  char *wgt_char = (char *)wgt;
  drv->m.wgt_ptrs.push_back(wgt_char);
  drv->m.wgt_sizes.push_back(wgt_size);
  drv->m.dim_M.push_back(M);
  drv->m.dim_K.push_back(K);
  bool alloced = drv->t.alloc_layer(layer, K, M, wgt_type);
  if (alloced) {
    char *DMA_W1 = reinterpret_cast<char *>(drv->DMA_W1);
    char *DMA_W2 = reinterpret_cast<char *>(drv->DMA_W2);
    char *DMA_W3 = reinterpret_cast<char *>(drv->DMA_W3);
    char *DMA_W4 = reinterpret_cast<char *>(drv->DMA_W4);
    acc_block_q6_K *wgt_blk_6 = (acc_block_q6_K *)wgt;
    acc_block_q5_K *wgt_blk_5 = (acc_block_q5_K *)wgt;
    acc_block_q4_K *wgt_blk_4 = (acc_block_q4_K *)wgt;
    acc_block_q3_K *wgt_blk_3 = (acc_block_q3_K *)wgt;
    acc_block_q2_K *wgt_blk_2 = (acc_block_q2_K *)wgt;
    uint32_t wgt_blck = 0;

    if (wgt_type == 6) wgt_blck = Q6_X32;
    if (wgt_type == 5) wgt_blck = Q5_X32;
    if (wgt_type == 4) wgt_blck = Q4_X32;
    if (wgt_type == 3) wgt_blck = Q3_X32;
    if (wgt_type == 2) wgt_blck = Q2_X32;

    int kb = K / 256;
    uint32_t *ld_p = &drv->t.curr_offsets[0];
    char *buf_p = DMA_W1;
    int curr_dma = 0;
    uint32_t tile_kmb = roundDown(min((SUP_KMB), M * kb), kb);
    uint32_t tile_m = tile_kmb / kb;
    map<int, tuple<uint32_t, uint32_t>> tile_map_A;
    map<int, tuple<uint32_t, uint32_t>> tile_map_B;
    map<int, tuple<uint32_t, uint32_t>> tile_map_C;
    map<int, tuple<uint32_t, uint32_t>> tile_map_D;

    for (uint32_t i = 0; i < M; i += tile_m) {
      uint32_t mstep = std::min(tile_m, M - i);
      int curr_offset_A = drv->t.curr_offsets[0];
      int curr_offset_B = drv->t.curr_offsets[1];
      int curr_offset_C = drv->t.curr_offsets[2];
      int curr_offset_D = drv->t.curr_offsets[3];
      int bytes_per_tile_A = 0;
      int bytes_per_tile_B = 0;
      int bytes_per_tile_C = 0;
      int bytes_per_tile_D = 0;
      curr_dma = 0;
      ld_p = &drv->t.curr_offsets[0];
      buf_p = DMA_W1;
      for (int j = 0; j < mstep; j++) {
        int m = i + j;
        for (int k = 0; k < kb; k++) {
          int *wgt_blk_6_p = (int *)&wgt_blk_6[m * kb + k];
          int *wgt_blk_5_p = (int *)&wgt_blk_5[m * kb + k];
          int *wgt_blk_4_p = (int *)&wgt_blk_4[m * kb + k];
          int *wgt_blk_3_p = (int *)&wgt_blk_3[m * kb + k];
          int *wgt_blk_2_p = (int *)&wgt_blk_2[m * kb + k];
          if (wgt_type == 6) memcpy(&buf_p[*ld_p], wgt_blk_6_p, wgt_blck * 4);
          if (wgt_type == 5) memcpy(&buf_p[*ld_p], wgt_blk_5_p, wgt_blck * 4);
          if (wgt_type == 4) memcpy(&buf_p[*ld_p], wgt_blk_4_p, wgt_blck * 4);
          if (wgt_type == 3) memcpy(&buf_p[*ld_p], wgt_blk_3_p, wgt_blck * 4);
          if (wgt_type == 2) memcpy(&buf_p[*ld_p], wgt_blk_2_p, wgt_blck * 4);
          *ld_p += wgt_blck * 4;
          if (curr_dma == 0) {
            bytes_per_tile_A += wgt_blck;
            ld_p = &drv->t.curr_offsets[1];
            buf_p = DMA_W2;
            curr_dma = 1;
          } else if (curr_dma == 1) {
            bytes_per_tile_B += wgt_blck;
            ld_p = &drv->t.curr_offsets[2];
            buf_p = DMA_W3;
            curr_dma = 2;
          } else if (curr_dma == 2) {
            bytes_per_tile_C += wgt_blck;
            ld_p = &drv->t.curr_offsets[3];
            buf_p = DMA_W4;
            curr_dma = 3;
          } else if (curr_dma == 3) {
            bytes_per_tile_D += wgt_blck;
            ld_p = &drv->t.curr_offsets[0];
            buf_p = DMA_W1;
            curr_dma = 0;
          }
        }
      }
      tile_map_A[i] = std::make_tuple(curr_offset_A, bytes_per_tile_A);
      tile_map_B[i] = std::make_tuple(curr_offset_B, bytes_per_tile_B);
      tile_map_C[i] = std::make_tuple(curr_offset_C, bytes_per_tile_C);
      tile_map_D[i] = std::make_tuple(curr_offset_D, bytes_per_tile_D);
    }
    drv->t.tile_offset_map_A.push_back(tile_map_A);
    drv->t.tile_offset_map_B.push_back(tile_map_B);
    drv->t.tile_offset_map_C.push_back(tile_map_C);
    drv->t.tile_offset_map_D.push_back(tile_map_D);
  }
  drv->mdma->dmas[0].dma_sync_mem();
  drv->mdma->dmas[1].dma_sync_mem();
  drv->mdma->dmas[2].dma_sync_mem();
  drv->mdma->dmas[3].dma_sync_mem();
  return alloced;
}

static void initACC() {
  // Init SystemC Modules & Profilier
  if (!dparams.init) {
    std::cout << "===========================" << std::endl;
    static struct acc_container _drv;
    static class acc_times _a_t;
#ifdef SYSC
    static ACCNAME _acc("FBFP_ACC");
    static struct sysC_sigs scs1(1);
    static struct a_ctrl ctrl1;
    static struct h_ctrl hwc1;
    static struct s_mdma mdma1(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                               DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
    sysC_init();
    hwc1.init_hwc(HWC_Monitor_Count);
    ctrl1.init_sigs(CTRL_Reg_Count);
    sysC_binder(&_acc, &scs1, &ctrl1, &hwc1, &mdma1);
    acc = &_acc;
    scs = &scs1;
    ctrl = &ctrl1;
    hwc = &hwc1;
    mdma = &mdma1;
    std::cout << "Initialised the SystemC Modules" << std::endl;
#else
    dparams.acc = getAccBaseAddress<int>(acc_ctrl_address, 65536);
    int *acc_ctrl_base = getAccBaseAddress<int>(acc_ctrl_address, 65536);
    int *acc_hwc_base = getAccBaseAddress<int>(acc_hwc_address, 65536);
    static struct a_ctrl ctrl1(acc_ctrl_base);
    static struct h_ctrl hwc1(acc_hwc_base);
    static struct s_mdma mdma1(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                               DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
    acc = dparams.acc;
    ctrl1.init_sigs(CTRL_Reg_Count);
    hwc1.init_hwc(HWC_Monitor_Count);
    ctrl = &ctrl1;
    hwc = &hwc1;
    mdma = &mdma1;
#ifdef KRIA
    std::cout << "Initialised the Kria DMA" << std::endl;
#else
    std::cout << "Initialised the Z1 DMA" << std::endl;
#endif
#endif
    std::cout << "BFPP Accelerator Driver Version 3";
#ifdef ACC_NEON
    std::cout << " with Neon";
#endif
    std::cout << std::endl;

#if defined(GGML_SECDA_QK2)
    std::cout << "Support Q2" << std::endl;
#endif
#if defined(GGML_SECDA_QK3)
    std::cout << "Support Q3" << std::endl;
#endif
#if defined(GGML_SECDA_QK4)
    std::cout << "Support Q4" << std::endl;
#endif
#if defined(GGML_SECDA_QK5)
    std::cout << "Support Q5" << std::endl;
#endif
#if defined(GGML_SECDA_QK6)
    std::cout << "Support Q6" << std::endl;
#endif

    std::cout << "===========================" << std::endl;
    drv = &_drv;
    a_t = &_a_t;
    drv->acc = acc;
    drv->hwc = hwc;
    drv->ctrl = ctrl;
    drv->mdma = mdma;
    drv->profile = &profile;
    drv->mt_context = dparams.mt_context;
    drv->DMA_I1 = drv->mdma->dmas[0].dma_get_inbuffer();
    drv->DMA_I2 = drv->mdma->dmas[1].dma_get_inbuffer();
    drv->DMA_I3 = drv->mdma->dmas[2].dma_get_inbuffer();
    drv->DMA_I4 = drv->mdma->dmas[3].dma_get_inbuffer();
    drv->DMA_W1 = drv->DMA_I1 + (DMA_INP_SIZE / 4);
    drv->DMA_W2 = drv->DMA_I2 + (DMA_INP_SIZE / 4);
    drv->DMA_W3 = drv->DMA_I3 + (DMA_INP_SIZE / 4);
    drv->DMA_W4 = drv->DMA_I4 + (DMA_INP_SIZE / 4);

    drv->hwc->set_target_state(0, 1);  // Control_Unit
    drv->hwc->set_target_state(1, 4);  // Load_Unit
    drv->hwc->set_target_state(2, 1);  // Store_Unit
    drv->hwc->set_target_state(3, 31); // Scheduler
    drv->hwc->set_target_state(4, 1);  // Weight_Transfer
    drv->hwc->reset_hwc();             // Reset HWC

#ifdef DELEGATE_VERBOSE
    std::ofstream layer_file;
    layer_file.open("layers.csv", std::ofstream::out | std::ofstream::trunc);
    layer_file.close();
    layer_file.open("layers.csv", std::ofstream::out | std::ofstream::app);
    layer_file << "layer,M,N,K,inp_stride,wgt_stride,out_stride,wgt_type,"
                  "preloaded,tile_m,tile_n,tile_kmb,tile_knb,compute_time_ns,"
                  "weight_transfer_time_ns,send_speed_MBps,recv_speed_MBps"
               << std::endl;
    layer_file.close();
#endif
    dparams.layer = 0;
    dparams.init = true;
  }
}

} // namespace bfpp_acc

#endif // DRIVER_INTERFACE_H
