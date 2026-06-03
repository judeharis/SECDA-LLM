#ifndef DRIVER_MT
#define DRIVER_MT

#include "acc_container.h"
#include <cstring>
#include <mutex>

// #define TOG(X)

// #define TOG2(X) threadsafe_cout(X)
#define TOG2(X)

#define STR(X) std::to_string(X)
#define HEX(X) std::hex << X << std::dec

namespace bfpp_acc {
using namespace std;

void threadsafe_cout(std::string log_msg) {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  cout << std::move(log_msg);
}

void LoadWeights_Inference(acc_container *drv, int m, int mstep, int nstep,
                           int kb, char *wgt_block) {
  int *DMA_I1 = drv->mdma->dmas[0].dma_get_inbuffer();
  int *DMA_I2 = drv->mdma->dmas[1].dma_get_inbuffer();
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
  int lenA_adr = 0;
  int lenB_adr = 0;
  uint32_t op = OPCODE_LOAD_WGT, ld = 0, ld2 = 0;
  DMA_I1[ld++] = op;
  DMA_I1[ld++] = mstep * kb;
  DMA_I1[ld++] = wgt_blck;
  lenA_adr = ld++;
  lenB_adr = ld++;

  drv->a_t->esti_weight_transfer_cycles +=
      duration_ns(mstep * kb * wgt_blck * 5);
  prf_start(6);
  uint32_t *ld_p = &ld;
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
        ld_p = &ld;
        buf_p = DMA_I1;
        curr_dma = 0;
        lenB += wgt_blck;
      }
    }
  }
  DMA_I1[lenA_adr] = lenA;
  DMA_I1[lenB_adr] = lenB;

  op = OPCODE_COMPUTE;
  DMA_I1[ld++] = op;
  DMA_I1[ld++] = mstep * kb;
  DMA_I1[ld++] = nstep * kb;
  DMA_I1[ld++] = mstep;
  DMA_I1[ld++] = nstep;
  prf_end(6, drv->a_t->fpga_wgt_pack);

  TOG2("Weight Send Start\n");
  prf_start(1);
  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[1].dma_change_start(0);
  drv->mdma->dmas[1].dma_start_send(ld2);
  drv->mdma->multi_dma_wait_send();
  prf_end(1, drv->a_t->fpga_wgt_send);
  TOG2("Weight Send End\n");
}

void LoadWeights_Preloaded(acc_container *drv, int m, int mstep, int nstep,
                           int kb, char *wgt_block) {
  TOG2(cout << "Preparing Weight OPCODE" << endl);
  // Send Weight Load OPCODE
  int *DMA_I = drv->mdma->dmas[0].dma_get_inbuffer();
  int wgt_type = drv->wgt_type;
  int wgt_blck = 0;
  if (wgt_type == 6) wgt_blck = Q6_X32;
  if (wgt_type == 5) wgt_blck = Q5_X32;
  if (wgt_type == 4) wgt_blck = Q4_X32;
  if (wgt_type == 3) wgt_blck = Q3_X32;
  if (wgt_type == 2) wgt_blck = Q2_X32;

  uint32_t tile_offset_A =
      std::get<0>(drv->t.tile_offset_map_A[drv->t.layer][m]) + DMA_INP_SIZE;
  uint32_t tile_offset_B =
      std::get<0>(drv->t.tile_offset_map_B[drv->t.layer][m]) + DMA_INP_SIZE;
  uint32_t packets_per_tile_A =
      std::get<1>(drv->t.tile_offset_map_A[drv->t.layer][m]);
  uint32_t packets_per_tile_B =
      std::get<1>(drv->t.tile_offset_map_B[drv->t.layer][m]);

  uint32_t op = OPCODE_LOAD_WGT, ld = 0;
  DMA_I[ld++] = op;
  DMA_I[ld++] = mstep * kb;
  DMA_I[ld++] = wgt_blck;
  DMA_I[ld++] = packets_per_tile_A;
  DMA_I[ld++] = packets_per_tile_B;

  TOG2(cout << "Weight OPCODE Prepared" << endl);
  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[0].dma_wait_send();

  TOG2(cout << "Weight OPCODE Sent" << endl);

  drv->a_t->esti_weight_transfer_cycles +=
      duration_ns(mstep * kb * wgt_blck * 5);

  // Send Weight Data
  prf_start(6);

  drv->mdma->dmas[0].dma_change_start(tile_offset_A);
  drv->mdma->dmas[1].dma_change_start(tile_offset_B);

  prf_end(6, drv->a_t->fpga_wgt_pack_opt);

  prf_start(1);
  // drv->mdma->dmas[0].dma_start_send(wgt_blck * kb * mstep);
  TOG2(cout << "Weight Send Start" << endl);
  drv->mdma->dmas[0].dma_start_send(packets_per_tile_A);
  drv->mdma->dmas[1].dma_start_send(packets_per_tile_B);
  TOG2(cout << "Weight Send Wait" << endl);


  drv->mdma->multi_dma_wait_send();
  prf_end(1, drv->a_t->fpga_wgt_send);

#ifdef PRINT_FILE
  ofstream myfile;
  myfile.open("_aData/cgpt2/wgt/wgt_" + std::to_string(drv->wgt_blk++) + "_" +
              std::to_string(drv->t.layer) + "_acc_preloaded.csv");
  // int *res_pointer =
  //     (int *)drv->t.get_buffer_from_layer(drv->t.layer, drv->DMA_W);
  // myfile << (int)op << endl;
  // myfile << (int)mstep * kb << endl;
  // myfile << (int)wgt_blck << endl;
  // for (int r = 0; r < (wgt_blck * kb * mstep); r++) {
  //   myfile << (int)res_pointer[r + ((wgt_blck * 4) * m * kb / 4)] << endl;
  // }
  myfile << (int)OPCODE_COMPUTE << endl;
  myfile << (int)mstep * kb << endl;
  myfile << (int)nstep * kb << endl;
  myfile << (int)mstep << endl;
  myfile << (int)nstep << endl;
  myfile.close();
#endif

  // Start Compute
  ld = 0;
  op = OPCODE_COMPUTE;
  DMA_I[ld++] = op;
  DMA_I[ld++] = mstep * kb;
  DMA_I[ld++] = nstep * kb;
  DMA_I[ld++] = mstep;
  DMA_I[ld++] = nstep;

  TOG2(cout << "Send Compute OP" << endl);
  prf_start(2);
  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[0].dma_wait_send();
  prf_end(2, drv->a_t->fpga_compute);
  TOG2(cout << "Compute Op Sent" << endl);
}

void LoadWeights(acc_container *drv, int m, int mstep, int nstep, int kb,
                 char *wgt_block) {
  if (drv->t.layer_alloced[drv->t.layer]) {
    LoadWeights_Preloaded(drv, m, mstep, nstep, kb, wgt_block);
  } else {
    LoadWeights_Inference(drv, m, mstep, nstep, kb, wgt_block);
  }
}

void StoreOutputs(acc_container *drv, int m, int mstep, int nstep, int n,
                  int out_stride) {
  int *DMA_O = drv->mdma->dmas[0].dma_get_outbuffer();
  TOG2("Compute Start\n");
  prf_start(2);
  drv->mdma->dmas[0].dma_start_recv(nstep * mstep);
  drv->mdma->dmas[0].dma_wait_recv();
  prf_end(2, drv->a_t->fpga_compute);
  TOG2("Compute End\n");
  int out_idx = 0;
  for (uint32_t ns = n; ns < (n + nstep); ns++) {
    float *dst_col = (float *)(drv->out + (ns * out_stride));
    memcpy(&dst_col[m], &DMA_O[out_idx], mstep * sizeof(float));
    out_idx += mstep;
  }
}

struct StoreOutputTask : Task {
  StoreOutputTask(acc_container *drv_, int _M, int _N, int _tile_n, int _tile_m,
                  int _out_stride)
      : drv(drv_), M(_M), N(_N), tile_n(_tile_n), tile_m(_tile_m),
        out_stride(_out_stride) {}
  void Run() override {
    for (uint32_t n = 0; n < N; n += tile_n) {
      uint32_t nstep = std::min(tile_n, tile_n - n);
      for (uint32_t m = 0; m < M; m += tile_m) {
        uint32_t mstep = std::min(tile_m, M - m);
        bfpp_acc::StoreOutputs(drv, m, mstep, nstep, n, out_stride);
      }
    }
  }
  acc_container *drv;
  uint32_t M, N, tile_n, tile_m, out_stride;
};

struct LoadWeightTask : Task {

  LoadWeightTask(acc_container *drv_, int _N, int _M, int _tile_n, int _tile_m,
                 int kb, char *_wgt_block)
      : drv(drv_), N(_N), M(_M), tile_n(_tile_n), tile_m(_tile_m), kb(kb),
        wgt_block(_wgt_block) {}
  void Run() override {
    for (uint32_t n = 0; n < N; n += tile_n) {
      uint32_t nstep = std::min(tile_n, tile_n - n);
      for (uint32_t m = 0; m < M; m += tile_m) {
        uint32_t mstep = std::min(tile_m, M - m);
        bfpp_acc::LoadWeights(drv, m, mstep, nstep, kb, wgt_block);
      }
    }
  }

  acc_container *drv;
  uint32_t N, M, tile_n, tile_m, kb;
  char *wgt_block;
};

} // namespace bfpp_acc
#endif // DRIVER_MT