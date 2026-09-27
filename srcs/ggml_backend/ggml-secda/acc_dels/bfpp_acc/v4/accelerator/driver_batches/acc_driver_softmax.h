#ifndef ACC_DRIVER_SOFTMAX_H
#define ACC_DRIVER_SOFTMAX_H

// Jude: Added - host-side driver for the tiled/streaming hardware softmax
// unit. Hides the two-pass tile loop behind a single per-row call
// (EntrySoftmax), the same way MM() already hides its own M/N tiling inside
// one EntryMM call rather than exposing it to the backend.

#include "acc_container.h"
#include <algorithm>
#include <cstring>

namespace bfpp_acc {

// flags bit layout, matches acc_softmax_unit.sc.h / Control_Unit's softmax
// branch.
enum SoftmaxFlagBits {
  SM_HAS_MASK = 0x1,
  SM_MASK_F16 = 0x2,
  SM_HAS_SINK = 0x4,
  SM_IS_FIRST_TILE = 0x8,
  SM_IS_LAST_TILE_PASS1 = 0x10,
  SM_IS_PASS2 = 0x20,
};

static inline unsigned int SoftmaxFloatBits(float f) {
  unsigned int u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

// Sends one tile/pass round: [OPCODE_SOFTMAX, tile_len, scale_bits,
// slope_bits, flags, sink_bits, logits[tile_len], mask[tile_len] if present].
static void SendSoftmaxTile(unsigned int tile_len, float scale, float slope,
                            unsigned int flags, float sink,
                            const float *logits, const void *mask,
                            bool mask_is_f16) {
  int *DMA_I = drv->mdma->dmas[0].dma_get_inbuffer();
  int ld = 0;
  DMA_I[ld++] = OPCODE_SOFTMAX;
  DMA_I[ld++] = (int)tile_len;
  DMA_I[ld++] = (int)SoftmaxFloatBits(scale);
  DMA_I[ld++] = (int)SoftmaxFloatBits(slope);
  DMA_I[ld++] = (int)flags;
  DMA_I[ld++] = (int)SoftmaxFloatBits(sink);

  for (unsigned int i = 0; i < tile_len; i++) {
    DMA_I[ld++] = (int)SoftmaxFloatBits(logits[i]);
  }
  if (mask) {
    if (mask_is_f16) {
      const uint16_t *mask16 = (const uint16_t *)mask;
      for (unsigned int i = 0; i < tile_len; i++) {
        DMA_I[ld++] = (int)(unsigned int)mask16[i];
      }
    } else {
      const float *mask32 = (const float *)mask;
      for (unsigned int i = 0; i < tile_len; i++) {
        DMA_I[ld++] = (int)SoftmaxFloatBits(mask32[i]);
      }
    }
  }

  drv->mdma->dmas[0].dma_change_start(0);
  drv->mdma->dmas[0].dma_start_send(ld);
  drv->mdma->dmas[0].dma_wait_send();
}

// One row: mask/logits point at the row's first element already (row-level
// broadcast selection happens in ops_support.cpp, mirroring CPU's own
// per-row mask-pointer computation). Safe for GGML's inplace variants: pass
// 1 never writes host memory, and pass 2 reads-then-writes each tile before
// advancing, so a tile's output write can't clobber a not-yet-read later
// tile of the same row.
static void Softmax(const float *logits, const void *mask, bool mask_is_f16,
                    bool has_sink, float sink_value, float *out,
                    unsigned int n, float scale, float slope) {
  const unsigned int base_flags =
      (mask ? SM_HAS_MASK : 0) | (mask && mask_is_f16 ? SM_MASK_F16 : 0) |
      (has_sink ? SM_HAS_SINK : 0);

  auto mask_tile_ptr = [&](unsigned int off) -> const void * {
    if (!mask) return nullptr;
    return mask_is_f16 ? (const void *)((const uint16_t *)mask + off)
                        : (const void *)((const float *)mask + off);
  };

  // Pass 1: stats (online-softmax max/sum accumulation, no output).
  for (unsigned int off = 0; off < n; off += SOFTMAX_TILE_N) {
    unsigned int tile_len = std::min((unsigned int)SOFTMAX_TILE_N, n - off);
    unsigned int flags = base_flags;
    if (off == 0) flags |= SM_IS_FIRST_TILE;
    if (off + tile_len >= n) flags |= SM_IS_LAST_TILE_PASS1;
    SendSoftmaxTile(tile_len, scale, slope, flags, sink_value, logits + off,
                    mask_tile_ptr(off), mask_is_f16);
  }

  // Pass 2: output (running max/recip already finalized from pass 1).
  for (unsigned int off = 0; off < n; off += SOFTMAX_TILE_N) {
    unsigned int tile_len = std::min((unsigned int)SOFTMAX_TILE_N, n - off);
    unsigned int flags = base_flags | SM_IS_PASS2;
    SendSoftmaxTile(tile_len, scale, slope, flags, sink_value, logits + off,
                    mask_tile_ptr(off), mask_is_f16);
    drv->mdma->dmas[0].dma_start_recv(tile_len);
    drv->mdma->dmas[0].dma_wait_recv();
    int *DMA_O = drv->mdma->dmas[0].dma_get_outbuffer();
    memcpy(out + off, DMA_O, tile_len * sizeof(float));
  }
}

// m = wgt_rows analogue for softmax: one row per call. `node_done` is true
// only for a SOFT_MAX node's last row - that's when the shared layer
// counter (see advance_layer() in acc_driver.h) actually ticks forward, so
// a multi-row node consumes exactly one layer slot, not one per row.
static void EntrySoftmax(const float *logits, const void *mask,
                         bool mask_is_f16, bool has_sink, float sink_value,
                         float *out, unsigned int n, float scale, float slope,
                         bool node_done) {
  drv->t.layer = dparams.layer;
  drv->a_t = a_t;

  drv->hwc->reset_hwc();
  prf_start(1);

  Softmax(logits, mask, mask_is_f16, has_sink, sink_value, out, n, scale,
          slope);

  SYSC_ON(drv->profile->saveProfile(drv->acc->profiling_vars));
  prf_end(1, a_t->driver_total);

  if (node_done) advance_layer();
}

} // namespace bfpp_acc

#endif // ACC_DRIVER_SOFTMAX_H
