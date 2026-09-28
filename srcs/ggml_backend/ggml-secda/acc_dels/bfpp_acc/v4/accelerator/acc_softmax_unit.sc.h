#ifndef __ACCNAME_SOFTMAX_UNIT_SC_H__
#define __ACCNAME_SOFTMAX_UNIT_SC_H__

// Jude: Added - F32 tiled/streaming softmax datapath:
// softmax(logits*scale + mask*slope), with an optional attention-sink
// correction to the denominator only (never the output row). `slope` is
// precomputed host-side (see acc_driver_softmax.h) so this unit never needs
// head-index/n_head bookkeeping - just scale, slope, an optional mask row,
// and an optional sink value. Per-element math ported from
// example_acc/bfp_softmm/v1/accelerator/acc_softmax_unit.sc.h (itself a port
// of ggml_compute_forward_soft_max_f32 / ggml_vec_soft_max_f32 in
// llama.cpp/GGML). Unlike that prototype (whole row on-chip, capped at
// SOFTMAX_MAX_N), this unit processes the row in SOFTMAX_TILE_N-sized tiles
// across two host-driven passes (see acc_driver_softmax.h for the loop):
// pass 1 computes running max/sum via the standard online-softmax rescale
// identity, pass 2 re-streams the same tiles and emits the normalized
// output - so rows of any width are supported with a small, fixed on-chip
// buffer.

// Plain libm expf, used both for -DSYSC behavioral simulation and for HLS
// synthesis. hls_math.h's hls::expf (a templated hotbm/table-based
// implementation) was tried first but crashes Vivado HLS 2019.2's SystemC
// elaborator with a StackOverflowError while evaluating a constant
// expression (autopilot.scelaborator.ScExpressionCalculator) - a known-
// fragile part of the toolchain's SystemC front end, distinct from its
// plain-C HLS front end. Plain expf() is directly synthesizable by Vivado
// HLS's built-in floating-point library without going through hls_math.h.
#include <cmath>
#define BFP_EXPF(x) expf(x)

static inline float softmax_bits_to_float(unsigned int bits) {
  union {
    unsigned int as_bits;
    float as_value;
  } u;
  u.as_bits = bits;
  return u.as_value;
}

static inline unsigned int softmax_float_to_bits(float f) {
  union {
    float as_value;
    unsigned int as_bits;
  } u;
  u.as_value = f;
  return u.as_bits;
}

// Fresh, self-contained copy of the same fp16->fp32 bit-trick algorithm used
// by ggml_compute_fp16_to_fp32 (bfp_processor_functions.sc.h) - duplicated
// rather than shared so the proven matmul path is never touched.
static inline float softmax_fp16_to_fp32(sc_uint<16> h) {
  const uint32_t w = (uint32_t)h.to_uint() << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = softmax_bits_to_float(UINT32_C(0x7800000));

  const float normalized_value =
      softmax_bits_to_float((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      softmax_bits_to_float((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff
                  ? softmax_float_to_bits(denormalized_value)
                  : softmax_float_to_bits(normalized_value));
  return softmax_bits_to_float(result);
}

void ACCNAME::Softmax_Unit() {
  HWC_SIG(Softmax_Unit, 0);
  wait();
  while (1) {
    HWC_SIG(Softmax_Unit, 0);
    while (!do_softmax.read()) wait();
    HWC_SIG(Softmax_Unit, 1);

    unsigned int tile_len = softmax_n.read();
    float scale = softmax_bits_to_float(softmax_scale_bits.read());
    float slope = softmax_bits_to_float(softmax_slope_bits.read());
    unsigned int flags = softmax_flags.read();
    bool has_mask = flags & 0x1;
    bool mask_f16 = flags & 0x2;
    bool has_sink = flags & 0x4;
    bool is_first_tile = flags & 0x8;
    bool is_last_tile_pass1 = flags & 0x10;
    bool is_pass2 = flags & 0x20;
    float sink = softmax_bits_to_float(softmax_sink_bits.read());

    if (is_first_tile) {
      sm_running_max = -3.402823466e+38F; // -FLT_MAX
      sm_running_sum = 0.0;
    }

    // Read tile_len logits, apply scale.
    for (unsigned int i = 0; i < tile_len; i++) {
#pragma HLS pipeline II = 1
      float v = softmax_bits_to_float(din1.read().data.to_uint());
      softmax_row[i] = v * scale;
      DWAIT(SCHED_BFP_Acc_Softmax_Unit_L1_1);
    }

    // Optional: read tile_len mask values, add slope*mask.
    if (has_mask) {
      for (unsigned int i = 0; i < tile_len; i++) {
#pragma HLS pipeline II = 1
        sc_uint<32> raw = din1.read().data.to_uint();
        float mv = mask_f16 ? softmax_fp16_to_fp32(raw.range(15, 0))
                             : softmax_bits_to_float(raw.to_uint());
        softmax_row[i] += slope * mv;
        DWAIT(SCHED_BFP_Acc_Softmax_Unit_L1_2);
      }
    }

    if (!is_pass2) {
      // Pass 1 (stats): online-softmax rescale update, folding this tile
      // into the running max/sum computed so far across the row.
      float tile_max = -3.402823466e+38F; // -FLT_MAX
      for (unsigned int i = 0; i < tile_len; i++) {
#pragma HLS pipeline II = 1
        if (softmax_row[i] > tile_max) tile_max = softmax_row[i];
        wait();
      }

      float new_max = (tile_max > sm_running_max) ? tile_max : sm_running_max;
      double tile_sum = 0.0;
      for (unsigned int i = 0; i < tile_len; i++) {
#pragma HLS pipeline II = 1
        tile_sum += (double)BFP_EXPF(softmax_row[i] - new_max);
        wait();
        DWAIT(SCHED_BFP_Acc_Softmax_Unit_L1_4);
      }
      // Rescale the running sum to the new max before folding in this
      // tile's contribution - the standard online-softmax identity. On the
      // first tile this degenerates safely: expf(-FLT_MAX - anything)
      // underflows to 0.0, never NaN/Inf.
      sm_running_sum =
          sm_running_sum * (double)BFP_EXPF(sm_running_max - new_max) +
          tile_sum;
      sm_running_max = new_max;

      if (is_last_tile_pass1) {
        // Sink correction: matches CPU semantics exactly - the sink can
        // itself become the new row max (rescaling the sum again), then
        // always contributes to the denominator (never the output row).
        if (has_sink && sink > sm_running_max) {
          sm_running_sum *= (double)BFP_EXPF(sm_running_max - sink);
          sm_running_max = sink;
        }
        if (has_sink) {
          sm_running_sum += (double)BFP_EXPF(sink - sm_running_max);
        }
        sm_recip = (float)(1.0 / sm_running_sum);
      }
    } else {
      // Pass 2 (output): running max/recip are already finalized from pass
      // 1 - normalize by reciprocal-multiply (not per-element division,
      // matching GGML) and stream results out.
      ADATA d = {0, 0};
      for (unsigned int i = 0; i < tile_len; i++) {
#pragma HLS pipeline II = 1
        float e = BFP_EXPF(softmax_row[i] - sm_running_max);
        d.data = softmax_float_to_bits(e * sm_recip);
        d.tlast = (i == tile_len - 1);
        dout1.write(d);
        DWAIT(SCHED_BFP_Acc_Softmax_Unit_L1_5);
      }
    }

    HWC_SIG(Softmax_Unit, 0);
    do_softmax.write(false);
    wait();
  }
}

#endif // __ACCNAME_SOFTMAX_UNIT_SC_H__
