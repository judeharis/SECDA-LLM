#ifndef __ACCNAME_SOFTMAX_UNIT_SC_H__
#define __ACCNAME_SOFTMAX_UNIT_SC_H__

// F32 softmax datapath: softmax(logits*scale + mask*slope), with an optional
// attention-sink correction to the denominator only (never the output row).
// `slope` is precomputed host-side (see driver.h / bfp_softmax.h) so this unit
// never needs head-index/n_head bookkeeping - just scale, slope, an optional
// mask row, and an optional sink value. Ported from
// ggml_compute_forward_soft_max_f32 (ggml/src/ggml-cpu/ops.cpp) and
// ggml_vec_soft_max_f32 (ggml/src/ggml-cpu/vec.cpp) in llama.cpp/GGML.

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
// by ggml_compute_fp16_to_fp32 (bfp_processor_functions.sc.h / bfp_util.h) -
// duplicated rather than shared so the proven matmul path is never touched.
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

    unsigned int n = softmax_n.read();
    float scale = softmax_bits_to_float(softmax_scale_bits.read());
    float slope = softmax_bits_to_float(softmax_slope_bits.read());
    unsigned int flags = softmax_flags.read();
    bool has_mask = flags & 0x1;
    bool mask_f16 = flags & 0x2;
    bool has_sink = flags & 0x4;
    float sink = softmax_bits_to_float(softmax_sink_bits.read());

    // Pass 1: read n logits, apply scale.
    for (unsigned int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
      float v = softmax_bits_to_float(din1.read().data.to_uint());
      softmax_row[i] = v * scale;
    }

    // Pass 2 (optional): read n mask values, add slope*mask.
    if (has_mask) {
      for (unsigned int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
        sc_uint<32> raw = din1.read().data.to_uint();
        float mv = mask_f16 ? softmax_fp16_to_fp32(raw.range(15, 0))
                             : softmax_bits_to_float(raw.to_uint());
        softmax_row[i] += slope * mv;
      }
    }

    // Pass 3: row max (numerical stability), folding in the sink if present.
    float maxv = -3.402823466e+38F; // -FLT_MAX
    for (unsigned int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
      if (softmax_row[i] > maxv) maxv = softmax_row[i];
      wait();
    }
    if (has_sink && sink > maxv) maxv = sink;

    // Pass 4: exp + sum. Sum accumulates in double, matching GGML's
    // ggml_float, even though softmax_row is float.
    double sum = 0.0;
    for (unsigned int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
      float e = BFP_EXPF(softmax_row[i] - maxv);
      softmax_row[i] = e;
      sum += (double)e;
      wait();
    }
    if (has_sink) sum += (double)BFP_EXPF(sink - maxv);

    // Pass 5: normalize by multiplying the reciprocal (not per-element
    // division, matching GGML) and stream results out.
    float recip = (float)(1.0 / sum);
    ADATA d = {0, 0};
    for (unsigned int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
      d.data = softmax_float_to_bits(softmax_row[i] * recip);
      d.tlast = (i == n - 1);
      dout1.write(d);
    }

    HWC_SIG(Softmax_Unit, 0);
    do_softmax.write(false);
    wait();
  }
}

#endif // __ACCNAME_SOFTMAX_UNIT_SC_H__
