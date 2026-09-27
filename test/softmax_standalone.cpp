#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

// Standalone softmax implementation that mirrors llama.cpp CPU backend call flow:
// - ggml_vec_soft_max_f32
// - ggml_compute_forward_soft_max_f32
//
// This file is self-contained so it can be copied into another repository.

using ggml_float = float;
using ggml_fp16_t = uint16_t;

static constexpr int CACHE_LINE_SIZE_F32 = 16; // 64 bytes / sizeof(float)

enum ggml_type {
    GGML_TYPE_F16 = 1,
    GGML_TYPE_F32 = 2,
};

struct ggml_tensor {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1}; // extents
    int64_t nb[4] = {0, 0, 0, 0}; // byte strides
    void * data = nullptr;
    ggml_tensor * src[3] = {nullptr, nullptr, nullptr};
    std::array<uint8_t, 8> op_params{}; // scale, max_bias
};

struct ggml_compute_params {
    int ith = 0;
    int nth = 1;
    void * wdata = nullptr;
};

static inline float ggml_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h & 0x7C00u) >> 10;
    const uint32_t frac = (h & 0x03FFu);

    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            // subnormal
            uint32_t mant = frac;
            int e = -1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            const uint32_t exp32 = (uint32_t) (127 - 15 + 1 + e);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // inf / nan
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        const uint32_t exp32 = exp + (127 - 15);
        out = sign | (exp32 << 23) | (frac << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline bool ggml_is_contiguous(const ggml_tensor * t) {
    if (t->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        const int64_t expected = t->nb[i - 1] * std::max<int64_t>(1, t->ne[i - 1]);
        if (t->nb[i] != expected) {
            return false;
        }
    }
    return true;
}

static inline bool ggml_are_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < 4; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

static inline int64_t ggml_nrows(const ggml_tensor * t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}

static inline void ggml_vec_cpy_f32(int n, float * dst, const float * src) {
    std::memcpy(dst, src, (size_t) n * sizeof(float));
}

static inline void ggml_vec_scale_f32(int n, float * x, float s) {
    for (int i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

static inline void ggml_vec_max_f32(int n, float * out, const float * x) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        m = std::max(m, x[i]);
    }
    *out = m;
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// Lane-wise exp fallback to keep the same call shape as llama.cpp's NEON softmax path.
// Replace with a faster vector exp approximation if needed.
static inline float32x4_t ggml_v_expf(float32x4_t x) {
    alignas(16) float tmp[4];
    vst1q_f32(tmp, x);
    tmp[0] = std::exp(tmp[0]);
    tmp[1] = std::exp(tmp[1]);
    tmp[2] = std::exp(tmp[2]);
    tmp[3] = std::exp(tmp[3]);
    return vld1q_f32(tmp);
}
#endif

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;

#if defined(__ARM_NEON) && defined(__aarch64__)
    // Copied structure from llama.cpp NEON branch.
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i), vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float) vaddvq_f32(val);
    }
#endif

    for (; i < n; ++i) {
        float val = std::exp(x[i] - max);
        sum += (ggml_float) val;
        y[i] = val;
    }

    return sum;
}

void ggml_compute_forward_soft_max_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, dst->op_params.data() + 0, sizeof(float));
    std::memcpy(&max_bias, dst->op_params.data() + sizeof(float), sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head = (uint32_t) ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((double) n_head));

    const float m0 = std::pow(2.0f, -(max_bias) / n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);

    float * wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;
    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const float * sk = src2 ? (const float *) src2->data : nullptr;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const int64_t i11 = i01;
                const int64_t i12 = i02 % ne12;
                const int64_t i13 = i03 % ne13;

                const uint32_t h = (uint32_t) i02;
                const float slope = (max_bias > 0.0f)
                    ? (h < n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1))
                    : 1.0f;

                float * sp = (float *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                float * dp = (float *) ((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                ggml_fp16_t * mp_f16 = src1
                    ? (ggml_fp16_t *) ((char *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13)
                    : nullptr;
                float * mp_f32 = src1
                    ? (float *) ((char *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13)
                    : nullptr;

                ggml_vec_cpy_f32((int) ne00, wp, sp);
                ggml_vec_scale_f32((int) ne00, wp, scale);

                if (mp_f32) {
                    if (use_f16) {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * ggml_fp16_to_fp32(mp_f16[i]);
                        }
                    } else {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * mp_f32[i];
                        }
                    }
                }

                float maxv = -std::numeric_limits<float>::infinity();
                ggml_vec_max_f32((int) ne00, &maxv, wp);

                if (sk) {
                    maxv = std::max(maxv, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32((int) ne00, dp, wp, maxv);
                assert(sum > 0.0f);

                if (sk) {
                    sum += (ggml_float) std::exp(sk[i02] - maxv);
                }

                sum = 1.0f / sum;
                ggml_vec_scale_f32((int) ne00, dp, sum);

                for (int i = 0; i < ne00; ++i) {
                    assert(!std::isnan(dp[i]));
                    assert(!std::isinf(dp[i]));
                }
            }
        }
    }
}

// Convenience wrapper for standalone random-data testing.
// Uses the same backend-call entrypoint and tensor wiring as above.
void run_softmax_ext_like_llama(
    const float * src0,
    const void * mask,
    ggml_type mask_type,
    const float * sinks,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    float scale,
    float max_bias,
    float * dst_data) {

    ggml_tensor src0_t;
    src0_t.type = GGML_TYPE_F32;
    src0_t.ne[0] = ne0;
    src0_t.ne[1] = ne1;
    src0_t.ne[2] = ne2;
    src0_t.ne[3] = ne3;
    src0_t.nb[0] = sizeof(float);
    src0_t.nb[1] = src0_t.nb[0] * ne0;
    src0_t.nb[2] = src0_t.nb[1] * ne1;
    src0_t.nb[3] = src0_t.nb[2] * ne2;
    src0_t.data = (void *) src0;

    ggml_tensor mask_t;
    if (mask != nullptr) {
        mask_t.type = mask_type;
        mask_t.ne[0] = ne0;
        mask_t.ne[1] = ne1;
        mask_t.ne[2] = ne2;
        mask_t.ne[3] = ne3;
        const int64_t eb = mask_type == GGML_TYPE_F16 ? (int64_t) sizeof(ggml_fp16_t) : (int64_t) sizeof(float);
        mask_t.nb[0] = eb;
        mask_t.nb[1] = mask_t.nb[0] * mask_t.ne[0];
        mask_t.nb[2] = mask_t.nb[1] * mask_t.ne[1];
        mask_t.nb[3] = mask_t.nb[2] * mask_t.ne[2];
        mask_t.data = const_cast<void *>(mask);
    }

    ggml_tensor sinks_t;
    if (sinks != nullptr) {
        sinks_t.type = GGML_TYPE_F32;
        sinks_t.ne[0] = ne2;
        sinks_t.nb[0] = sizeof(float);
        sinks_t.data = (void *) sinks;
    }

    ggml_tensor dst;
    dst.type = GGML_TYPE_F32;
    dst.ne[0] = ne0;
    dst.ne[1] = ne1;
    dst.ne[2] = ne2;
    dst.ne[3] = ne3;
    dst.nb[0] = sizeof(float);
    dst.nb[1] = dst.nb[0] * ne0;
    dst.nb[2] = dst.nb[1] * ne1;
    dst.nb[3] = dst.nb[2] * ne2;
    dst.data = dst_data;
    dst.src[0] = &src0_t;
    dst.src[1] = (mask != nullptr) ? &mask_t : nullptr;
    dst.src[2] = (sinks != nullptr) ? &sinks_t : nullptr;

    std::memcpy(dst.op_params.data(), &scale, sizeof(float));
    std::memcpy(dst.op_params.data() + sizeof(float), &max_bias, sizeof(float));

    std::vector<float> wdata((size_t) (ne0 + CACHE_LINE_SIZE_F32));
    ggml_compute_params params;
    params.ith = 0;
    params.nth = 1;
    params.wdata = wdata.data();

    ggml_compute_forward_soft_max_f32(&params, &dst);
}

#ifdef SOFTMAX_STANDALONE_MAIN
int main() {
    const int64_t ne0 = 128; // softmax width
    const int64_t ne1 = 16;  // rows
    const int64_t ne2 = 4;   // heads
    const int64_t ne3 = 1;   // batch

    std::vector<float> logits((size_t) (ne0 * ne1 * ne2 * ne3));
    std::vector<float> mask((size_t) (ne0 * ne1 * ne2 * ne3));
    std::vector<float> sinks((size_t) ne2);
    std::vector<float> out((size_t) (ne0 * ne1 * ne2 * ne3));

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dlogit(-8.0f, 8.0f);
    std::uniform_real_distribution<float> dmask(-10.0f, 0.0f);

    for (float & v : logits) {
        v = dlogit(rng);
    }
    for (float & v : mask) {
        v = dmask(rng);
    }
    for (float & v : sinks) {
        v = dmask(rng);
    }

    run_softmax_ext_like_llama(
        logits.data(),
        mask.data(),
        GGML_TYPE_F32,
        sinks.data(),
        ne0,
        ne1,
        ne2,
        ne3,
        0.125f,
        8.0f,
        out.data());

    double checksum = 0.0;
    for (float v : out) {
        checksum += v;
    }

    std::cout << "softmax output checksum = " << checksum << "\n";

    // quick row-sum sanity for first head/batch rows
    for (int r = 0; r < 3; ++r) {
        double rowsum = 0.0;
        const size_t base = (size_t) r * (size_t) ne0;
        for (int i = 0; i < ne0; ++i) {
            rowsum += out[base + (size_t) i];
        }
        std::cout << "row " << r << " sum = " << rowsum << "\n";
    }

    return 0;
}
#endif
