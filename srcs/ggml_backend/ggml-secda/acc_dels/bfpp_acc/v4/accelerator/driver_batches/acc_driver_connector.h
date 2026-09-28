#ifndef ACC_DRIVER_CONNECTOR_H2
#define ACC_DRIVER_CONNECTOR_H2



#ifdef __cplusplus
extern "C" {
#endif

void initACC();

void resetPlan();

void updatePlan(int supported_nodes);

bool modelPlanned();

void setLayer(int layer);

#ifdef __cplusplus
void updateProfile(std::chrono::nanoseconds);
#else
void updateProfile(long long nanoseconds);
#endif

// void updateProfile(double time);

bool checkDim(int M, int N, int K);

bool preloadWeights(unsigned wgt_size, int layer, int M, int K, const void *wgt,
                    int wgt_type);

// ne2/ne3 are the destination's batch/head extents beyond the M x N slab
// (ggml_tensor::ne[2]/ne[3]); inp_stride2/3 and out_stride2/3 are the
// corresponding byte strides (ggml_tensor::nb[2]/nb[3]) for the quantized
// input buffer and the destination respectively, letting EntryMM walk each
// batch slice instead of assuming a single flat M x N block.
void EntryMM(const void *inp, const void *wgt, void *out, int M, int N, int K,
             int inp_stride, int wgt_stride, int out_stride, int wgt_type,
             int ne2, int ne3, int inp_stride2, int inp_stride3,
             int out_stride2, int out_stride3);

// Jude: Added - one row per call; mask/logits already point at the row's
// first element (row/broadcast selection happens host-side in
// ops_support.cpp). node_done is true only for a SOFT_MAX node's last row,
// which is when the shared layer counter actually advances (see
// advance_layer() in acc_driver.h) - a multi-row node consumes exactly one
// layer slot, not one per row.
void EntrySoftmax(const float *logits, const void *mask, bool mask_is_f16,
                  bool has_sink, float sink_value, float *out,
                  unsigned int n, float scale, float slope, bool node_done);

#ifdef __cplusplus
}
#endif

#endif // ACC_DRIVER_CONNECTOR_H2
