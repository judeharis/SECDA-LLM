#ifndef ACC_DRIVER_CONNECTOR_H2
#define ACC_DRIVER_CONNECTOR_H2



#ifdef __cplusplus
extern "C" {
#endif

void initACC();

void resetPlan();

void updatePlan(int supported_nodes);

bool modelPlanned();

#ifdef __cplusplus
void updateProfile(std::chrono::nanoseconds);
#else
void updateProfile(long long nanoseconds);
#endif

// void updateProfile(double time);

bool checkDim(int M, int N, int K);

bool preloadWeights(unsigned wgt_size, int layer, int M, int K, const void *wgt,
                    int wgt_type);

void EntryMM(const void *inp, const void *wgt, void *out, int M, int N, int K,
             int inp_stride, int wgt_stride, int out_stride, int wgt_type);

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
