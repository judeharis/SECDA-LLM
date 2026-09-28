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

#ifdef __cplusplus
}
#endif

#endif // ACC_DRIVER_CONNECTOR_H2
