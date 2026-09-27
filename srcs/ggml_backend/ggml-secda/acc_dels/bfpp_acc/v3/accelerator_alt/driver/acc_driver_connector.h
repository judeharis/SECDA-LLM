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

#ifdef __cplusplus
}
#endif

#endif // ACC_DRIVER_CONNECTOR_H2
