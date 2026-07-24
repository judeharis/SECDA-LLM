#include <cstdlib>

#include "acc_driver.h"
#include "acc_driver_connector.h"
#include "driver_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inside this "extern C" block, I can implement functions in C++, which will
// externally
//   appear as C functions (which means that the function IDs will be their
//   names, unlike the regular C++ behavior, which allows defining multiple
//   functions with the same name (overloading) and hence uses function
//   signature hashing to enforce unique IDs),

void initACC() { bfpp_acc::initACC(); }

void resetPlan() { bfpp_acc::resetPlan(); }

void updatePlan(int supported_nodes) { bfpp_acc::updatePlan(supported_nodes); }

void updateProfile(std::chrono::nanoseconds time) {
  bfpp_acc::updateProfile(time);
}
// void updateProfile(double time) { bfpp_acc::updateProfile(time); }

bool modelPlanned() { return bfpp_acc::modelPlanned(); }

bool preloadWeights(unsigned wgt_size, int layer, int M, int K, const void *wgt,
                    int wgt_type) {
  return bfpp_acc::preloadWeights(wgt_size, layer, M, K, wgt, wgt_type);
}

bool checkDim(int M, int N, int K) { return bfpp_acc::DimCheck(M, N, K); }


void EntryMM(const void *inp, const void *wgt, void *out, int M, int N, int K,
             int inp_stride, int wgt_stride, int out_stride, int wgt_type) {
  bfpp_acc::EntryMM(inp, wgt, out, M, N, K, inp_stride, wgt_stride, out_stride,
                    wgt_type);
}

#ifdef __cplusplus
}
#endif