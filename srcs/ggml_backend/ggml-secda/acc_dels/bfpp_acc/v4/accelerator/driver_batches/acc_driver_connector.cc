#include <cstdlib>

#include "acc_driver.h"
#include "acc_driver_softmax.h" // Jude: Added - after acc_driver.h (needs advance_layer())
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

void setLayer(int layer) { bfpp_acc::setLayer(layer); }

bool preloadWeights(unsigned wgt_size, int layer, int M, int K, const void *wgt,
                    int wgt_type) {
  return bfpp_acc::preloadWeights(wgt_size, layer, M, K, wgt, wgt_type);
}

bool checkDim(int M, int N, int K) { return bfpp_acc::DimCheck(M, N, K); }


void EntryMM(const void *inp, const void *wgt, void *out, int M, int N, int K,
             int inp_stride, int wgt_stride, int out_stride, int wgt_type,
             int ne2, int ne3, int inp_stride2, int inp_stride3,
             int out_stride2, int out_stride3) {
  bfpp_acc::EntryMM(inp, wgt, out, M, N, K, inp_stride, wgt_stride, out_stride,
                    wgt_type, ne2, ne3, inp_stride2, inp_stride3, out_stride2,
                    out_stride3);
}

// Jude: Added
void EntrySoftmax(const float *logits, const void *mask, bool mask_is_f16,
                  bool has_sink, float sink_value, float *out,
                  unsigned int n, float scale, float slope, bool node_done) {
  bfpp_acc::EntrySoftmax(logits, mask, mask_is_f16, has_sink, sink_value, out,
                         n, scale, slope, node_done);
}

#ifdef __cplusplus
}
#endif