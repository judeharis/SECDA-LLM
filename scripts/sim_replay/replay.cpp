// Replay captured bfpp_acc MUL_MAT calls through this build's EntryMM and compare bit for bit.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
extern "C" {
void initACC();
void resetPlan();
void updatePlan(int);
void EntryMM(const void *inp, const void *wgt, void *out, int M, int N, int K, int inp_stride,
             int wgt_stride, int out_stride, int wgt_type);
}
static int64_t ulp(float a, float b) {
  int32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4);
  if (x < 0) x = INT32_MIN - x; if (y < 0) y = INT32_MIN - y;
  return llabs((int64_t)x - y);
}
int main(int argc, char **argv) {
  initACC();
  long tot = 0, tot_bad = 0;
  for (int a = 1; a < argc; a++) {
    FILE *f = fopen(argv[a], "rb");
    if (!f) { perror(argv[a]); return 1; }
    int32_t h[8]; fread(h, sizeof(h), 1, f);
    int M = h[0], N = h[1], K = h[2], ty = h[3], is = h[4], ws = h[5], os = h[6];
    std::vector<char> inp((size_t)N * is + 64), wgt((size_t)M * ws + 256), out((size_t)N * os + 64);
    std::vector<float> ref((size_t)N * M);
    fread(inp.data(), 1, (size_t)N * is, f); fread(wgt.data(), 1, (size_t)M * ws, f);
    fread(ref.data(), sizeof(float), ref.size(), f); fclose(f);
    resetPlan(); updatePlan(1);
    EntryMM(wgt.data(), inp.data(), out.data(), M, N, K, is, ws, os, ty);
    long bad = 0; int64_t maxu = 0; double maxa = 0, maxr = 0;
    for (int n = 0; n < N; n++)
      for (int m = 0; m < M; m++) {
        float g = ((float *)(out.data() + (size_t)n * os))[m], r = ref[(size_t)n * M + m];
        if (memcmp(&g, &r, 4)) {
          bad++; int64_t u = ulp(g, r); if (u > maxu) maxu = u;
          double d = fabs((double)g - r); if (d > maxa) maxa = d;
          if (r != 0 && d / fabs(r) > maxr) maxr = d / fabs(r);
        }
      }
    tot += (long)N * M; tot_bad += bad;
    printf("%s M=%d N=%d K=%d q%d layer=%d: %ld/%ld differ, max ulp %lld, max abs %.3g, max rel %.3g\n", argv[a], M, N, K,
           ty, h[7], bad, (long)N * M, (long long)maxu, maxa, maxr);
  }
  printf("TOTAL %ld/%ld outputs differ\n", tot_bad, tot);
  return 0;
}
