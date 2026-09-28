#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#include <cstring>
#include <future>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_secda_init(void);

GGML_BACKEND_API bool ggml_backend_is_secda(ggml_backend_t backend);

// this will also set the number of threads used for secda operations
GGML_BACKEND_API void ggml_backend_secda_set_n_threads(ggml_backend_t backend_secda,
                                               int n_threads);

// plan standalone (uid 0) graphs on compute (default on); see ggml-secda.cpp
GGML_BACKEND_API void ggml_backend_secda_set_auto_plan(ggml_backend_t backend_secda,
                                               bool enable);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_secda_reg(void);

#ifdef __cplusplus
}
#endif
