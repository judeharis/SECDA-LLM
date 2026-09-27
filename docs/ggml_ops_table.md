# llama.cpp GGML Ops Table

Generated from:
- ggml enum: llama.cpp/ggml/include/ggml.h
- CPU dispatch switch: llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
- SECDA supports_op switch: llama.cpp/ggml/src/ggml-secda/ggml-secda.cpp
- SECDA graph_compute switch: llama.cpp/ggml/src/ggml-secda/ggml-secda.cpp

| OP | CPU Dispatch | SECDA supports_op | SECDA graph_compute handling |
|---|---|---|---|
| GGML_OP_NONE | yes | no | pass-through case |
| GGML_OP_DUP | yes | no | not handled (default abort) |
| GGML_OP_ADD | yes | no | not handled (default abort) |
| GGML_OP_ADD_ID | yes | no | not handled (default abort) |
| GGML_OP_ADD1 | yes | no | not handled (default abort) |
| GGML_OP_ACC | yes | no | not handled (default abort) |
| GGML_OP_SUB | yes | no | not handled (default abort) |
| GGML_OP_MUL | yes | no | not handled (default abort) |
| GGML_OP_DIV | yes | no | not handled (default abort) |
| GGML_OP_SQR | yes | no | not handled (default abort) |
| GGML_OP_SQRT | yes | no | not handled (default abort) |
| GGML_OP_LOG | yes | no | not handled (default abort) |
| GGML_OP_SIN | yes | no | not handled (default abort) |
| GGML_OP_COS | yes | no | not handled (default abort) |
| GGML_OP_SUM | yes | no | not handled (default abort) |
| GGML_OP_SUM_ROWS | yes | no | not handled (default abort) |
| GGML_OP_CUMSUM | yes | no | not handled (default abort) |
| GGML_OP_MEAN | yes | no | not handled (default abort) |
| GGML_OP_ARGMAX | yes | no | not handled (default abort) |
| GGML_OP_COUNT_EQUAL | yes | no | not handled (default abort) |
| GGML_OP_REPEAT | yes | no | not handled (default abort) |
| GGML_OP_REPEAT_BACK | yes | no | not handled (default abort) |
| GGML_OP_CONCAT | yes | no | not handled (default abort) |
| GGML_OP_SILU_BACK | yes | no | not handled (default abort) |
| GGML_OP_NORM | yes | no | not handled (default abort) |
| GGML_OP_RMS_NORM | yes | no | not handled (default abort) |
| GGML_OP_RMS_NORM_BACK | yes | no | not handled (default abort) |
| GGML_OP_GROUP_NORM | yes | no | not handled (default abort) |
| GGML_OP_L2_NORM | yes | no | not handled (default abort) |
| GGML_OP_MUL_MAT | yes | yes | kernel case |
| GGML_OP_MUL_MAT_ID | yes | no | not handled (default abort) |
| GGML_OP_OUT_PROD | yes | no | kernel case |
| GGML_OP_SCALE | yes | no | not handled (default abort) |
| GGML_OP_SET | yes | no | not handled (default abort) |
| GGML_OP_CPY | yes | no | not handled (default abort) |
| GGML_OP_CONT | yes | no | not handled (default abort) |
| GGML_OP_RESHAPE | yes | no | pass-through case |
| GGML_OP_VIEW | yes | no | pass-through case |
| GGML_OP_PERMUTE | yes | no | pass-through case |
| GGML_OP_TRANSPOSE | yes | no | pass-through case |
| GGML_OP_GET_ROWS | yes | no | not handled (default abort) |
| GGML_OP_GET_ROWS_BACK | yes | no | not handled (default abort) |
| GGML_OP_SET_ROWS | yes | no | not handled (default abort) |
| GGML_OP_DIAG | yes | no | not handled (default abort) |
| GGML_OP_DIAG_MASK_INF | yes | no | not handled (default abort) |
| GGML_OP_DIAG_MASK_ZERO | yes | no | not handled (default abort) |
| GGML_OP_SOFT_MAX | yes | no | not handled (default abort) |
| GGML_OP_SOFT_MAX_BACK | yes | no | not handled (default abort) |
| GGML_OP_ROPE | yes | no | not handled (default abort) |
| GGML_OP_ROPE_BACK | yes | no | not handled (default abort) |
| GGML_OP_CLAMP | yes | no | not handled (default abort) |
| GGML_OP_CONV_TRANSPOSE_1D | yes | no | not handled (default abort) |
| GGML_OP_IM2COL | yes | no | not handled (default abort) |
| GGML_OP_IM2COL_BACK | yes | no | not handled (default abort) |
| GGML_OP_IM2COL_3D | yes | no | not handled (default abort) |
| GGML_OP_CONV_2D | yes | no | not handled (default abort) |
| GGML_OP_CONV_3D | yes | no | not handled (default abort) |
| GGML_OP_CONV_2D_DW | yes | no | not handled (default abort) |
| GGML_OP_CONV_TRANSPOSE_2D | yes | no | not handled (default abort) |
| GGML_OP_POOL_1D | yes | no | not handled (default abort) |
| GGML_OP_POOL_2D | yes | no | not handled (default abort) |
| GGML_OP_POOL_2D_BACK | yes | no | not handled (default abort) |
| GGML_OP_UPSCALE | yes | no | not handled (default abort) |
| GGML_OP_PAD | yes | no | not handled (default abort) |
| GGML_OP_PAD_REFLECT_1D | yes | no | not handled (default abort) |
| GGML_OP_ROLL | yes | no | not handled (default abort) |
| GGML_OP_ARANGE | yes | no | not handled (default abort) |
| GGML_OP_TIMESTEP_EMBEDDING | yes | no | not handled (default abort) |
| GGML_OP_ARGSORT | yes | no | not handled (default abort) |
| GGML_OP_TOP_K | yes | no | not handled (default abort) |
| GGML_OP_LEAKY_RELU | yes | no | not handled (default abort) |
| GGML_OP_TRI | yes | no | not handled (default abort) |
| GGML_OP_FILL | yes | no | not handled (default abort) |
| GGML_OP_FLASH_ATTN_EXT | yes | no | not handled (default abort) |
| GGML_OP_FLASH_ATTN_BACK | yes | no | not handled (default abort) |
| GGML_OP_SSM_CONV | yes | no | not handled (default abort) |
| GGML_OP_SSM_SCAN | yes | no | not handled (default abort) |
| GGML_OP_WIN_PART | yes | no | not handled (default abort) |
| GGML_OP_WIN_UNPART | yes | no | not handled (default abort) |
| GGML_OP_GET_REL_POS | yes | no | not handled (default abort) |
| GGML_OP_ADD_REL_POS | yes | no | not handled (default abort) |
| GGML_OP_RWKV_WKV6 | yes | no | not handled (default abort) |
| GGML_OP_GATED_LINEAR_ATTN | yes | no | not handled (default abort) |
| GGML_OP_RWKV_WKV7 | yes | no | not handled (default abort) |
| GGML_OP_SOLVE_TRI | yes | no | not handled (default abort) |
| GGML_OP_GATED_DELTA_NET | yes | no | not handled (default abort) |
| GGML_OP_UNARY | yes | no | not handled (default abort) |
| GGML_OP_MAP_CUSTOM1 | yes | no | not handled (default abort) |
| GGML_OP_MAP_CUSTOM2 | yes | no | not handled (default abort) |
| GGML_OP_MAP_CUSTOM3 | yes | no | not handled (default abort) |
| GGML_OP_CUSTOM | yes | no | not handled (default abort) |
| GGML_OP_CROSS_ENTROPY_LOSS | yes | no | not handled (default abort) |
| GGML_OP_CROSS_ENTROPY_LOSS_BACK | yes | no | not handled (default abort) |
| GGML_OP_OPT_STEP_ADAMW | yes | no | not handled (default abort) |
| GGML_OP_OPT_STEP_SGD | yes | no | not handled (default abort) |
| GGML_OP_GLU | yes | no | not handled (default abort) |

## Notes
- CPU dispatch reflects presence in the main tensor->op compute switch.
- SECDA supports_op currently advertises only GGML_OP_MUL_MAT.
- SECDA graph_compute includes GGML_OP_OUT_PROD, but supports_op does not advertise it; scheduler placement usually follows supports_op.
