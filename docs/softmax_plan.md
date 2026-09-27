## Plan: SECDA Softmax Node + Kernel Evaluation

Build a forward-only softmax evaluation and integration plan for SECDA by tracing the current GGML softmax node semantics (especially `ggml_soft_max_ext`) and the CPU f32 kernel behavior, then defining a gated offload path with deterministic fallback to CPU. The approach minimizes risk by first matching existing numerical behavior and graph scheduling rules before adding accelerator-specific optimizations.

**Steps**
1. Phase 1 - Baseline behavior capture (blocks all later phases)
1.1 Document softmax node construction semantics from graph build through GGML op creation.
1.2 Capture exact parameter meaning and data contracts for `scale`, `max_bias`, optional mask, and optional sinks.
1.3 Capture tensor shape/broadcast invariants enforced by GGML assertions for `ggml_soft_max_ext`.

2. Phase 2 - CPU f32 kernel decomposition (depends on Phase 1)
2.1 Break down `ggml_compute_forward_soft_max_f32` into deterministic sub-stages: copy, scale, mask/ALiBi fusion, max-reduction, exp+sum, sink correction, normalize.
2.2 Record multithreading decomposition (row partitioning by `ith/nth`) and temporary workspace layout (`wdata`).
2.3 Record vectorized implementation details from `ggml_vec_soft_max_f32` and scalar tail behavior.
2.4 Define numerical parity targets against CPU output (absolute/relative tolerance and softmax row-sum checks).

3. Phase 3 - Node-level accelerator integration design (depends on Phases 1-2)
3.1 Define op support contract additions in SECDA device support checks for `GGML_OP_SOFT_MAX` (forward path).
3.2 Define graph-compute dispatch additions in SECDA backend compute switch.
3.3 Define lightweight driver-facing wrapper API for softmax execution and parameter marshalling.
3.4 Keep CPU fallback mandatory for unsupported shape/type/mask configurations.

4. Phase 4 - Gated offload policy design (parallel with Phase 3 after 3.1)
4.1 Introduce deterministic gating predicates (dtype, contiguity, supported mask type, shape bounds).
4.2 Add heuristic thresholds (for example sequence length and head count) to avoid low-benefit offload.
4.3 Define observability fields and logs for each gate decision (accepted/rejected + reason).
4.4 Decide default policy: gated offload enabled with safe fallback.

5. Phase 5 - Validation matrix and node/kernel evaluation artifacts (depends on Phases 3-4)
5.1 Build a softmax-focused test matrix by varying mask presence/type, ALiBi on/off, sink on/off, and sequence widths.
5.2 Add per-node correctness checks comparing SECDA output to CPU f32 reference.
5.3 Add stress validation for edge values (large positive/negative logits, masked rows).
5.4 Define acceptance criteria: parity, stability (no NaN/Inf), and expected scheduler/backend assignment.

6. Phase 6 - Performance and rollout (depends on Phase 5)
6.1 Measure CPU vs SECDA latency per softmax node shape bucket.
6.2 Tune gate thresholds using measured crossover points.
6.3 Stage rollout behind a runtime switch so regression triage remains simple.

**V1 Offload Contract**
1. Supported node form
1.1 Operation: GGML_OP_SOFT_MAX created by ggml_soft_max_ext.
1.2 Input logits type: F32 only.
1.3 Logits tensor layout: contiguous.
1.4 Mask: optional; contiguous F32 or F16.
1.5 ALiBi: allowed when max_bias > 0 and mask is present.
1.6 Sinks: allowed by contract, but v1 can be optionally gated to CPU if hardware path is not ready.

2. Required semantic equivalence to CPU
2.1 Compute softmax over ne0 dimension per row/head/batch slice.
2.2 Preserve fused formula: softmax((logits * scale) + (mask * slope(head))).
2.3 Preserve max-subtraction stabilization and sink denominator correction.
2.4 Preserve broadcast behavior for mask dimensions (ne12/ne13) exactly as GGML asserts.

3. Explicit fallback conditions (force CPU)
3.1 Non-F32 logits.
3.2 Non-contiguous logits or mask.
3.3 Unsupported mask representation.
3.4 Unsupported shape bounds for accelerator kernel.
3.5 Sink-enabled path when sink support is disabled in v1 build.
3.6 Gate heuristic says expected speedup is below threshold.

4. Gate output schema (for logs/counters)
4.1 Decision: offload or fallback.
4.2 Reason code: dtype, layout, mask, shape, sinks, heuristic.
4.3 Node metadata: ne0-ne3, mask type, scale, max_bias, sinks present.

5. Initial heuristic recommendations
5.1 Prefer offload when sequence width is medium-to-large and mask fusion is active.
5.2 Prefer CPU for tiny rows where transfer/launch dominates.
5.3 Keep thresholds configurable and tune from profile crossover data.


**Relevant files**
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/src/llama-graph.cpp - attention graph build path where `ggml_soft_max_ext` is emitted and sinks are attached.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml.c - `ggml_soft_max_impl` op construction, assertion contracts, and op params packing.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/include/ggml.h - public op contracts and shape/broadcast semantics for `ggml_soft_max_ext`.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-cpu/ops.cpp - forward f32 softmax compute path and mask/ALiBi/sink fusion logic.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-cpu/vec.cpp - vectorized exp/sum core used by softmax.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c - task partitioning and workspace sizing for `GGML_OP_SOFT_MAX`.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-secda/ggml-secda.cpp - backend support checks and node dispatch switch that must be extended for softmax.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-secda/ops_support.h - SECDA op wrapper declarations to extend with softmax entry points.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-secda/ops_support.cpp - SECDA op wrapper implementations to add driver/kernel bridge for softmax.
- /mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/ggml/src/ggml-backend.cpp - scheduler behavior and support-based backend assignment logic.

**Verification**
1. Functional parity
1.1 Compare SECDA softmax output against CPU f32 output for each matrix-case in the validation grid.
1.2 Validate row sums are approximately 1.0 and outputs are finite.
2. Graph/scheduler correctness
2.1 Confirm `GGML_OP_SOFT_MAX` nodes are assigned to SECDA only when gating says supported.
2.2 Confirm unsupported cases consistently route to CPU.
3. Stability checks
3.1 Validate no NaN/Inf across masked and ALiBi-enabled runs.
3.2 Validate sink-path correctness where sink tensors are present.
4. Performance checks
4.1 Collect latency distributions by shape bucket for CPU vs SECDA.
4.2 Verify gate thresholds improve end-to-end runtime over always-CPU baseline.

**Decisions**
- In scope: forward inference softmax only (`GGML_OP_SOFT_MAX` path, including `ggml_soft_max_ext` semantics).
- Out of scope for first pass: backward/training op (`GGML_OP_SOFT_MAX_BACK`) and gradient kernels.
- Policy decision: gated offload (not always-offload), with strict CPU fallback.
- Numerical goal: match existing CPU f32 behavior including mask fusion, ALiBi slope logic, and sink normalization correction.

**Further Considerations**
1. Gate thresholds source
Recommendation: start with conservative static thresholds from profiling, then tune from measured crossover points.
2. Mask type coverage for v1
Recommendation: support both F16 and F32 masks if driver path is straightforward; otherwise support F32 first and leave F16 gated to CPU.
3. Sinks support for v1
Recommendation: if sink integration is expensive in hardware, keep sink-enabled nodes on CPU initially and offload only sink-free nodes in v1.
