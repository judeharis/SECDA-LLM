# Why 7B models don't fit on this SoC under llama.cpp — and why bare-metal could change that

## TL;DR

Llama-2-7B, even at the smallest practical quant we tested (Q3_K_S, 2.75 GiB on
disk), cannot run reliably on the KV260 (ZynqMP, 3.8 GiB total RAM) under the
current llama.cpp-based SECDA-LLM stack. This was empirically confirmed on
real hardware, not simulation, on 2026-08-06: the run either stalls for many
minutes re-faulting evicted page-cache pages, or — with mmap prefetch
disabled — gets killed outright by the Linux OOM killer partway through
loading. Neither failure mode touches the BFPP_ACC accelerator or its driver;
both happen before or independently of any SECDA/DMA code path executing.

The root cause is a **host-side memory budget problem inherent to running a
full Linux distribution with llama.cpp's memory model on a 3.8 GiB SoC**, not
a limitation of the accelerator's compute design. A bare-metal (or
minimal-RTOS, no-page-cache) control application built around the same
BFPP_Acc driver primitives could plausibly avoid this failure mode entirely,
because the accelerator itself was never the part that needed multi-gigabyte
resident memory in the first place.

## What we observed

Reproduced directly on the Kria board (`bfps.q3v1.0A` runtime, `BFPS_Q3_v1`
bitstream, `llama-2-7b-hf-q3_k_s.gguf`, 2,948,305,408 bytes ≈ 2.75 GiB):

**Attempt 1 — default `-mmp 1` (llama.cpp's default mmap-prefetch behavior).**
The process sat in kernel state `D` (uninterruptible disk sleep) for minutes,
with this stack:

```
wait_on_page_bit_common → filemap_fault → do_read_fault → handle_pte_fault
→ __handle_mm_fault → faultin_page → __get_user_pages
→ populate_vma_page_range → __mm_populate → vm_mmap_pgoff → ksys_mmap_pgoff
```

That is `mmap(..., MAP_POPULATE, ...)` synchronously prefaulting the entire
model file into page cache — confirmed in this fork's own source:

```c
// llama.cpp/src/llama-mmap.cpp:455
if (prefetch) { flags |= MAP_POPULATE; }
```

`prefetch` is on by default. Observed effective throughput reading the model
file was ~5–15 MB/s, and `free -h` on the board showed it running at its
memory ceiling throughout (`available` pinned at ~2.3 GiB regardless of what
else was going on). A prior full run (the one that originally prompted this
investigation) got much further — 61 of the ~224 MUL_MAT nodes into the first
decode token — before stalling the same way, consistent with page-cache
thrashing appearing at different points depending on exact memory state
rather than at a fixed, deterministic location.

**Attempt 2 — `-mmp 0` (disable the mmap prefetch).** This does not fix the
problem; it changes the failure mode. Weight data now lands in ordinary
(non-reclaimable) anonymous memory instead of reclaimable file-backed page
cache. `dmesg` shows the outcome directly:

```
oom-kill: constraint=CONSTRAINT_NONE, ... task=llama_cli_v1A_s, pid=74891
Out of memory: Killed process 74891 (llama_cli_v1A_s) total-vm:2905140kB, anon-rss:2486896kB
```

The process was killed by the kernel OOM killer after resident anonymous
memory reached ~2.37 GiB (~86% of the file) — it never even finished loading.
With `-mmp 1`, the kernel can quietly evict and re-read reclaimable page-cache
pages under pressure (slow, but survivable); with `-mmp 0`, there is nothing
left to reclaim once physical memory runs out, so the kernel has no option
but to kill the process.

## The quantitative budget

| Item | Size | Source |
|---|---|---|
| Total board RAM | 3.8 GiB | `free -h` |
| CMA reserved for accelerator DMA (4 channels × (192 MiB in + 16 MiB out)) | ~832 MiB | `dmesg`: `udmabuf0..7` allocations, matches the bench log's `DMA 0..3` headers |
| Desktop/OS overhead (GNOME Shell, Xorg, dockerd/containerd, snapd, PipeWire/PulseAudio, CUPS, Jupyter, etc.) | ~300–500 MiB | `ps aux --sort=-%mem` on the board |
| **Genuinely available for the model + inference** | **~2.3 GiB** | `free -h`'s `available` column, stable across every measurement this session |
| Llama-2-7B, Q3_K_S | **2.75 GiB** | file size on disk |

2.75 GiB of raw weights alone already exceeds the ~2.3 GiB actually available
— before accounting for anything else a running inference needs on top:

- **KV cache**: Llama-2-7B has 32 layers, `n_embd=4096`, F16 K+V by default.
  Per-token cost is `32 layers × 2 (K,V) × 4096 × 2 bytes ≈ 512 KiB/token`. A
  modest 2048-token context alone is **~1 GiB** of additional, unavoidable
  memory — on top of the weights that already don't fit.
- **Compute/graph buffers**: ggml's compute graph and per-op scratch buffers
  (quantization work buffers, activation intermediates) add further overhead
  that scales with batch size and hidden dimension.

Even the next quant down (Q2_K, meaningfully smaller than Q3_K_S) would leave
little to no headroom once KV cache and compute buffers are added at any
non-trivial context length. This is not a "just disable one flag" problem —
the numbers don't close regardless of loading strategy, because the
constraint is the *sum* of resident weights + KV cache + compute buffers
against a fixed ~2.3 GiB ceiling, and the weights alone already consume more
than that ceiling at 7B/Q3.

## Why this isn't a BFPP_ACC / SECDA driver problem

Every failure observed happens in host-side model loading — before the
accelerator's `EntryMM`/`EntrySoftmax` driver entry points are ever reached
(in the `-mmp 0` OOM case) or independent of them (in the `-mmp 1` page-fault
case, which stalls on `mmap()`/page reclaim regardless of whether SECDA is
even the active backend). `test-backend-ops` and `llama-bench` runs against
BFPP_ACC_V4 (both MUL_MAT and the new tiled SOFT_MAX) pass cleanly on models
that actually fit in the available budget (TinyLlama-1B, MobileLLM-125M,
Llama-3.2-1B). The accelerator itself is also not designed around holding
multi-gigabyte buffers in the first place — its on-chip tiling constants
(`SUP_KMB=1024`, `SUP_KNB=512` blocks, `QK_K=256` elements/block, in
[acc_config.sc.h](../srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/v4/accelerator/acc_config.sc.h))
mean it only ever holds a few hundred KB of one weight tile on-chip at a
time. The 2.75 GiB memory problem is entirely a property of **how llama.cpp
loads and holds the full quantized model host-side under Linux**, layered on
top of a board that's also running a full desktop environment rather than a
headless/minimal image.

## Why a bare-metal approach could plausibly work

A bare-metal (or minimal-RTOS, no page-cache) control application built
around the same driver primitives (`acc_driver_connector.h`'s `EntryMM`,
`EntrySoftmax`, `preloadWeights`) removes the specific mechanisms that caused
both failures above, because none of them are inherent to running the model —
they're inherent to running it *through a general-purpose OS's virtual memory
subsystem*:

- **No whole-file residency requirement.** The accelerator already only
  consumes one weight tile at a time (`LoadWeights`/`tile_kmb`/`tile_knb` in
  `acc_driver.h`). A bare-metal loader could read exactly the bytes needed for
  the tile about to be DMA'd from storage into a small fixed staging buffer
  (order of the `SUP_KMB`/`SUP_KNB` tile size, not gigabytes), then discard
  it — mirroring what the *accelerator* already does, instead of requiring
  the *host* to hold the entire 2.75 GiB file resident (via page cache or
  heap) as llama.cpp's mmap-based loader does today.
- **No OS-level nondeterminism.** The failure mode we hit was itself
  flag-dependent (slow thrash vs. hard OOM-kill) — a symptom of virtual
  memory overcommit, reclaimable-vs-anonymous memory accounting, and a page
  cache shared with an entire desktop environment, none of which exist in a
  statically-linked bare-metal image with a build-time-fixed memory map.
  There is no OOM killer to invoke because there is no dynamic overcommit to
  exceed.
- **A deterministic, purpose-built budget.** KV cache size, weight
  staging-buffer size, and the accelerator's CMA/DMA regions could all be
  sized precisely for the target model and context length at build time,
  instead of the current setup's large fixed DMA reservation (~832 MiB across
  4 channels, presumably sized generically rather than per-model) competing
  with an entire desktop OS's footprint for the same 3.8 GiB.
- **Removing OS overhead entirely.** GNOME Shell, Xorg, Docker, snapd,
  PipeWire/PulseAudio, and CUPS collectively consumed several hundred MB for
  no benefit to an inference workload — none of that exists in a bare-metal
  target.

## What bare-metal does *not* solve for free

- It's a substantial reimplementation, not a flag flip: weight streaming, KV
  cache management, tokenization, and ggml's graph execution model all
  currently lean on Linux/libc facilities (mmap, malloc, the page cache) that
  a bare-metal target would need to replace deliberately.
- Storage throughput is still finite. We measured ~5–15 MB/s from this
  board's storage; a bare-metal streaming loader is still bounded by that for
  whatever data it has to read per token. The advantage is *not* re-reading
  data unnecessarily (which is what Linux's page-cache thrashing did here),
  not a faster storage device.
- It only removes *host-side* memory pressure. The accelerator's own on-chip
  buffer sizes (`SOFTMAX_TILE_N`, `SUP_KMB`/`SUP_KNB`) are unaffected either
  way — they were never the bottleneck.

## Recommendation

Short term: keep 7B-class models off this board's llama.cpp path — it's
confirmed impossible at Q3 and very unlikely to have workable headroom even
at Q2 once KV cache and compute buffers are counted. Continue validating
BFPP_ACC_V4 (MUL_MAT + the new tiled SOFT_MAX) on the models already proven to
fit this board's real budget (TinyLlama-1B, MobileLLM-125M, Llama-3.2-1B).

If 7B-class support becomes a real requirement on this exact SoC, a
bare-metal (or minimal-RTOS) rewrite of the host control path — reusing the
existing BFPP_Acc driver/accelerator design as-is — is the credible way to
get there, specifically because it targets the mechanism we just proved
empirically responsible for the failure (general-purpose OS memory
management), not because the accelerator's compute design is limited to
small models.
