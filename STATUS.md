# SECDA-LLM — Status

**Objective:** integrate llama.cpp with the SECDA design methodology so LLM accelerators can be
designed, simulated and deployed inside a real inference framework. See [README.md](README.md).

**Last checkpoint:** 2026-09-10

> **First checkpoint, written from repository inspection — not from a working session.** The
> narrative below is inferred from `plan.md`, the working tree and `git log`; no claim here was
> observed running. Treat the next entry as the first with real evidence behind it.

---

## Headline

**49 uncommitted files, and the most significant work is untracked entirely.** `bfpp_acc/v4/`,
`example_acc/`, `test/` and `plan.md` itself exist only on this disk — the last commit
(`183376a "Update"`) predates all of it. The repo is level with its remote, so nothing is
recoverable from `origin` either.

Judging by `plan.md` and `docs/softmax_plan.md`, the active thread is **BFPP_Acc v4**: taking the
standalone softmax accelerator in `example_acc/` and integrating it into the SECDA backend,
driver and `ggml-secda`.

---

## Where things stand

| Area | State |
|---|---|
| Working tree | **49 dirty** — 30 modified, 17 untracked, 2 deleted |
| `bfpp_acc/v4/` | **untracked** — the active accelerator version |
| `example_acc/` | **untracked** — standalone softmax accelerator, per `plan.md` the v4 starting point |
| `plan.md`, `docs/softmax_plan.md` | **untracked** — the design intent is not in git |
| `bfpp_acc/v3/` | mid-restructure — 2 compute files deleted, `accelerator_alt/` and `driver_batches/` added |
| Submodules | `llama.cpp` and `secda_tools` pointers both modified |
| vs `origin/main` | level — ahead 0, behind 0 |

---

## Next session

1. **Commit, or at least stash-and-tag.** 49 files, 17 of them untracked, with no remote copy.
   Everything else is secondary to this. The v3 restructure (2 deletions + 2 new directories)
   is the part most likely to be lost or half-applied.
2. **Track `plan.md` and `docs/softmax_plan.md`.** They hold the design intent for BFPP_Acc v4 and
   the llama-perplexity work, and are currently one `rm` from gone.
3. **Finish BFPP_Acc v4** — integrate `example_acc/`'s standalone softmax into the SECDA backend
   and driver, extending `ggml-secda` for softmax offload behind a `SECDA_BFPP_ACC_V4` path.
   Validate in simulation with `llama-bench` and `test-backend-ops`, per `plan.md`.
4. **Cross-check against BFP_Acc.** That project has a parallel `bfp_softmm` experiment (f32
   softmax matching `ggml_soft_max_ext`) and a five-phase `softmax_plan.md`. Same problem, two
   repos — worth confirming they agree before both are built out.
5. **Commit messages.** Six of eight commits are "Update"/"update". Combined with the untracked
   work, the history currently cannot answer "when did this break".

---

## Key files

| Path | Role |
|---|---|
| [README.md](README.md) | what the platform is, and setup |
| `plan.md` | active design intent — llama-perplexity progress reporting, BFPP_Acc v4 |
| `docs/softmax_plan.md` | softmax integration plan |
| `srcs/ggml_backend/ggml-secda/` | the SECDA ggml backend — 15 of the modified files |
| `srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/v4/` | the in-progress accelerator |
| `example_acc/` | standalone softmax accelerator, v4's starting point |
| `benchmark/scripts/` | benchmark and perplexity harness |

---

## Open risks

- **Untracked work has no second copy.** `example_acc/`, `bfpp_acc/v4/`, `test/` and `plan.md`
  exist on one disk with no remote and no git history.
- **Two submodule pointers are modified** (`llama.cpp`, `secda_tools`). If those commits are not
  pushed in their own repos, a fresh clone cannot reproduce this tree.
- **v3 is mid-restructure.** Two compute files deleted with replacements untracked — v3 may not
  build in its current state, and nothing records whether that is intentional.

---

## Session log

_Most recent 10 entries. Older entries roll into `status-archive/<YYYY>.md`._

### 2026-09-10 — first checkpoint (inspection only)
Created this file as the first use of the `project-checkpoint` skill on a repo other than AMD.
Written from `plan.md`, the working tree and `git log` — nothing here was observed running. The
finding that matters: 49 dirty files with the active work (`bfpp_acc/v4/`, `example_acc/`,
`plan.md`) entirely untracked and no remote copy.
