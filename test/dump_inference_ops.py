#!/usr/bin/env python3
"""
Run llama.cpp inference for a model and print GGML operations from the dumped cgraph.

This uses the OpenVINO backend dump hook:
- env GGML_OPENVINO_DUMP_CGRAPH=1
- file produced: cgraph_ov.txt (or custom path)

Example:
  python3 test/dump_inference_ops.py \
    --model /path/to/model.gguf \
    --prompt "Hello" \
    --n-predict 8
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess
import sys
import os

# Example node line from cgraph_ov.txt (format is fixed-width):
#  -   0: [   32,    1,    1,    1] MUL_MAT              ...
NODE_RE = re.compile(r"^\s*-\s*\d+:\s*\[[^\]]+\]\s+([A-Z0-9_]+)\b")


def parse_ops_from_cgraph(cgraph_path: pathlib.Path) -> list[str]:
    ops: list[str] = []
    for line in cgraph_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = NODE_RE.match(line)
        if not m:
            continue
        ops.append(m.group(1))
    return ops


def run_inference_and_dump(
    llama_cli: pathlib.Path,
    model: pathlib.Path,
    prompt: str,
    n_predict: int,
    threads: int,
    cgraph_path: pathlib.Path,
    workdir: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(llama_cli),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-t",
        str(threads),
    ]

    env = os.environ.copy()
    env["GGML_OPENVINO_DUMP_CGRAPH"] = "1"

    # OpenVINO backend currently writes cgraph_ov.txt in cwd.
    # We run in workdir and then read/rename from there.
    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    default_dump = workdir / "cgraph_ov.txt"
    if default_dump.exists() and default_dump.resolve() != cgraph_path.resolve():
        cgraph_path.write_text(default_dump.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    return result


def pick_default_llama_cli(repo_root: pathlib.Path) -> pathlib.Path:
    # Match your workspace preset output first, then common upstream location.
    candidates = [
        repo_root / "out" / "build" / "SECDA-sim-x64" / "bin" / "llama-cli",
        repo_root / "out" / "build" / "SECDA-nosim-x64" / "bin" / "llama-cli",
        repo_root / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference and print GGML ops executed.")
    parser.add_argument("--model", required=True, help="Path to .gguf model")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--n-predict", type=int, default=8, help="Generated tokens")
    parser.add_argument("--threads", type=int, default=1, help="Inference threads")
    parser.add_argument(
        "--llama-cli",
        default=None,
        help="Path to llama-cli binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root used for default binary detection and working directory",
    )
    parser.add_argument(
        "--cgraph-file",
        default="cgraph_ov.txt",
        help="Path to dumped graph text file",
    )
    parser.add_argument(
        "--keep-going-on-inference-error",
        action="store_true",
        help="Try to parse cgraph dump even if inference command exits non-zero.",
    )

    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    model = pathlib.Path(args.model).resolve()
    cgraph_path = pathlib.Path(args.cgraph_file).resolve()

    if not model.exists():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2

    llama_cli = pathlib.Path(args.llama_cli).resolve() if args.llama_cli else pick_default_llama_cli(repo_root)
    if not llama_cli.exists():
        print("error: llama-cli not found.", file=sys.stderr)
        print(f"checked/expected: {llama_cli}", file=sys.stderr)
        print("tip: pass --llama-cli explicitly", file=sys.stderr)
        return 2

    result = run_inference_and_dump(
        llama_cli=llama_cli,
        model=model,
        prompt=args.prompt,
        n_predict=args.n_predict,
        threads=args.threads,
        cgraph_path=cgraph_path,
        workdir=repo_root,
    )

    if result.returncode != 0 and not args.keep_going_on_inference_error:
        print("error: inference command failed", file=sys.stderr)
        print(f"command: {llama_cli}", file=sys.stderr)
        print(f"return code: {result.returncode}", file=sys.stderr)
        if result.stderr:
            print("stderr:", file=sys.stderr)
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    if not cgraph_path.exists():
        print("error: graph dump file was not produced.", file=sys.stderr)
        print(f"expected: {cgraph_path}", file=sys.stderr)
        print("note: this requires OpenVINO backend path that honors GGML_OPENVINO_DUMP_CGRAPH.", file=sys.stderr)
        if result.stderr:
            print("stderr:", file=sys.stderr)
            print(result.stderr.strip(), file=sys.stderr)
        return 3

    ops = parse_ops_from_cgraph(cgraph_path)
    if not ops:
        print("warning: no ops parsed from cgraph file", file=sys.stderr)
        print(f"file: {cgraph_path}", file=sys.stderr)
        return 4

    counts = collections.Counter(ops)

    print("=== OP_SEQUENCE ===")
    for i, op in enumerate(ops):
        print(f"{i:04d} {op}")

    print("\n=== OP_COUNTS ===")
    for op, n in counts.most_common():
        print(f"{op:28s} {n}")

    print("\n=== SUMMARY ===")
    print(f"total_nodes: {len(ops)}")
    print(f"unique_ops : {len(counts)}")
    print(f"cgraph_file: {cgraph_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
