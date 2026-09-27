#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

MODEL_KEYS = [
    "MobileLLMQ2",
    "GPT2Q2",
    "NanoMistralQ3S",
    "llama600M",
    "TinyLlama1Q2",
    "MobileLLaMAQ2",
    "Mamba790Q3S",
    "Mamba130Q3S",
]


def find_llama_cli(repo_root: pathlib.Path) -> pathlib.Path:
    candidates = [
        repo_root / "out" / "build" / "SECDA-sim-x64" / "bin" / "llama-cli",
        repo_root / "out" / "build" / "SECDA-nosim-x64" / "bin" / "llama-cli",
        repo_root / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def find_model_file(repo_root: pathlib.Path, model_file: str) -> pathlib.Path | None:
    roots = [
        repo_root / "llama.cpp" / "models" / "secda_models",
        repo_root / "llama.cpp" / "models",
        repo_root / "models",
    ]
    for r in roots:
        p = r / model_file
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch op dump for selected model keys")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--n-predict", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-dir", default="test/inference_ops_reports")
    parser.add_argument("--timeout-sec", type=int, default=120, help="Per-model timeout in seconds")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_map = json.loads((repo_root / "benchmark/configs/models/models_dict.json").read_text(encoding="utf-8"))
    script = repo_root / "test" / "dump_inference_ops.py"
    llama_cli = find_llama_cli(repo_root)

    index_lines = ["# Inference Ops Reports Index", ""]

    for key in MODEL_KEYS:
        model_file = model_map.get(key)
        model_path = find_model_file(repo_root, model_file) if model_file else None
        md_path = out_dir / f"{key}.md"
        cgraph_path = out_dir / f"{key}.cgraph_ov.txt"

        lines = [f"# Inference Ops Report: {key}", ""]
        lines.append(f"- Model key: `{key}`")
        lines.append(f"- Model filename: `{model_file}`")
        lines.append(f"- llama-cli: `{llama_cli}`")

        if model_path is None:
            lines.append("- Status: missing model file")
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            index_lines.append(f"- {key}: missing-model-file")
            continue

        lines.append(f"- Model path: `{model_path}`")

        cmd = [
            "python3", str(script),
            "--repo-root", str(repo_root),
            "--llama-cli", str(llama_cli),
            "--model", str(model_path),
            "--prompt", args.prompt,
            "--n-predict", str(args.n_predict),
            "--threads", str(args.threads),
            "--cgraph-file", str(cgraph_path),
            "--keep-going-on-inference-error",
        ]

        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                timeout=args.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
            stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
            proc = subprocess.CompletedProcess(cmd, 124, stdout=stdout, stderr=stderr)

        lines.append(f"- Exit code: `{proc.returncode}`")
        lines.append("")

        if proc.stdout.strip():
            lines.append("## Output")
            lines.append("```text")
            lines.append(proc.stdout.rstrip())
            lines.append("```")

        if proc.stderr.strip():
            lines.append("## stderr")
            lines.append("```text")
            lines.append(proc.stderr.rstrip())
            lines.append("```")

        if timed_out:
            status = "timeout"
            lines.append("")
            lines.append(f"- Timeout: exceeded {args.timeout_sec}s")
        else:
            status = "ok" if proc.returncode == 0 else "error"
        lines.append("")
        lines.append(f"- Status: {status}")

        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        index_lines.append(f"- {key}: {status}")

    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"Reports written to: {out_dir}")
    for line in index_lines[2:]:
        print(line.replace("- ", ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
