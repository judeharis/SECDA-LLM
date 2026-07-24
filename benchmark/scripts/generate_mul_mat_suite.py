#!/usr/bin/env python3
"""MUL_MAT test-file generator for test-backend-ops --test-file.

Two modes:
  Suite mode (default): reads a JSON config and writes all permutations.
      python generate_mul_mat_suite.py [--config PATH] [--output PATH] [--ggml-h PATH]

  Single mode: emits one test line to stdout (mirrors the old generate_mul_mat_test.py CLI).
      python generate_mul_mat_suite.py single --type-a Q4_K --type-b f32 --m 128 --n 4 --k 256

The emitted format matches llama.cpp/tests/export-graph-ops.cpp serialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants (from llama.cpp/ggml/include/ggml.h)
# ---------------------------------------------------------------------------
GGML_OP_MUL_MAT = 29
GGML_MAX_OP_PARAMS_I32 = 16

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GGML_H = SCRIPT_DIR.parent.parent / "llama.cpp" / "ggml" / "include" / "ggml.h"
DEFAULT_CONFIG = SCRIPT_DIR / "mul_mat_suite_config.json"


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def load_type_map_from_header(header_path: Path) -> Dict[str, int]:
    """Load ggml_type names from enum ggml_type in ggml.h.

    Returned map supports both short names (f32, q4_0, ...) and
    full enum names (GGML_TYPE_F32, ...), both case-insensitive.
    """
    content = header_path.read_text(encoding="utf-8")

    enum_match = re.search(r"enum\s+ggml_type\s*\{(.*?)\};", content, re.S)
    if enum_match is None:
        raise ValueError(f"Unable to find enum ggml_type in {header_path}")

    enum_block = enum_match.group(1)
    type_map: Dict[str, int] = {}
    current_value = -1

    for raw_line in enum_block.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue

        m = re.match(r"(GGML_TYPE_[A-Z0-9_]+)\s*(=\s*([0-9]+))?\s*,?", line)
        if m is None:
            continue

        enum_name = m.group(1)
        explicit = m.group(3)
        if explicit is not None:
            current_value = int(explicit)
        else:
            current_value += 1

        if enum_name == "GGML_TYPE_COUNT":
            continue

        type_map[enum_name.lower()] = current_value
        short_name = enum_name.removeprefix("GGML_TYPE_").lower()
        type_map[short_name] = current_value

    if not type_map:
        raise ValueError(f"No ggml types parsed from {header_path}")

    return type_map


def parse_type(type_value: str, type_map: Dict[str, int]) -> int:
    key = type_value.strip().lower()
    if key in type_map:
        return type_map[key]

    try:
        value = int(key)
    except ValueError as exc:
        known_short = sorted(k for k in type_map.keys() if not k.startswith("ggml_type_"))
        preview = ", ".join(known_short[:12])
        if len(known_short) > 12:
            preview += ", ..."
        raise argparse.ArgumentTypeError(
            f"Unsupported type '{type_value}'. Use numeric ggml_type id or one of: {preview}"
        ) from exc

    max_id = max(type_map.values())
    if value < 0 or value > max_id:
        raise argparse.ArgumentTypeError(f"Type id {value} is out of range [0, {max_id}].")

    return value


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

@dataclass
class SourceTensor:
    type_id: int
    ne: Tuple[int, int, int, int]
    nb: Tuple[int, int, int, int]


def serialize_source(src: SourceTensor) -> List[str]:
    values: List[str] = [str(src.type_id)]
    values.extend(str(x) for x in src.ne)
    values.extend(str(x) for x in src.nb)
    return values


def build_mul_mat_line(type_a: int, type_b: int, m: int, n: int, k: int, name: str) -> str:
    if min(m, n, k) <= 0:
        raise ValueError("m, n, and k must all be positive integers.")

    # ggml_mul_mat(a, b)
    # a: [k, m, 1, 1]
    # b: [k, n, 1, 1]
    # out: [m, n, 1, 1]
    a_ne = (k, m, 1, 1)
    b_ne = (k, n, 1, 1)
    out_ne = (m, n, 1, 1)

    # test-backend-ops treats nb[0] == 0 as "use default contiguous strides".
    # This avoids hardcoding block-size layouts for every quantized type.
    contiguous_marker_nb = (0, 0, 0, 0)
    a_src = SourceTensor(type_id=type_a, ne=a_ne, nb=contiguous_marker_nb)
    b_src = SourceTensor(type_id=type_b, ne=b_ne, nb=contiguous_marker_nb)

    op_params = [0] * GGML_MAX_OP_PARAMS_I32

    fields: List[str] = [
        str(GGML_OP_MUL_MAT),
        str(0),  # output tensor type for MUL_MAT is F32
        *(str(x) for x in out_ne),
        str(len(op_params)),
        *(str(x) for x in op_params),
        "2",  # number of source tensors
        *serialize_source(a_src),
        *serialize_source(b_src),
        name if name else "-",
    ]

    return " ".join(fields)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    for key in ("type_a", "type_b", "m", "n", "k"):
        if key not in cfg:
            raise ValueError(f"Config is missing required key: '{key}'")
        if not isinstance(cfg[key], list) or len(cfg[key]) == 0:
            raise ValueError(f"Config key '{key}' must be a non-empty list")

    return cfg


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def cmd_suite(args: argparse.Namespace) -> int:
    ggml_h = Path(args.ggml_h) if args.ggml_h else DEFAULT_GGML_H
    type_map = load_type_map_from_header(ggml_h)
    print(f"Loaded {len(type_map)} type aliases from {ggml_h}")

    config_path = Path(args.config)
    cfg = load_config(config_path)
    print(f"Loaded config from {config_path}")

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        raw_output = cfg.get("output", "mul_mat_suite_generated.txt")
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = config_path.parent / output_path

    lines = []
    for idx, (type_a, type_b, m, n, k) in enumerate(
        product(
            cfg["type_a"],
            cfg["type_b"],
            cfg["m"],
            cfg["n"],
            cfg["k"],
        ),
        start=1,
    ):
        type_a_id = parse_type(str(type_a), type_map)
        type_b_id = parse_type(str(type_b), type_map)
        name = f"MUL_MAT_suite_{idx:04d}"

        line = build_mul_mat_line(
            type_a=type_a_id,
            type_b=type_b_id,
            m=int(m),
            n=int(n),
            k=int(k),
            name=name,
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} test lines to {output_path}")
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    ggml_h = Path(args.ggml_h) if args.ggml_h else DEFAULT_GGML_H
    type_map = load_type_map_from_header(ggml_h)

    type_a_id = parse_type(args.type_a, type_map)
    type_b_id = parse_type(args.type_b, type_map)

    line = build_mul_mat_line(
        type_a=type_a_id,
        type_b=type_b_id,
        m=args.m,
        n=args.n,
        k=args.k,
        name=args.name,
    )
    print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MUL_MAT test-file generator for test-backend-ops --test-file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ggml-h",
        default=None,
        help="Path to ggml.h (default: auto-resolved relative to this script)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- suite sub-command (default) -----------------------------------------
    suite_p = subparsers.add_parser(
        "suite",
        help="Generate all permutations from a JSON config (default mode)",
    )
    suite_p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to JSON config file (default: mul_mat_suite_config.json alongside this script)",
    )
    suite_p.add_argument(
        "--output",
        default="./configs/synth_configs.txt",
        help="Path to write test lines (overrides 'output' key in config)",
    )

    # -- single sub-command --------------------------------------------------
    single_p = subparsers.add_parser(
        "single",
        help="Emit one test line to stdout",
    )
    single_p.add_argument(
        "--type-a",
        required=True,
        help="Type of source tensor A (short name, full enum name, or numeric id)",
    )
    single_p.add_argument(
        "--type-b",
        required=True,
        help="Type of source tensor B (short name, full enum name, or numeric id)",
    )
    single_p.add_argument("--m", required=True, type=int, help="Rows of output (and rows of A)")
    single_p.add_argument("--n", required=True, type=int, help="Cols of output (and cols of B)")
    single_p.add_argument("--k", required=True, type=int, help="Inner dimension (cols of A, rows of B)")
    single_p.add_argument("--name", default="MUL_MAT_generated", help="Name token appended to the test line")

    args = parser.parse_args()

    # Default to suite mode when no sub-command is given
    if args.command is None or args.command == "suite":
        if args.command is None:
            # Re-parse as suite so suite defaults apply
            args = suite_p.parse_args([], namespace=args)
        return cmd_suite(args)

    return cmd_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
