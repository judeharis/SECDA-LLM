#!/usr/bin/env python3
"""SOFT_MAX test-file generator for test-backend-ops --test-file / --test-params.

Two modes:
  Suite mode (default): reads a JSON config and writes all permutations.
      python generate_soft_max_suite.py [--config PATH] [--output PATH] [--ggml-h PATH]

  Single mode: emits one test line to stdout.
      python generate_soft_max_suite.py single --ne0 512 --ne1 1 --ne2 32 --mask --scale 0.125

The emitted format is test-backend-ops's generic-op test-file/--test-params line
(see make_test_cases_from_string / test_generic_op in tests/test-backend-ops.cpp):
falls through to test_generic_op for any op other than MUL_MAT, so the encoding
here just needs to match that struct's field order, not a SOFT_MAX-specific one.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (from llama.cpp/ggml/include/ggml.h)
# ---------------------------------------------------------------------------
GGML_OP_SOFT_MAX = 46
GGML_MAX_OP_PARAMS_I32 = 16

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GGML_H = SCRIPT_DIR.parent.parent / "llama.cpp" / "ggml" / "include" / "ggml.h"
DEFAULT_CONFIG = SCRIPT_DIR / "soft_max_suite_config.json"

# Reuse the same ggml_type enum loader as the MUL_MAT generator so both stay in
# sync with a single source of truth (ggml.h) instead of hardcoding type ids.
import importlib.util

_MUL_MAT_SUITE = SCRIPT_DIR / "generate_mul_mat_suite.py"
_spec = importlib.util.spec_from_file_location("generate_mul_mat_suite", _MUL_MAT_SUITE)
_mul_mat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mul_mat_mod)
load_type_map_from_header = _mul_mat_mod.load_type_map_from_header
parse_type = _mul_mat_mod.parse_type


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def float_to_i32_bits(value: float) -> int:
    """Reinterpret a float's bit pattern as a signed int32 (matches how
    ggml_set_op_params/memcpy packs op_params, which test-backend-ops's
    make_test_cases_from_string then reads back as plain int32 decimals)."""
    return struct.unpack("<i", struct.pack("<f", value))[0]


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


def build_soft_max_line(
    f32_type_id: int,
    ne: Tuple[int, int, int, int],
    mask: bool,
    mask_type_id: Optional[int],
    sinks: bool,
    scale: float,
    max_bias: float,
    name: str,
) -> str:
    if min(ne) <= 0:
        raise ValueError("ne0..ne3 must all be positive integers.")
    if mask and mask_type_id is None:
        raise ValueError("mask=True requires a mask_type_id.")
    if max_bias > 0.0 and not mask:
        # Matches ggml_soft_max_impl's own assert (GGML/ggml.c): ALiBi bias
        # requires a mask to apply the per-position slope against.
        raise ValueError("max_bias > 0 requires mask=True.")

    # test-backend-ops treats nb[0] == 0 as "use default contiguous strides".
    contiguous_marker_nb = (0, 0, 0, 0)

    # src[0]: logits (F32). softmax's dst has the same shape/type as src[0].
    sources = [SourceTensor(type_id=f32_type_id, ne=ne, nb=contiguous_marker_nb)]

    # src[1]: mask (F32 or F16), same ne as logits (no broadcast — nr23={1,1}).
    if mask:
        sources.append(SourceTensor(type_id=mask_type_id, ne=ne, nb=contiguous_marker_nb))
    elif sinks:
        # ggml_soft_max_add_sinks writes to src[2] directly, so a placeholder
        # mask slot isn't needed to reach it — sinks-without-mask is valid.
        pass

    op_params = [0] * GGML_MAX_OP_PARAMS_I32
    op_params[0] = float_to_i32_bits(scale)
    op_params[1] = float_to_i32_bits(max_bias)

    fields: List[str] = [
        str(GGML_OP_SOFT_MAX),
        str(f32_type_id),  # dst type is always F32
        *(str(x) for x in ne),
        str(len(op_params)),
        *(str(x) for x in op_params),
        str(len(sources) + (1 if sinks else 0)),
    ]
    for src in sources:
        fields.extend(serialize_source(src))
    if sinks:
        # src[2]: attention sinks, 1D, one value per head (ne[2], nr23={1,1}).
        sinks_src = SourceTensor(type_id=f32_type_id, ne=(ne[2], 1, 1, 1), nb=contiguous_marker_nb)
        fields.extend(serialize_source(sinks_src))
    fields.append(name if name else "-")

    return " ".join(fields)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    for key in ("ne0", "ne1", "ne2", "ne3", "scale", "max_bias"):
        if key not in cfg:
            raise ValueError(f"Config is missing required key: '{key}'")
        if not isinstance(cfg[key], list) or len(cfg[key]) == 0:
            raise ValueError(f"Config key '{key}' must be a non-empty list")

    if "mask_variants" not in cfg or not cfg["mask_variants"]:
        raise ValueError("Config must have a non-empty 'mask_variants' list")
    if "sinks" not in cfg or not cfg["sinks"]:
        raise ValueError("Config must have a non-empty 'sinks' list")

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
        raw_output = cfg.get("output", "soft_max_suite_generated.txt")
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = config_path.parent / output_path

    f32_id = parse_type("f32", type_map)

    lines = []
    idx = 0
    skipped = 0
    for ne0, ne1, ne2, ne3, mask_variant, sinks, scale, max_bias in product(
        cfg["ne0"], cfg["ne1"], cfg["ne2"], cfg["ne3"],
        cfg["mask_variants"], cfg["sinks"], cfg["scale"], cfg["max_bias"],
    ):
        idx += 1
        mask = bool(mask_variant.get("mask", False))
        mask_type_id = None
        mask_tag = "noMask"
        if mask:
            mask_type_id = parse_type(str(mask_variant["mask_type"]), type_map)
            mask_tag = f"Mask{str(mask_variant['mask_type']).upper()}"

        name = (
            f"SOFT_MAX_suite_{idx:04d}_{ne0}x{ne1}x{ne2}x{ne3}_{mask_tag}"
            f"{'_Sinks' if sinks else ''}_s{scale}_b{max_bias}"
        )

        try:
            line = build_soft_max_line(
                f32_type_id=f32_id,
                ne=(int(ne0), int(ne1), int(ne2), int(ne3)),
                mask=mask,
                mask_type_id=mask_type_id,
                sinks=bool(sinks),
                scale=float(scale),
                max_bias=float(max_bias),
                name=name,
            )
        except ValueError as exc:
            skipped += 1
            print(f"Skipping invalid combo {name}: {exc}")
            continue
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} test lines to {output_path} ({skipped} invalid combos skipped)")
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    ggml_h = Path(args.ggml_h) if args.ggml_h else DEFAULT_GGML_H
    type_map = load_type_map_from_header(ggml_h)

    f32_id = parse_type("f32", type_map)
    mask_type_id = parse_type(args.mask_type, type_map) if args.mask else None

    line = build_soft_max_line(
        f32_type_id=f32_id,
        ne=(args.ne0, args.ne1, args.ne2, args.ne3),
        mask=args.mask,
        mask_type_id=mask_type_id,
        sinks=args.sinks,
        scale=args.scale,
        max_bias=args.max_bias,
        name=args.name,
    )
    print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SOFT_MAX test-file generator for test-backend-ops --test-file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ggml-h",
        default=None,
        help="Path to ggml.h (default: auto-resolved relative to this script)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- suite sub-command ----------------------------------------------------
    suite_p = subparsers.add_parser(
        "suite",
        help="Generate all permutations from a JSON config (default mode)",
    )
    suite_p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to JSON config file (default: soft_max_suite_config.json alongside this script)",
    )
    suite_p.add_argument(
        "--output",
        default="./configs/synth_configs/soft_max_synth_configs.txt",
        help="Path to write test lines (overrides 'output' key in config)",
    )

    # -- single sub-command ----------------------------------------------------
    single_p = subparsers.add_parser(
        "single",
        help="Emit one test line to stdout",
    )
    single_p.add_argument("--ne0", required=True, type=int, help="Row width (context length / n_kv)")
    single_p.add_argument("--ne1", required=True, type=int, help="Rows per (head,batch) (n_tokens)")
    single_p.add_argument("--ne2", required=True, type=int, help="Heads (n_head)")
    single_p.add_argument("--ne3", default=1, type=int, help="Batch (default 1)")
    single_p.add_argument("--mask", action="store_true", help="Include a KQ mask input")
    single_p.add_argument("--mask-type", default="f32", help="Mask type (f32 or f16), only used with --mask")
    single_p.add_argument("--sinks", action="store_true", help="Include attention sinks input")
    single_p.add_argument("--scale", required=True, type=float, help="Softmax pre-scale (e.g. 1/sqrt(head_dim))")
    single_p.add_argument("--max-bias", required=True, type=float, help="ALiBi max_bias (0.0 = disabled)")
    single_p.add_argument("--name", default="SOFT_MAX_generated", help="Name token appended to the test line")

    args = parser.parse_args()

    if args.command is None or args.command == "suite":
        if args.command is None:
            args = suite_p.parse_args([], namespace=args)
        return cmd_suite(args)

    return cmd_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
