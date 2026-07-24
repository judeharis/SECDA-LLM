import os
import sys
import re

import pandas as pd
from pathlib import Path


sys.dont_write_bytecode = True
# get_all files from a directory and process them

# --- DMA parsing helpers (inlined so parse_results does not depend on dma_profiles.csv) ---
DMA_RE = re.compile(r"^-+DMA:\s*(\d+)-+", re.IGNORECASE)
SEP_RE = re.compile(r"^[=\-]{3,}$")


def normalize_key(k: str) -> str:
    k = k.strip().lower()
    k = k.replace(' ', '_')
    k = k.replace('/', '_')
    k = k.replace('(', '')
    k = k.replace(')', '')
    k = k.replace('-', '_')
    k = k.replace('%', 'pct')
    return k


def try_parse_value(k: str, v: str):
    extra = {}
    s = v.strip()
    m = re.match(r"^([0-9,]+)\s*bytes$", s, re.IGNORECASE)
    if m:
        num = int(m.group(1).replace(',', ''))
        extra[f"{k}_bytes"] = num
        return v, extra
    m = re.match(r"^([0-9.,+-eE]+)\s*([KMGTP]?B)/s$", s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        mul = {'KB': 1 / 1024, 'MB': 1.0, 'GB': 1024.0, 'TB': 1024.0 * 1024}
        mulv = mul.get(unit, 1.0)
        extra[f"{k}_mb_s"] = val * mulv
        return v, extra
    m = re.match(r"^([0-9,]+)$", s)
    if m:
        num = int(m.group(1).replace(',', ''))
        extra[f"{k}_int"] = num
        return v, extra
    return v, extra


def parse_dma_block(lines, start_idx):
    m = DMA_RE.match(lines[start_idx])
    dma_id = m.group(1) if m else None
    values = {}
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].rstrip('\n')
        if DMA_RE.match(line) or SEP_RE.match(line):
            break
        if line.strip() == '':
            i += 1
            continue
        if ':' in line:
            k, v = line.split(':', 1)
            k = normalize_key(k)
            v = v.strip()
            values[k] = v
            _, extra = try_parse_value(k, v)
            for ek, ev in extra.items():
                if ek not in values:
                    values[ek] = ev
        i += 1

    values['dma_id'] = dma_id
    # if we stopped because of a separator, try to capture summary metrics between separators
    j = i
    if j < len(lines) and SEP_RE.match(lines[j]):
        while j < len(lines) and SEP_RE.match(lines[j]):
            j += 1
        kidx = j
        while kidx < len(lines) and not SEP_RE.match(lines[kidx]) and not DMA_RE.match(lines[kidx]):
            l = lines[kidx].strip()
            if l == '':
                kidx += 1
                continue
            if ':' in l:
                sk, sv = l.split(':', 1)
                skn = 'summary_' + normalize_key(sk)
                sv = sv.strip()
                values[skn] = sv
                _, extra = try_parse_value(skn, sv)
                for ek, ev in extra.items():
                    if ek not in values:
                        values[ek] = ev
            kidx += 1
        i = kidx

    return values, i


def parse_dma_from_file(path):
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except Exception:
        return []
    lines = text.splitlines()
    idx = 0
    rows = []
    while idx < len(lines):
        line = lines[idx]
        if DMA_RE.match(line):
            vals, nxt = parse_dma_block(lines, idx)
            # include filename for matching
            vals['file_name'] = os.path.basename(path)
            rows.append(vals)
            idx = nxt
        else:
            idx += 1
    return rows

# --- end DMA helpers ---

RESULT_SUFFIXES = (
    "_llama_perf.csv",
    "_prf.csv",
    "_bench_power.txt",
    "_power.txt",
    "_llama-bench.csv",
    "_vtbo.csv",
    "_tbo.csv",
    "_results.txt",
    ".txt",
    ".csv",
)

TOOL_ORDER = ("bench", "cli", "synth")
LLAMA_DMA_METRICS = [
    "data_transfered_bytes",
    "data_transfered_recv_bytes",
    "data_per_send_bytes",
    "data_per_recv_bytes",
    "data_send_count_int",
    "data_recv_count_int",
    "send_speed_mb_s",
    "recv_speed_mb_s",
    "send_wait_int",
    "recv_wait_int",
    "summary_layer_total_int",
]
MEAN_DMA_METRICS = {"send_speed_mb_s", "recv_speed_mb_s", "send_wait_int", "recv_wait_int"}


def read_csv_frame(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    frame = frame.loc[:, ~pd.Index(frame.columns).str.match(r"^Unnamed")]
    return frame


def first_existing_path(directory, candidates):
    for candidate in candidates:
        candidate_path = os.path.join(directory, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def strip_known_suffix(filename):
    for suffix in RESULT_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def collect_run_groups(directory):
    groups = {}
    for filename in sorted(os.listdir(directory)):
        if filename == "results.csv" or filename.endswith("_results.csv"):
            continue
        root = strip_known_suffix(filename)
        if not root:
            continue
        tool = root.split("_", 1)[0]
        if tool not in TOOL_ORDER:
            continue
        groups.setdefault(tool, {}).setdefault(root, []).append(filename)
    return groups


def numeric_scalar(value, default=0.0):
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def numeric_series(frame, column, default=0.0):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(frame), index=frame.index, dtype="float64")


def to_bool_scalar(value):
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "0", "0.0", "no", "n", "fail", "failed", "", "none", "nan"}:
        return False

    try:
        return float(text) != 0.0
    except Exception:
        return False


def parse_llama_metadata(root):
    tool, _, tail = root.partition("_")
    metadata = {
        "tool": tool,
        "run_name": root,
        "run_group": tail,
        "model": tail,
        "threads": "",
        "board": "",
        "version": "",
        "opt": "",
        "hw": "",
    }

    parts = tail.split("_") if tail else []
    if len(parts) >= 5:
        metadata["opt"] = parts[-1]
        metadata["version"] = parts[-2]
        metadata["board"] = parts[-3]
        metadata["hw"] = parts[-4]
        metadata["threads"] = parts[-5]
        metadata["model"] = "_".join(parts[:-5])

    return metadata


def parse_synth_shape(root):
    # Expected pattern: synth_Q2_128M_4N_256K_<hardware>
    match = re.match(r"^synth_(Q\d+)_([0-9]+)M_([0-9]+)N_([0-9]+)K(?:_(.+))?$", root)
    if not match:
        return {
            "q_type": "",
            "m": 0,
            "n": 0,
            "k": 0,
            "hardware": "",
        }

    return {
        "q_type": match.group(1),
        "m": int(match.group(2)),
        "n": int(match.group(3)),
        "k": int(match.group(4)),
        # Keep the full hardware tag exactly as encoded in run_name.
        "hardware": (match.group(5) or ""),
    }


def read_avg_power(directory, root):
    power_path = first_existing_path(
        directory,
        [
            f"{root}_power.txt",
            f"{root}_bench_power.txt",
        ],
    )
    if not power_path:
        return 0.0

    try:
        power = pd.read_csv(power_path, header=None)
        power = power.dropna()
        if power.empty or len(power.index) <= 1:
            return 0.0
        values = pd.to_numeric(power.iloc[1:, 0], errors="coerce").dropna()
        if values.empty:
            return 0.0
        return float(values.mean())
    except Exception:
        return 0.0


def parse_dma_metrics(directory, root):
    parsed_rows = []
    for filename in os.listdir(directory):
        if not filename.startswith(root):
            continue
        if not filename.endswith(".txt"):
            continue
        if filename.endswith("_power.txt") or filename.endswith("_bench_power.txt"):
            continue
        parsed_rows.extend(parse_dma_from_file(os.path.join(directory, filename)))

    dma_columns = {}
    if not parsed_rows:
        for did in range(4):
            for metric in LLAMA_DMA_METRICS:
                dma_columns[f"dma{did}_{metric}"] = 0
        return dma_columns

    dma_sel = pd.DataFrame(parsed_rows)
    if "dma_id" in dma_sel.columns:
        dma_sel["dma_id"] = dma_sel["dma_id"].astype(str)
    else:
        dma_sel["dma_id"] = ""

    for did in range(4):
        df_d = dma_sel[dma_sel["dma_id"] == str(did)]
        for metric in LLAMA_DMA_METRICS:
            key = f"dma{did}_{metric}"
            if df_d.empty or metric not in df_d.columns:
                dma_columns[key] = 0
                continue
            values = pd.to_numeric(df_d[metric], errors="coerce")
            if metric in MEAN_DMA_METRICS:
                dma_columns[key] = float(values.dropna().mean()) if not values.dropna().empty else 0
            else:
                dma_columns[key] = float(values.fillna(0).sum())

    return dma_columns


def parse_llama_group(directory, root):
    perf_path = first_existing_path(
        directory,
        [
            f"{root}_llama_perf.csv",
            f"{root}_llama-bench.csv",
        ],
    )
    if not perf_path:
        return None

    perf = read_csv_frame(perf_path)
    if perf.empty:
        return None

    row = perf.iloc[0].to_dict()
    row.update(parse_llama_metadata(root))
    row["source_perf_file"] = os.path.basename(perf_path)
    row["avg_power"] = read_avg_power(directory, root)
    row["mm layers"] = 0

    ## Disabling mm layers for now until we get a more reliable way to compute it from the graph stats. 
    # prf_path = first_existing_path(directory, [f"{root}_prf.csv"])
    # if prf_path:
    #     prf = read_csv_frame(prf_path)
    #     if not prf.empty:
    #         if "layer_total" in prf.columns:
    #             row["mm layers"] = numeric_scalar(prf.iloc[0]["layer_total"], 0)
    #         for column in prf.columns:
    #             if column == "layer_total":
    #                 continue
    #             row[f"prf_{column}"] = prf.iloc[0][column]

    row.update(parse_dma_metrics(directory, root))
    frame = pd.DataFrame([row])
    frame["token/s"] = (
        numeric_series(frame, "eval tokens") + numeric_series(frame, "prompt tokens")
    ) / ((numeric_series(frame, "eval time") + numeric_series(frame, "prompt eval time")) / 1000)
    frame["actual_total_time"] = numeric_series(frame, "prompt eval time") + numeric_series(frame, "eval time")
    frame["mm layers"] = numeric_series(frame, "mm layers")
    frame["other_layers"] = frame["actual_total_time"] - (frame["mm layers"] / 1000)
    frame["mm_total"] = frame["actual_total_time"] - frame["other_layers"]
    frame["Joules"] = (numeric_series(frame, "avg_power") / 1000000) * (frame["actual_total_time"] / 1000)
    frame = frame.rename(columns={"hw": "Hardware", "model": "Model", "total time": "e2e_time", "total_time": "e2e_time"})
    frame["Total Time (s)"] = frame["actual_total_time"] / 1000
    frame["Matmul Time (s)"] = frame["mm_total"] / 1000
    return frame


def parse_synth_group(directory, root):
    tbo_path = first_existing_path(directory, [f"{root}_tbo.csv"])
    if not tbo_path:
        return None

    tbo = read_csv_frame(tbo_path)
    if tbo.empty:
        return None

    tbo["tool"] = "synth"
    tbo["run_name"] = root
    shape = parse_synth_shape(root)
    tbo["q_type"] = shape["q_type"]
    tbo["m"] = shape["m"]
    tbo["n"] = shape["n"]
    tbo["k"] = shape["k"]
    tbo["hardware"] = shape["hardware"]
    tbo["source_tbo_file"] = os.path.basename(tbo_path)
    tbo["avg_power"] = read_avg_power(directory, root)
    if "time_us" in tbo.columns:
        time_us = pd.to_numeric(tbo["time_us"], errors="coerce")
        n_runs = pd.to_numeric(tbo.get("n_runs", 1), errors="coerce").replace(0, pd.NA)
        tbo["latency_ns"] = (time_us / n_runs).fillna(0) * 1000
        tbo["total_time (ms)"] = time_us / 1000
        avg_power_uw = pd.to_numeric(tbo["avg_power"], errors="coerce").fillna(0)
        tbo["micro_joules_per_run"] = avg_power_uw * (tbo["latency_ns"] / 1000000000)
    else:
        tbo["latency_ns"] = 0
        tbo["total_time (ms)"] = 0
        tbo["micro_joules_per_run"] = 0
    tbo["valid"] = False

    vtbo_path = first_existing_path(directory, [f"{root}_vtbo.csv"])
    if vtbo_path:
        vtbo = read_csv_frame(vtbo_path)
        if not vtbo.empty and "passed" in vtbo.columns:
            valid = vtbo["passed"].reset_index(drop=True).apply(to_bool_scalar)
            if len(valid) == len(tbo.index):
                tbo["valid"] = valid.values
            elif len(valid) > 0:
                tbo["valid"] = bool(valid.iloc[0])

    if "error" in tbo.columns:
        error_text = tbo["error"].astype(str).str.strip()
        tbo["error"] = error_text.mask(
            error_text.str.lower().isin({"", "0", "0.0", "none", "nan"}),
            "n/a",
        )
    else:
        tbo["error"] = "n/a"

    prf_path = first_existing_path(directory, [f"{root}_prf.csv"])
    if prf_path:
        prf = read_csv_frame(prf_path)
        if not prf.empty:
            for column in prf.columns:
                value = prf.iloc[0][column]
                if column in tbo.columns:
                    tbo[f"prf_{column}"] = value
                else:
                    tbo[column] = value

    dma_metrics = parse_dma_metrics(directory, root)
    for key, value in dma_metrics.items():
        tbo[key] = value
    return tbo


def sort_llama_frame(frame):
    if frame.empty:
        return frame

    sort_columns = [column for column in ["Model", "Hardware", "opt"] if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(by=sort_columns)

    ordered_columns = []
    preferred = [
        "tool",
        "run_name",
        "Model",
        "Hardware",
        "Total Time (s)",
        "Matmul Time (s)",
        "token/s",
        "Joules",
        "other_layers",
        "total_time (ms)",
        "actual_total_time",
        "mm_total",
        "avg_power",
        "threads",
        "board",
        "version",
        "opt",
    ]
    for column in preferred:
        if column in frame.columns:
            ordered_columns.append(column)

    for column in frame.columns:
        if column not in ordered_columns and not re.match(r"^dma[0-3]_", column):
            ordered_columns.append(column)

    dma_columns = sorted([column for column in frame.columns if re.match(r"^dma[0-3]_", column)])
    ordered_columns.extend([column for column in dma_columns if column not in ordered_columns])

    return frame[ordered_columns]


def sort_synth_frame(frame):
    if frame.empty:
        return frame

    if "run_name" in frame.columns:
        frame = frame.sort_values(by=["run_name"])

    ordered_columns = []
    preferred = [
        "tool",
        "run_name",
        "q_type",
        "m",
        "n",
        "k",
        "hardware",
        "test_time",
        "backend_name",
        "op_name",
        "test_mode",
        "supported",
        "passed",
        "time_us",
        "latency_ns",
        "total_time (ms)",
        "avg_power",
        "micro_joules_per_run",
        "layer_total",
    ]
    for column in preferred:
        if column in frame.columns:
            ordered_columns.append(column)

    for column in frame.columns:
        if column not in ordered_columns and not re.match(r"^dma[0-3]_", column):
            ordered_columns.append(column)

    dma_columns = sorted([column for column in frame.columns if re.match(r"^dma[0-3]_", column)])
    ordered_columns.extend([column for column in dma_columns if column not in ordered_columns])

    return frame[ordered_columns]


def write_frame(directory, name, frame):
    if frame is None or frame.empty:
        return None

    output_path = os.path.join(directory, name)
    frame = frame.fillna(0)
    frame.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saving results to csv: {output_path}")
    return output_path


def main():
    directory = sys.argv[1]
    print(directory)

    groups = collect_run_groups(directory)
    print(groups)

    tool_frames = {}
    for tool in TOOL_ORDER:
        tool_groups = groups.get(tool, {})
        tool_rows = []
        for root in sorted(tool_groups):
            if tool in {"bench", "cli"}:
                frame = parse_llama_group(directory, root)
                if frame is not None:
                    tool_rows.append(frame)
            elif tool == "synth":
                frame = parse_synth_group(directory, root)
                if frame is not None:
                    tool_rows.append(frame)

        if not tool_rows:
            continue

        combined = pd.concat(tool_rows, ignore_index=True, sort=False)
        if tool in {"bench", "cli"}:
            combined = sort_llama_frame(combined)
        else:
            combined = sort_synth_frame(combined)
        tool_frames[tool] = combined
        write_frame(directory, f"{tool}_results.csv", combined)

    if tool_frames:
        combined_all = pd.concat(tool_frames.values(), ignore_index=True, sort=False)
        if "tool" in combined_all.columns:
            combined_all = combined_all.sort_values(by=["tool"] + (["run_name"] if "run_name" in combined_all.columns else []))
        write_frame(directory, "results.csv", combined_all)


if __name__ == "__main__":
    main()
