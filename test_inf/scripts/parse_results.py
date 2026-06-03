import os
import sys
import re

import pandas as pd
from pathlib import Path

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


def get_all_files(directory):
    files = []
    # get csv files
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            # skip aggregated or helper CSVs
            if "results" in file:
                continue
            if file.startswith('dma_profiles'):
                continue
            # remove "_llama_perf.csv" from file name
            file = re.sub("_llama_perf.csv", "", file)
            # remove "_prf.csv" from file name
            file = re.sub("_prf.csv", "", file)
            # append if not already in list
            if file not in files:
                files.append(file)
    return files


def process_file(file, run, df):
    # add first row to data frame open file name  + _llama_perf.csv
    df1 = pd.read_csv(file + "_llama_perf.csv")
    # remove leading space from column names
    df1.columns = df1.columns.str.lstrip()
    # keep original run string for filename matching, then split into parts
    run_str = run
    # from run  name split with _
    run = run.split("_")
    opt = run[-1]
    version = run[-2]
    board = run[-3]
    hw = run[-4]
    threads = run[-5]
    model = run[:-5]
    model = "_".join(model)
    run_tag = model + "_" + threads + "_" + hw + "_" + board + "_" + version + "_" + opt
    # add run tag to data frame
    df1["runtag"] = run_tag
    df1["model"] = model
    df1["threads"] = threads
    df1["board"] = board
    df1["version"] = version
    df1["opt"] = opt
    df1["hw"] = hw
    # find total time by adding "prompt eval time" and "eval time"

    # process power file
    # the file is in format where you can ignore the first row and then every row as a single value, read until empty row
    power = pd.read_csv(file + "_power.txt", header=None)
    power = power.dropna()
    power = power.iloc[1:]  # drop the first row
    power = power[0].astype(float)
    avg_power = power.mean()
    df1["avg_power"] = avg_power

    # --- DMA metrics: look for dma_profiles.csv in the same directory and aggregate metrics for this run ---
    # parse DMA directly from matching .txt files in the same directory as the perf files
    try:
        dirpath = os.path.dirname(file)
        # collect parsed DMA rows from files whose basename starts with run
        parsed_rows = []
        for fname in os.listdir(dirpath):
            if not fname.endswith('.txt'):
                continue
            # match files that start with the run tag (use original run string)
            if not fname.startswith(run_str):
                continue
            fullp = os.path.join(dirpath, fname)
            parsed_rows.extend(parse_dma_from_file(fullp))

        if parsed_rows:
            dma_sel = pd.DataFrame(parsed_rows)

            def col_sum(df, name):
                if name in df.columns:
                    return pd.to_numeric(df[name], errors='coerce').fillna(0).sum()
                return 0

            def col_mean(df, name):
                if name in df.columns:
                    return pd.to_numeric(df[name], errors='coerce').dropna().mean()
                return 0

            # For each DMA id (assume up to 4: 0..3) compute metrics per-dma and attach as separate columns
            per_dma_metrics = {}
            dma_ids = [str(x) for x in range(4)]
            metric_names = [
                'data_transfered_bytes',
                'data_transfered_recv_bytes',
                'data_per_send_bytes',
                'data_per_recv_bytes',
                'data_send_count_int',
                'data_recv_count_int',
                'send_speed_mb_s',
                'recv_speed_mb_s',
                'send_wait_int',
                'recv_wait_int',
                'summary_layer_total_int',
            ]
            for did in dma_ids:
                df_d = dma_sel[dma_sel.get('dma_id', '') == did]
                for m in metric_names:
                    key = f'dma{did}_{m}'
                    if m in ['send_speed_mb_s', 'recv_speed_mb_s', 'send_wait_int', 'recv_wait_int']:
                        # mean-type metrics
                        per_dma_metrics[key] = col_mean(df_d, m) if not df_d.empty else 0
                    else:
                        per_dma_metrics[key] = col_sum(df_d, m) if not df_d.empty else 0

            # aggregated totals are intentionally omitted; per-DMA metrics (dma0_.. dma3_..) are provided below
        else:
            dma_data_sent = dma_data_recv = dma_send_count = dma_recv_count = 0
            dma_send_speed = dma_recv_speed = dma_send_wait = dma_recv_wait = dma_layer_total = 0
    except Exception:
        dma_data_sent = dma_data_recv = dma_send_count = dma_recv_count = 0
        dma_send_speed = dma_recv_speed = dma_send_wait = dma_recv_wait = dma_layer_total = 0

    # per-DMA columns (dma0_..., dma1_..., dma2_..., dma3_...) will be attached below

    # attach per-DMA columns (dma0_..., dma1_..., dma2_..., dma3_...)
    try:
        for k, v in per_dma_metrics.items():
            df1[k] = v
    except NameError:
        # per_dma_metrics may not exist if parsed_rows was empty; ensure zeros
        for did in range(4):
            for m in ['data_transfered_bytes','data_transfered_recv_bytes','data_per_send_bytes','data_per_recv_bytes','data_send_count_int','data_recv_count_int','send_speed_mb_s','recv_speed_mb_s','send_wait_int','recv_wait_int','summary_layer_total_int']:
                df1[f'dma{did}_{m}'] = 0

    # print avg power
    # print("Avg power for ", run_tag, " is ", avg_power)

    # add to first row to data frame open file name + _prf.csv
    if hw != "CPU":
        df2 = pd.read_csv(file + "_prf.csv")
        df2["runtag"] = run_tag
        df1["mm layers"] = df2["layer_total"]
        # remove layer_total from df2
        df2 = df2.drop(columns=["layer_total"])

        df3 = pd.merge(df1, df2, on="runtag")
        # add to main data frame
        df = pd.concat([df, df3])
    else:
        df = pd.concat([df, df1])

    df["token/s"] = (df["eval tokens"] + df["promt tokens"]) / (
        (df["eval time"] + df["prompt eval time"]) / 1000
    )
    df["actual_total_time"] = df["prompt eval time"] + df["eval time"]
    df["mm layers"] = df["mm layers"] if "mm layers" in df.columns else 0
    df["other_layers"] = df["actual_total_time"] - (df["mm layers"] / 1000)
    df["mm_total"] = df["actual_total_time"] - df["other_layers"]
    # df1["Joules"] = (df1["avg_power"]/1000000) * (df1["actual_total_time"] / 1000 )
    # df["avg_power"] = df["avg_power"] if "avg_power" in df.columns else 0
    df["Joules"] = (df["avg_power"] / 1000000) * (df["actual_total_time"] / 1000)

    return df


# main function
def main():
    # dir is an arg
    dir = sys.argv[1]
    print(dir)

    files = get_all_files(dir)

    print(files)
    df = pd.DataFrame()
    for file in files:
        # ensure proper path join (dir may not end with slash)
        filepath = os.path.join(dir, file)
        df = process_file(filepath, file, df)

    # sort df by model then hw the opt
    df = df.rename(columns={"total_time": "e2e_time"})
    df = df.rename(columns={"actual_total_time": "total_time (ms)"})
    df = df.rename(columns={"hw": "Hardware"})
    df = df.rename(columns={"model": "Model"})
    df["Total Time (s)"] = df["total_time (ms)"] / 1000
    df["Matmul Time (s)"] = df["mm_total"] / 1000



    df = df.sort_values(by=["Model", "Hardware", "opt"])
    # move model to first column, then threads, then board, then version, then opt, then hw
    cols = list(df.columns)
    cols.remove("Hardware")
    cols.remove("Total Time (s)")
    cols.remove("Matmul Time (s)")
    cols.remove("opt")
    cols.remove("version")
    cols.remove("board")
    cols.remove("threads")
    cols.remove("Model")
    cols.remove("total_time (ms)")
    cols.remove("mm_total")
    cols.remove("other_layers")
    cols.remove("avg_power")
    cols.remove("Joules")
    cols.remove("token/s")
    cols = [
        "Model",
        "Hardware",
        "Total Time (s)",
        "Matmul Time (s)",
        "token/s",
        "Joules",
        "other_layers",
        "total_time (ms)",
        "mm_total",
        "avg_power",
        "threads",
        "board",
        "version",
        "opt",
        
    ] + cols
    # ensure DMA columns (dma0_.. dma3_..) come last in the CSV
    dma_cols = [c for c in df.columns if re.match(r'^dma[0-3]_', c)]
    # remove any dma cols from the middle
    cols = [c for c in cols if c not in dma_cols]
    cols = cols + sorted(dma_cols)
    df = df[cols]

    df = df.drop(columns=["mm layers"])
    # file nan with 0
    df = df.fillna(0)
    print("Saving results to csv: ", f"{dir}/results.csv")

    # write to csv without index column
    # Use fixed-point float format to avoid scientific notation in output
    df.to_csv(f"{dir}/results.csv", index=False, float_format='%.6f')


if __name__ == "__main__":
    main()
