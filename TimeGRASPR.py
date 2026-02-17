# compare_grasp_pr_params.py
# Comparació de paràmetres per a GRASP+PR (Max-Min) usant el teu grasp_pr_time.execute(...)
#
# - Base GRASP: CGR + FLS
# - PR: greedy PR (prgreedy_good) + FLS després de PR
#
# Genera:
#   results/grasppr_per_instance.csv
#   results/grasppr_summary.csv
#
# IMPORTANT:
#  - Aquest script assumeix que tens grasp_pr_time.py i que exposa execute(inst, alpha, es_size, time_limit, time_doing_grasp)
#  - I que tens els teus loaders d'instàncies Geo/Ran com als experiments anteriors.

import os
import re
import csv
import math
import random
import statistics as stats
from datetime import datetime

from algorithms.grasp_pr_time import execute
# Si execute està en un altre mòdul/carpeta, ajusta l'import.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EPS = 1e-9

# -----------------------
# CONFIG
# -----------------------
INST_DIR = "instances"
DATASETS = ["Geo", "Ran"]
NS = [100, 250]
M_FRACS = [0.1, 0.3]
INSTANCES_PER_GROUP = 2

ALPHA = 0.3          # el que vols usar ara
SEED = 12345

TIME_LIMIT = 10      # segons per execució
RUNS = 2            # execucions per configuració (com que és time-based, 100 pot ser massa)

# Configs a comparar:
ES_SIZES = [5, 10, 15, 20]
TIME_SPLITS = [0.2, 0.4, 0.6, 0.8]   # time_doing_grasp
# Pots comentar una dimensió si vols fer només un estudi.
COMPARE_MODE = "es_only"  # "es_only", "split_only", "both"

# -----------------------
# Utils
# -----------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_geo_instance(path, p):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    K = int(lines[1])
    coords = []
    for ln in lines[2:2+n]:
        parts = ln.split()
        vec = list(map(float, parts[1:1+K]))
        coords.append(vec)

    d = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            s = 0.0
            for a, b in zip(coords[i], coords[j]):
                diff = a - b
                s += diff*diff
            dist = math.sqrt(s)
            d[i][j] = d[j][i] = dist
    return {"name": os.path.basename(path), "n": n, "p": p, "d": d, "dataset": "Geo"}

def load_ran_instance(path, p):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    d = [[0.0]*n for _ in range(n)]
    for ln in lines[1:]:
        i, j, val = ln.split()
        i = int(i); j = int(j); val = float(val)
        d[i][j] = d[j][i] = val
    return {"name": os.path.basename(path), "n": n, "p": p, "d": d, "dataset": "Ran"}

def load_instance(dataset, path, p):
    if dataset == "Geo":
        return load_geo_instance(path, p)
    if dataset == "Ran":
        return load_ran_instance(path, p)
    raise ValueError(dataset)

def pick_instance_paths(dataset, n, k, base_dir=INST_DIR):
    folder = os.path.join(base_dir, dataset)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    pat = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat.search(os.path.basename(f))]
    files.sort()
    return files[:k]

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# -----------------------
# Config generation
# -----------------------
def make_configs():
    cfgs = []
    if COMPARE_MODE in ("es_only", "both"):
        # Fixem split i variem es_size
        for b in ES_SIZES:
            cfgs.append({
                "name": f"ES={b},split=0.4",
                "es_size": b,
                "time_doing_grasp": 0.4,
            })
    if COMPARE_MODE in ("split_only", "both"):
        # Fixem es_size i variem split
        for g in TIME_SPLITS:
            cfgs.append({
                "name": f"ES=10,split={g}",
                "es_size": 10,
                "time_doing_grasp": g,
            })
    # Elimina duplicats si "both" (p.ex. ES=10,split=0.4 apareix dos voltes)
    seen = set()
    uniq = []
    for c in cfgs:
        key = (c["es_size"], c["time_doing_grasp"])
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

# -----------------------
# One run
# -----------------------
def run_one(inst, cfg, run_id):
    # Re-seed per run/config perquè siga reproduïble i comparable
    # (evita que l'ordre de proves afecte el RNG global)
    seed = SEED + 100000 * run_id + 97 * cfg["es_size"] + int(1000 * cfg["time_doing_grasp"])
    random.seed(seed)

    best_sol, iters = execute(
        inst,
        alpha=ALPHA,
        es_size=cfg["es_size"],
        time_limit=TIME_LIMIT,
        time_doing_grasp=cfg["time_doing_grasp"],
    )
    return best_sol["of"], iters

# -----------------------
# Experiment
# -----------------------
def experiment():
    log("== Comparativa GRASP+PR: elite set size / time split ==")
    log(f"CWD: {os.getcwd()}")
    log(f"ALPHA={ALPHA}, TIME_LIMIT={TIME_LIMIT}s, RUNS={RUNS}")
    os.makedirs("results", exist_ok=True)

    configs = make_configs()
    log(f"Configs: {len(configs)} -> " + ", ".join([c["name"] for c in configs]))

    per_instance_rows = []
    summary_rows = []

    # Per summary: guardem per grup (dataset,n,m) una llista de dicts {cfg_key: best_of}
    group_data = {}  # (dataset,n,m) -> list of dict(cfg_key -> best_of)

    per_instance_path = os.path.join("results", "grasppr_per_instance.csv")
    summary_path = os.path.join("results", "grasppr_summary.csv")

    try:
        for dataset in DATASETS:
            for n in NS:
                for frac in M_FRACS:
                    m = int(round(frac * n))
                    paths = pick_instance_paths(dataset, n, INSTANCES_PER_GROUP)

                    log(f"Grup dataset={dataset}, n={n}, m={m} | instàncies={len(paths)}")

                    for idx, path in enumerate(paths, start=1):
                        inst = load_instance(dataset, path, p=m)
                        log(f"  [{idx}/{len(paths)}] {inst['name']}")

                        cfg_best = {}

                        for cfg in configs:
                            vals = []
                            iters_list = []

                            log(f"    -> {cfg['name']} ({RUNS} runs)")
                            for r in range(RUNS):
                                ofv, itc = run_one(inst, cfg, r)
                                vals.append(ofv)
                                iters_list.append(itc)

                            best_v = max(vals)
                            avg_v = sum(vals) / len(vals)
                            std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0
                            avg_iters = sum(iters_list) / len(iters_list)

                            cfg_key = (cfg["es_size"], cfg["time_doing_grasp"])
                            cfg_best[cfg_key] = best_v

                            per_instance_rows.append({
                                "dataset": dataset,
                                "instance": inst["name"],
                                "n": n,
                                "m": m,
                                "alpha": ALPHA,
                                "time_limit_s": TIME_LIMIT,
                                "runs": RUNS,
                                "es_size": cfg["es_size"],
                                "time_doing_grasp": cfg["time_doing_grasp"],
                                "best_of": best_v,
                                "avg_of": avg_v,
                                "std_of": std_v,
                                "avg_grasp_iters": avg_iters,
                            })

                        group_data.setdefault((dataset, n, m), []).append(cfg_best)

                        # Guardat incremental (per si pares)
                        write_csv(per_instance_path, per_instance_rows)
                        log("    [SAVE] grasppr_per_instance.csv actualitzat")

        # Summary: Dev i #Best per configuració dins de cada grup (dataset,n,m)
        for (dataset, n, m), inst_list in group_data.items():
            # agreguem
            dev_sum = {}
            best_count = {}
            for cfg in configs:
                key = (cfg["es_size"], cfg["time_doing_grasp"])
                dev_sum[key] = 0.0
                best_count[key] = 0

            for inst_cfg_best in inst_list:
                best_overall = max(inst_cfg_best.values())
                for cfg_key, val in inst_cfg_best.items():
                    dev = 0.0 if best_overall <= EPS else 100.0 * (best_overall - val) / best_overall
                    dev_sum[cfg_key] += dev
                    if abs(val - best_overall) <= 1e-9:
                        best_count[cfg_key] += 1

            num_inst = len(inst_list)
            for cfg in configs:
                cfg_key = (cfg["es_size"], cfg["time_doing_grasp"])
                summary_rows.append({
                    "dataset": dataset,
                    "n": n,
                    "m": m,
                    "alpha": ALPHA,
                    "time_limit_s": TIME_LIMIT,
                    "runs": RUNS,
                    "es_size": cfg["es_size"],
                    "time_doing_grasp": cfg["time_doing_grasp"],
                    "Dev_avg_%": dev_sum[cfg_key] / num_inst,
                    "#Best": best_count[cfg_key],
                    "num_instances": num_inst,
                })

    finally:
        write_csv(per_instance_path, per_instance_rows)
        write_csv(summary_path, summary_rows)
        print("[OK] Saved:", os.path.abspath(per_instance_path))
        print("[OK] Saved:", os.path.abspath(summary_path))

if __name__ == "__main__":
    experiment()
