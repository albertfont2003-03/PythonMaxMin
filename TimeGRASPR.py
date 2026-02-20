
# compare_grasppr_configs_like_final.py
import os, re, csv, math, random, time
import statistics as stats
from datetime import datetime

from algorithms.grasp_pr_time import execute  # GRASP+PR time-based

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EPS = 1e-12

# ---------------- CONFIG ----------------
INST_DIR = "instances"
DATASETS = ["Geo", "Ran"]
NS = [100, 250, 500]
M_FRACS = [0.1, 0.3]
INSTANCES_PER_GROUP = 10

ALPHA = 0.1
SEED = 12345
TIME_LIMIT = 15
RUNS = 3

ELITE_SIZES = [ 5, 10, 15]
TIME_DOING_GRASP = 0.6   # fijo


# ---------------- Utils ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def extract_idx(path):
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.txt$", base)
    return int(m.group(1)) if m else 10**9

def list_instance_paths(dataset, n, base_dir=INST_DIR):
    folder = os.path.join(base_dir, dataset)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    pat = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat.search(os.path.basename(f))]
    files.sort(key=extract_idx)
    return files

def m_for_instance(n, idx_file):
    block = (idx_file - 1) // INSTANCES_PER_GROUP
    frac = M_FRACS[min(block, len(M_FRACS) - 1)]
    return int(round(frac * n)), frac

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# -------- loaders (cópialos de tu script actual) --------
def load_geo_instance(path, p):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0]); K = int(lines[1])
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

# ---------------- Runs ----------------
def run_one(inst, es_size, run_id):
    seed = SEED + 100000 * run_id + 97 * es_size
    random.seed(seed)

    best_sol, iters = execute(
        inst,
        alpha=ALPHA,
        es_size=es_size,
        time_limit=TIME_LIMIT,
        time_doing_grasp=TIME_DOING_GRASP,
    )
    return best_sol["of"], iters


# ---------------- Experiment ----------------
def experiment():
    os.makedirs("results", exist_ok=True)
    per_instance_rows = []
    summary_rows = []

    per_instance_path = os.path.join("results", "configs_per_instance.csv")
    summary_path = os.path.join("results", "configs_summary.csv")

    # acumuladores summary
    summary_acc = {}  # (dataset,n,m,es_size,split) -> dev_sum, best_sum, count

    for dataset in DATASETS:
        for n in NS:
            paths = list_instance_paths(dataset, n)
            if not paths:
                log(f"[WARN] No hay instancias para {dataset} n={n}")
                continue

            log(f"== Dataset={dataset} n={n}. Instancias={len(paths)} ==")

            for path in paths:
                idx_file = extract_idx(path)
                m, frac = m_for_instance(n, idx_file)
                inst = load_instance(dataset, path, p=m)

                log(f"  {inst['name']} (idx={idx_file}) -> m={m} (frac={frac})")

                es_best = {}
                es_avg = {}
                es_std = {}

                for es_size in ELITE_SIZES:
                    vals = []
                    for r in range(RUNS):
                        ofv, _ = run_one(inst, es_size, r)
                        vals.append(ofv)

                    es_best[es_size] = max(vals)
                    es_avg[es_size] = sum(vals) / len(vals)
                    es_std[es_size] = stats.pstdev(vals) if len(vals) > 1 else 0.0

                best_global = max(es_best.values())

                for es_size in ELITE_SIZES:
                    best_v = es_best[es_size]
                    rel_dev = 0.0 if best_global <= EPS else (best_global - best_v) / best_global
                    is_best = 1 if abs(best_v - best_global) <= 1e-9 else 0

                    per_instance_rows.append({
                        "dataset": dataset,
                        "instance": inst["name"],
                        "n": n,
                        "m": m,
                        "frac": frac,
                        "alpha": ALPHA,
                        "time_limit_s": TIME_LIMIT,
                        "runs": RUNS,

                        "es_size": es_size,
                        "time_doing_grasp": TIME_DOING_GRASP,

                        "best_es_of": best_v,
                        "best_global_of": best_global,
                        "relative_dev": rel_dev,
                        "is_best": is_best,

                        "avg_of": es_avg[es_size],
                        "std_of": es_std[es_size],
                    })

                    acc_key = (dataset, n, m, es_size)
                    acc = summary_acc.setdefault(acc_key, {"dev_sum": 0.0, "best_sum": 0, "count": 0})
                    acc["dev_sum"] += rel_dev
                    acc["best_sum"] += is_best
                    acc["count"] += 1

                # guardado incremental
                write_csv(per_instance_path, per_instance_rows)
                log("    [SAVE] configs_per_instance.csv actualizado")

    # 4) Summary
    for (dataset, n, m, es_size), acc in summary_acc.items():
        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "m": m,
            "alpha": ALPHA,
            "time_limit_s": TIME_LIMIT,
            "runs": RUNS,
            "es_size": es_size,
            "time_doing_grasp": TIME_DOING_GRASP,
            "Dev_avg": acc["dev_sum"] / acc["count"],
            "#Best": acc["best_sum"],
            "num_instances": acc["count"],
        })

    summary_rows.sort(key=lambda r: (r["dataset"], r["n"], r["m"], r["es_size"]))


    write_csv(summary_path, summary_rows)

    log("[OK] Saved:")
    log(os.path.abspath(per_instance_path))
    log(os.path.abspath(summary_path))

if __name__ == "__main__":
    experiment()

