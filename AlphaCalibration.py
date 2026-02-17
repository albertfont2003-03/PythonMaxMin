import os
import re
import csv
import math
import random
import statistics as stats
from datetime import datetime

from structure import solution
from constructives import cgrasp
from localsearch import lsfirstimp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EPS = 1e-9

# -----------------------
# Configuració
# -----------------------
INST_DIR = "instances"
DATASETS = ["Geo", "Ran"]
NS = [100, 250]
M_FRACS = [0.1, 0.3]
INSTANCES_PER_GROUP = 10

ITERS = 100
SEED = 12345

ALPHAS = [round(0.1 * k, 1) for k in range(1, 10)]  # 0.1..0.9
FLS_MAX_ITER = 200  # pressupost LS

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

def check_solution(sol):
    inst = sol["instance"]
    n = inst["n"]
    for v in sol["sol"]:
        assert 0 <= v < n
    assert len(sol["sol"]) == inst["p"]
    of_eval = solution.evaluate(sol)
    assert abs(sol["of"] - of_eval) <= 1e-6, (sol["of"], of_eval)

# -----------------------
# Mètode: CGR + FLS
# -----------------------
def run_one_iteration(inst, alpha, it):
    # seed per run (per a comparació més neta entre alphas)
    random.seed(SEED + 100000*it + int(alpha*100))

    sol = cgrasp.construct(inst, alpha)
    # check_solution(sol)  # opcional (ralentitza)
    lsfirstimp.improve(sol, max_iter=FLS_MAX_ITER)
    # check_solution(sol)  # opcional
    return sol["of"]

# -----------------------
# Guardat
# -----------------------
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
# Experiments
# -----------------------
def experiment_alpha():
    log("== Calibratge d'alpha per a CGR+FLS ==")
    log(f"CWD: {os.getcwd()}")
    os.makedirs("results", exist_ok=True)

    per_instance_rows = []
    summary_rows = []

    # guardarem bests per construir summary
    # key: (dataset,n,m,alpha) -> list of best_of per instance
    group_bests = {}

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

                        for alpha in ALPHAS:
                            vals = []
                            for it in range(ITERS):
                                vals.append(run_one_iteration(inst, alpha, it))

                            best_v = max(vals)
                            avg_v = sum(vals) / len(vals)
                            std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0

                            per_instance_rows.append({
                                "dataset": dataset,
                                "instance": inst["name"],
                                "n": n,
                                "m": m,
                                "alpha": alpha,
                                "best_of": best_v,
                                "avg_of": avg_v,
                                "std_of": std_v,
                                "iters": ITERS,
                                "fls_max_iter": FLS_MAX_ITER,
                            })

                            group_bests.setdefault((dataset, n, m, alpha), []).append(best_v)

                        # guardat parcial (per si pares)
                        write_csv(os.path.join("results", "alpha_per_instance.csv"), per_instance_rows)
                        log("    [SAVE] alpha_per_instance.csv actualitzat")

        # summary: mitjana i desviació respecte al millor alpha del mateix grup (dataset,n,m)
        # primer calculem el millor per grup (dataset,n,m)
        best_by_group = {}  # (dataset,n,m) -> best value among alphas (using avg of instance bests)
        avg_best_by_alpha = {}  # (dataset,n,m,alpha) -> avg(best_of over instances)

        for (dataset, n, m, alpha), best_list in group_bests.items():
            avg_best = sum(best_list) / len(best_list)
            avg_best_by_alpha[(dataset, n, m, alpha)] = avg_best
            key = (dataset, n, m)
            best_by_group[key] = max(best_by_group.get(key, -float("inf")), avg_best)

        for (dataset, n, m, alpha), avg_best in avg_best_by_alpha.items():
            best_ref = best_by_group[(dataset, n, m)]
            dev = 0.0 if best_ref <= EPS else 100.0 * (best_ref - avg_best) / best_ref
            summary_rows.append({
                "dataset": dataset,
                "n": n,
                "m": m,
                "alpha": alpha,
                "avg_best_of": avg_best,
                "dev_vs_best_alpha_%": dev,
                "num_instances": len(group_bests[(dataset, n, m, alpha)]),
                "iters": ITERS,
                "fls_max_iter": FLS_MAX_ITER,
            })

    finally:
        write_csv(os.path.join("results", "alpha_per_instance.csv"), per_instance_rows)
        write_csv(os.path.join("results", "alpha_summary.csv"), summary_rows)
        print("[OK] Saved:", os.path.abspath(os.path.join("results", "alpha_per_instance.csv")))
        print("[OK] Saved:", os.path.abspath(os.path.join("results", "alpha_summary.csv")))

if __name__ == "__main__":
    experiment_alpha()
