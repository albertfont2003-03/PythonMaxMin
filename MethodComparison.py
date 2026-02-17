import os
import re
import csv
import math
import random
import statistics as stats
from datetime import datetime
from structure import solution

from constructives import cgrasp,cgr2
from localsearch import lsfirstimp, lsbestimp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EPS = 1e-9


# -----------------------
# Configuració
# -----------------------
INST_DIR = "instances"  # dins tens Geo/ i Ran/
DATASETS = ["Geo","Ran"]
NS = [100,250]
M_FRACS = [0.1,0.3]           # m = 0.1n i m = 0.3n
INSTANCES_PER_GROUP = 10       # 10 instàncies per (dataset,n,m)
ITERS = 100                    # 100 solucions per instància i mètode
SEED = 12345                   # per reproduïbilitat

# Paràmetres constructius:
# paper: GRC alpha=0.95 però en la teua definició és invers -> alpha_teu = 1-0.95=0.05
ALPHA_TEUA = 0.05
BETA = 0.9

# IMLS param
IMLS_K = 3


# -----------------------
# Carrega instàncies
# -----------------------
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

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
    # filtre per n en el nom (p.ex. "Geo 100 1.txt" / "Ran 250 3.txt")
    pat = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat.search(os.path.basename(f))]
    files.sort()
    # agafa k (si vols aleatori, canvia-ho per random.sample)
    return files[:k]


# -----------------------
# Checks de correcció
# -----------------------
def check_solution(sol):
    inst = sol["instance"]
    n = inst["n"]
    # rang vàlid
    for v in sol["sol"]:
        assert 0 <= v < n
    # mida final
    assert len(sol["sol"]) == inst["p"]
    # consistència of
    of_eval = solution.evaluate(sol)
    assert abs(sol["of"] - of_eval) <= 1e-6, (sol["of"], of_eval)


# -----------------------
# Wrappers de mètodes
# -----------------------
def constructive_CGR(inst):
    sol = cgrasp.construct(inst, ALPHA_TEUA)
    return sol

def constructive_CGR2(inst):
    sol = cgr2.construct(inst, BETA)
    return sol

def apply_ls(sol, ls_name):
    if ls_name == "BLS":
        lsbestimp.improve(sol,max_iter=100)
    elif ls_name == "FLS":
        lsfirstimp.improve(sol,max_iter=200)
    else:
        raise ValueError(ls_name)
    return sol


METHODS = [
    ("CGR",  "BLS"),
    ("CGR",  "FLS"),
    ("CGR2", "BLS"),
    ("CGR2", "FLS"),
]



def run_one_iteration(inst, constructive_name, ls_name):
    if constructive_name == "CGR":
        sol = constructive_CGR(inst)
    elif constructive_name == "CGR2":
        sol = constructive_CGR2(inst)
    else:
        raise ValueError(constructive_name)

    # assegura factible i coherent abans de LS (si vols)
    check_solution(sol)

    apply_ls(sol, ls_name)

    # comprova després de LS
    check_solution(sol)
    return sol["of"]


# -----------------------
# Experiments i CSV
# -----------------------
def experiment():
    log("Començant experiments")
    log(f"CWD (on guardarà results): {os.getcwd()}")
    os.makedirs("results", exist_ok=True)
    log("Carpeta 'results' creada (si no existia)")
    random.seed(SEED)

    per_instance_rows = []
    summary_rows = []

    # per a Score: assignarem punts 6..1 segons rank per instància (empats: punts mitjans)
    def rank_points(values_dict):
        # values_dict: method_key -> best_value
        items = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
        # ranks amb empats
        points = {}
        pts_by_rank = [6, 5, 4, 3, 2, 1]
        i = 0
        while i < len(items):
            j = i
            while j < len(items) and abs(items[j][1] - items[i][1]) <= 1e-9:
                j += 1
            # empats en [i, j)
            avg_pts = sum(pts_by_rank[i:j]) / (j - i)
            for t in range(i, j):
                points[items[t][0]] = avg_pts
            i = j
        return points

    # recollim per grup per a summary
    # key: (dataset,n,m) -> list of per-instance method bests
    group_data = {}  # (dataset,n,m) -> list of dict(method->best)

    for dataset in DATASETS:
        for n in NS:
            for frac in M_FRACS:
                m = int(round(frac * n))

                log(f"== Grup: dataset={dataset}, n={n}, m={m} ==")
                log("Buscant instàncies...")

                paths = pick_instance_paths(dataset, n, INSTANCES_PER_GROUP)
                log(f"Trobades {len(paths)} instàncies. Primera: {os.path.basename(paths[0])}")

                for idx, path in enumerate(paths, start=1):
                    inst = load_instance(dataset, path, p=m)
                    log(f"  [{idx}/{len(paths)}] Instància: {inst['name']}")

                    # executa 100 iteracions per cada mètode
                    method_stats = {}
                    method_best = {}

                    for (cname, lsname) in METHODS:
                        vals = []
                        log(f"    -> Mètode {cname}+{lsname} ({ITERS} iteracions)")

                        for it in range(ITERS):
                            ofv = run_one_iteration(inst, cname, lsname)
                            vals.append(ofv)

                            # heartbeat cada 10 iteracions (opcional però útil)
                            if (it + 1) % 10 == 0:
                                log(f"       iter {it + 1}/{ITERS}")

                        best_v = max(vals)
                        avg_v = sum(vals) / len(vals)
                        log(f"       FI {cname}+{lsname}: best={best_v:.6f} avg={avg_v:.6f}")

                        avg_v = sum(vals) / len(vals)
                        std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0

                        method_stats[(cname, lsname)] = (best_v, avg_v, std_v)
                        method_best[(cname, lsname)] = best_v

                        per_instance_rows.append({
                            "dataset": dataset,
                            "instance": inst["name"],
                            "n": n,
                            "m": m,
                            "constructive": cname,
                            "local_search": lsname,
                            "best_of": best_v,
                            "avg_of": avg_v,
                            "std_of": std_v,
                            "iters": ITERS,
                            "alpha_teua": ALPHA_TEUA if cname == "CGR" else "",
                            "beta": BETA if cname == "CGR2" else "",
                            "imls_k": IMLS_K if lsname == "IMLS" else "",
                        })

                    # guarda per summary
                    key = (dataset, n, m)
                    group_data.setdefault(key, []).append(method_best)
                    # --- guardat parcial per no perdre progrés ---
                    os.makedirs("results", exist_ok=True)

                    per_instance_path = os.path.join("results", "per_instance.csv")
                    with open(per_instance_path, "w", newline="", encoding="utf-8") as f:
                        fieldnames = list(per_instance_rows[0].keys()) if per_instance_rows else []
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        if fieldnames:
                            w.writeheader()
                            w.writerows(per_instance_rows)

                    print("[SAVE] per_instance:", os.path.abspath(per_instance_path))

    # construeix summary: Dev, #Best, Score
    for (dataset, n, m), instance_list in group_data.items():
        # acumula
        dev_sum = {meth: 0.0 for meth in METHODS}
        best_count = {meth: 0 for meth in METHODS}
        score_sum = {meth: 0.0 for meth in METHODS}

        for method_best in instance_list:
            best_overall = max(method_best.values())
            # Dev i #Best
            for meth, val in method_best.items():
                dev = 0.0
                if best_overall > EPS:
                    dev = 100.0 * (best_overall - val) / best_overall
                dev_sum[meth] += dev
                if abs(val - best_overall) <= 1e-9:
                    best_count[meth] += 1

            # Score per rank
            pts = rank_points(method_best)
            for meth, pnt in pts.items():
                score_sum[meth] += pnt

        num_inst = len(instance_list)
        for (cname, lsname) in METHODS:
            summary_rows.append({
                "dataset": dataset,
                "n": n,
                "m": m,
                "constructive": cname,
                "local_search": lsname,
                "Dev_avg_%": dev_sum[(cname, lsname)] / num_inst,
                "#Best": best_count[(cname, lsname)],
                "Score": score_sum[(cname, lsname)],
                "num_instances": num_inst,
                "iters_per_method": ITERS,
            })

    # escriu CSVs
    os.makedirs("results", exist_ok=True)

    per_instance_path = os.path.join("results", "per_instance.csv")
    with open(per_instance_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_instance_rows[0].keys()) if per_instance_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_instance_rows)


    summary_path = os.path.join("results", "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[OK] CSV per instància: {per_instance_path}")
    print(f"[OK] CSV resum:        {summary_path}")


if __name__ == "__main__":
    experiment()
