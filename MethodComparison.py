import os
import re
import csv
import math
import random
import time
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
NS = [100,250, 500]
M_FRACS = [0.1,0.3]
INSTANCES_PER_GROUP = 10       # 10 instàncies per (dataset,n,m)
ITERS = 100                    # 100 solucions per instància i mètode
SEED = 12345                   # per reproduïbilitat


ALPHA_TEUA = 0.1
BETA = 0.9


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


def list_instance_paths(dataset, n, base_dir=INST_DIR):
    folder = os.path.join(base_dir, dataset)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

    # Filtra por n en el nombre
    pat_n = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat_n.search(os.path.basename(f))]

    # Extrae el índice final del fichero (último número antes de .txt)
    def extract_idx(path):
        base = os.path.basename(path)
        m = re.search(r"(\d+)\.txt$", base)
        return int(m.group(1)) if m else 10**9

    files.sort(key=extract_idx)
    return files, extract_idx




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
    start = time.perf_counter()

    if constructive_name == "CGR":
        sol = constructive_CGR(inst)
    elif constructive_name == "CGR2":
        sol = constructive_CGR2(inst)
    else:
        raise ValueError(f"Unknown constructive: {constructive_name}")
    check_solution(sol)
    apply_ls(sol, ls_name)
    check_solution(sol)

    end = time.perf_counter()
    elapsed = end - start

    return sol["of"], elapsed


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


    group_data = {}

    for dataset in DATASETS:
        for n in NS:

            paths, extract_idx = list_instance_paths(dataset, n)
            if not paths:
                log(f"[WARN] No hi ha instàncies per dataset={dataset}, n={n}")
                continue

            log(f"== Dataset={dataset}, n={n}. Total instàncies: {len(paths)} ==")

            for path in paths:
                idx_file = extract_idx(path)  # 1,2,3,...

                # bloque 0: 1-10 ; bloque 1: 11-20 ; ...
                block = (idx_file - 1) // INSTANCES_PER_GROUP


                if block >= len(M_FRACS):
                    continue



                frac = M_FRACS[block]
                m = int(round(frac * n))

                inst = load_instance(dataset, path, p=m)
                log(f"  Instància: {inst['name']}  -> m={m} (frac={frac})")

                # executa 100 iteracions per cada mètode
                method_stats = {}
                method_best = {}
                method_time = {}

                for (cname, lsname) in METHODS:
                    vals = []
                    times = []
                    log(f"    -> Mètode {cname}+{lsname} ({ITERS} iteracions)")

                    for it in range(ITERS):
                        ofv, dt = run_one_iteration(inst, cname, lsname)
                        vals.append(ofv)
                        times.append(dt)

                        if (it + 1) % 10 == 0:
                            log(f"       iter {it + 1}/{ITERS}")

                    best_v = max(vals)
                    avg_v = sum(vals) / len(vals)
                    std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0
                    avg_t = sum(times) / len(times)

                    log(f"       FI {cname}+{lsname}: best={best_v:.6f} avg={avg_v:.6f} t/iter={avg_t:.6f}s")

                    method_stats[(cname, lsname)] = (best_v, avg_v, std_v)
                    method_best[(cname, lsname)] = best_v
                    method_time[(cname, lsname)] = avg_t

                # mejor global en la instancia
                best_global = max(method_best.values())

                # filas por método, ya con desviación y bandera
                for (cname, lsname) in METHODS:
                    best_v, avg_v, std_v = method_stats[(cname, lsname)]
                    avg_t = method_time[(cname, lsname)]

                    rel_dev = 0.0
                    if best_global > EPS:
                        rel_dev = (best_global - best_v) / best_global

                    is_best = 1 if abs(best_v - best_global) <= 1e-9 else 0

                    per_instance_rows.append({
                        "dataset": dataset,
                        "instance": inst["name"],
                        "n": n,
                        "m": m,
                        "constructive": cname,
                        "local_search": lsname,
                        "best_method_of": best_v,
                        "best_global_of": best_global,
                        "relative_dev": rel_dev,
                        "is_best": is_best,
                        "avg_time_per_iter": avg_t,
                        "avg_of": avg_v,
                        "std_of": std_v,
                        "iters": ITERS,
                        "alpha_teua": ALPHA_TEUA if cname == "CGR" else "",
                        "beta": BETA if cname == "CGR2" else "",
                    })

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
    # -----------------------
    # Summary desde per_instance_rows
    # -----------------------
    summary_acc = {}  # key=(dataset,n,m,constructive,local_search) -> acumuladores

    for row in per_instance_rows:
        key = (row["dataset"], row["n"], row["m"], row["constructive"], row["local_search"])
        acc = summary_acc.setdefault(key, {"dev_sum": 0.0, "best_sum": 0, "count": 0})
        acc["dev_sum"] += row["relative_dev"]
        acc["best_sum"] += row["is_best"]
        acc["count"] += 1

    summary_rows = []
    for (dataset, n, m, cname, lsname), acc in summary_acc.items():
        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "m": m,
            "constructive": cname,
            "local_search": lsname,
            "Dev_avg": acc["dev_sum"] / acc["count"],  # media de desviaciones (0-1)
            "#Best": acc["best_sum"],  # suma de indicadores
            "num_instances": acc["count"],  # número de instancias (filas) en ese grupo y método
            "iters_per_method": ITERS,
        })

    # ordenado bonito
    summary_rows.sort(key=lambda r: (r["dataset"], r["n"], r["m"], r["constructive"], r["local_search"]))

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
