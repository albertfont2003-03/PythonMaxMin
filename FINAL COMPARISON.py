import os
import re
import csv
import math
import random
import time
import copy
import statistics as stats
from datetime import datetime

from structure import solution
from constructives import cgrasp
from localsearch import lsfirstimp


from algorithms.grasp_pr_time import execute as grasp_pr_execute


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EPS = 1e-12

# -----------------------
# Config
# -----------------------
INST_DIR = "instances"
DATASETS = ["Geo", "Ran"]
NS = [100, 250, 500]

M_FRACS = [0.1, 0.3]
INSTANCES_PER_GROUP = 10


REPS = 3
SEED = 12345

ALPHA = 0.1

# PR params
PR_ES_SIZE = 10
PR_TIME_DOING_GRASP = 0.6


# -----------------------
# Utils
# -----------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def extract_idx(path):
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.txt$", base)
    return int(m.group(1)) if m else 10**9

def list_instance_paths(dataset, n, base_dir=INST_DIR):
    folder = os.path.join(base_dir, dataset)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

    pat_n = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat_n.search(os.path.basename(f))]

    files.sort(key=extract_idx)
    return files

def m_for_instance(n, idx_file):
    # bloque 0: 1-10 => 0.1n
    # bloque 1: 11-20 => 0.3n
    block = (idx_file - 1) // INSTANCES_PER_GROUP

    frac = M_FRACS[min(block, len(M_FRACS) - 1)]
    return int(round(frac * n)), frac


# -----------------------
# Load instances
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


# -----------------------
# Checks
# -----------------------
def check_solution(sol):
    inst = sol["instance"]
    n = inst["n"]
    for v in sol["sol"]:
        assert 0 <= v < n
    assert len(sol["sol"]) == inst["p"]
    of_eval = solution.evaluate(sol)
    assert abs(sol["of"] - of_eval) <= 1e-6, (sol["of"], of_eval)


# -----------------------
# Methods (time-based)
# -----------------------
def grasp_time_execute(inst, alpha, time_limit):

    best = None
    iters = 0
    t0 = time.time()


    mi = 50 if inst["n"] >= 500 else 200

    while True:
        # No empieces otra iteración si ya se acabó el tiempo
        if time.time() - t0 >= time_limit:
            break

        iters += 1
        sol = cgrasp.construct(inst, alpha)
        check_solution(sol)


        if time.time() - t0 >= time_limit:
            break

        lsfirstimp.improve(sol, max_iter=mi)
        check_solution(sol)

        if best is None or sol["of"] > best["of"]:
            best = copy.deepcopy(sol)

    return best, iters



def run_method_reps(inst, method_name, time_limit_instance):


    best_vals = []
    total_time = 0.0

    for r in range(REPS):

        random.seed(SEED + 1000*r + (0 if method_name == "GRASP" else 1))

        t0 = time.time()
        if method_name == "GRASP":
            best_sol, iters = grasp_time_execute(inst, ALPHA, time_limit_instance)

        elif method_name == "GRASP_PR":
            best_sol, iters = grasp_pr_execute(
                inst,
                ALPHA,
                PR_ES_SIZE,
                time_limit_instance,
                PR_TIME_DOING_GRASP
            )

        else:
            raise ValueError(method_name)
        total_time += (time.time() - t0)

        best_vals.append(best_sol["of"])

    return {
        "best_method_of": max(best_vals),
        "avg_best_of": sum(best_vals) / len(best_vals),
        "std_best_of": stats.pstdev(best_vals) if len(best_vals) > 1 else 0.0,
        "avg_time_per_rep": total_time / REPS,
        "reps": REPS
    }


# -----------------------
# Experiment
# -----------------------
def experiment():
    os.makedirs("results", exist_ok=True)
    per_instance_rows = []

    for dataset in DATASETS:
        for n in NS:
            paths = list_instance_paths(dataset, n)
            if not paths:
                log(f"[WARN] No hay instancias para {dataset} n={n}")
                continue

            log(f"== Dataset={dataset} n={n}. Total instancias: {len(paths)} ==")

            for path in paths:
                idx_file = extract_idx(path)
                m, frac = m_for_instance(n, idx_file)

                inst = load_instance(dataset, path, p=m)
                if n == 500:
                    time_limit_instance = 30.0
                else:
                    time_limit_instance = 15.0

                log(f"  Instancia: {inst['name']} (idx={idx_file}) -> m={m} (frac={frac})")

                # Ejecutar ambos métodos (10 reps cada uno)
                r_grasp = run_method_reps(inst, "GRASP", time_limit_instance)
                r_pr = run_method_reps(inst, "GRASP_PR", time_limit_instance)

                best_global = max(r_grasp["best_method_of"], r_pr["best_method_of"])

                def make_row(method_label, r):
                    rel_dev = 0.0 if best_global <= EPS else (best_global - r["best_method_of"]) / best_global
                    is_best = 1 if abs(r["best_method_of"] - best_global) <= 1e-9 else 0
                    return {
                        "dataset": dataset,
                        "instance": inst["name"],
                        "n": n,
                        "m": m,
                        "frac": frac,

                        "method": method_label,
                        "best_method_of": r["best_method_of"],
                        "best_global_of": best_global,
                        "relative_dev": rel_dev,
                        "is_best": is_best,

                        "reps": r["reps"],
                        "time_limit_s": time_limit_instance,
                        "alpha": ALPHA,

                        # opcional (útil)
                        "avg_best_of": r["avg_best_of"],
                        "std_best_of": r["std_best_of"],
                        "avg_time_per_rep_s": r["avg_time_per_rep"],

                        # PR params (vacío en GRASP)
                        "es_size": PR_ES_SIZE if method_label == "GRASP_PR" else "",
                        "time_doing_grasp": PR_TIME_DOING_GRASP if method_label == "GRASP_PR" else "",
                    }

                per_instance_rows.append(make_row("GRASP", r_grasp))
                per_instance_rows.append(make_row("GRASP_PR", r_pr))

                # guardado parcial
                per_instance_path = os.path.join("results", "finalcomparison.csv")
                with open(per_instance_path, "w", newline="", encoding="utf-8") as f:
                    fieldnames = list(per_instance_rows[0].keys())
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(per_instance_rows)

                log(f"[SAVE] {os.path.abspath(per_instance_path)}")


    summary_acc = {}
    for row in per_instance_rows:
        key = (row["dataset"], row["n"], row["m"], row["method"], row["time_limit_s"])
        acc = summary_acc.setdefault(key, {"dev_sum": 0.0, "best_sum": 0, "count": 0})
        acc["dev_sum"] += row["relative_dev"]
        acc["best_sum"] += row["is_best"]
        acc["count"] += 1

    summary_rows = []
    for (dataset, n, m, method,tlim), acc in summary_acc.items():
        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "m": m,
            "method": method,
            "Dev_avg": acc["dev_sum"] / acc["count"],
            "#Best": acc["best_sum"],
            "num_instances": acc["count"],
            "reps": REPS,
            "time_limit_used": tlim,
            "alpha": ALPHA,
            "es_size": PR_ES_SIZE if method == "GRASP_PR" else "",
            "time_doing_grasp": PR_TIME_DOING_GRASP if method == "GRASP_PR" else "",
        })

    summary_rows.sort(key=lambda r: (r["dataset"], r["n"], r["m"], r["method"]))

    summary_path = os.path.join("results", "summaryfinalcomp.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader()
            w.writerows(summary_rows)

    log(f"[OK] summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    experiment()