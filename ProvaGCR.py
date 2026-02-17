"""
Tuning script for constructive heuristics (GRC and GRC2).
- Runs ONLY the constructive method multiple times with fixed seeds (common random numbers).
- Saves detailed runs and a summary table (per instance and global) to CSV.

Assumptions:
- instance.readInstance(path) returns dict with at least: inst['n'], inst['m'] (or feasibility uses sol), inst['d'].
- grc.construct(inst, alpha) returns a solution dict with sol['of'] objective value (as in your codebase).
- You will implement grc2.construct(inst, beta) similarly (optional).
"""

import os
import csv
import math
import random
import statistics
from collections import defaultdict

from structure import instance
from structure import solution
from constructives import cgrasp as grc # your module with construct(inst, alpha)
from constructives import cgr2 as grc2

# from algorithms import grc2  # uncomment when you have it

INSTANCE_FOLDER = "instances"
OUT_RUNS_CSV = "results/constructive_runs.csv"
OUT_SUMMARY_CSV = "results/constructive_summary.csv"

# Parameters to test
ALPHAS = [round(x / 10, 1) for x in range(1, 11)]  # 0.1..1.0
# If you also want 0.0 include it:
# ALPHAS = [round(x / 10, 1) for x in range(0, 11)]  # 0.0..1.0

BETAS = [round(x / 10, 1) for x in range(1, 11)]   # 0.1..1.0

# Repetitions (seeds). Use the SAME list for all algos/params -> fair comparison
N_RUNS = 20
BASE_SEED = 12345


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def run_grc(inst, alpha: float, seed: int):
    random.seed(seed)
    sol = grc.construct(inst, alpha)
    return sol

def run_grc2(inst, beta: float, seed: int):
    random.seed(seed)
    sol = grc2.construct(inst, beta)
    return sol


# def run_grc2(inst, beta: float, seed: int):
#     random.seed(seed)
#     sol = grc2.construct(inst, beta)
#     return sol["of"]


def get_instance_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        for name in filenames:
            path = os.path.join(root, name)
            files.append(path)
    return sorted(files)


def main():
    ensure_dir(OUT_RUNS_CSV)
    ensure_dir(OUT_SUMMARY_CSV)

    instance_files = get_instance_files(INSTANCE_FOLDER)
    used_instance_files = []

    seeds = [BASE_SEED + i for i in range(N_RUNS)]
    data = defaultdict(list)

    # ---- Detailed runs CSV ----
    with open(OUT_RUNS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "algo", "param_name", "param_value", "run_id", "seed", "objective"])

        for inst_file in instance_files:   # ✅ ACÍ era el problema
            inst = instance.readInstance(inst_file)

            if inst["p"] > inst["n"] or inst["p"] < 2:
                print(f"Skipping invalid instance: {inst_file} (p={inst['p']}, n={inst['n']})")
                continue

            used_instance_files.append(inst_file)

            # ---- GRC(alpha) ----
            for alpha in ALPHAS:
                for run_id, seed in enumerate(seeds, start=1):
                    sol = run_grc(inst, alpha, seed)
                    obj = sol["of"]

                    if obj <= 1e-12:
                        print("Zero objective in:", inst_file, "alpha:", alpha, "seed:", seed)
                        solution.printSolution(sol)

                    w.writerow([inst_file, "GRC", "alpha", alpha, run_id, seed, obj])
                    data[("GRC", alpha, inst_file)].append(obj)

            # ---- GRC2(beta) ----
            for beta in BETAS:
                for run_id, seed in enumerate(seeds, start=1):
                    sol2 = run_grc2(inst, beta, seed)
                    obj2 = sol2["of"]

                    if obj2 <= 1e-12:   # ✅ ací era obj2
                        print("Zero objective in:", inst_file, "beta:", beta, "seed:", seed)
                        solution.printSolution(sol2)

                    w.writerow([inst_file, "GRC2", "beta", beta, run_id, seed, obj2])
                    data[("GRC2", beta, inst_file)].append(obj2)

            print(f"Done: {inst_file}")

    # ---- Summary CSV ----
    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algo", "param_name", "param_value",
            "instances", "avg_mean", "avg_median", "avg_best", "avg_stdev",
            "#best_instances_by_mean", "#best_instances_by_best"
        ])

        for algo, params, pname in [
            ("GRC", ALPHAS, "alpha"),
            ("GRC2", BETAS, "beta"),
        ]:
            inst_stats = {}
            for inst_file in used_instance_files:
                per_param = {}
                for p in params:
                    vals = data.get((algo, p, inst_file), [])
                    if not vals:
                        continue
                    per_param[p] = {
                        "mean": mean(vals),
                        "median": median(vals),
                        "best": max(vals),
                        "stdev": stdev(vals),
                    }
                inst_stats[inst_file] = per_param

            best_by_mean = defaultdict(int)
            best_by_best = defaultdict(int)

            for inst_file, per_param in inst_stats.items():
                if not per_param:
                    continue

                max_mean = max(v["mean"] for v in per_param.values())
                for p, v in per_param.items():
                    if math.isclose(v["mean"], max_mean, rel_tol=1e-12, abs_tol=1e-12):
                        best_by_mean[p] += 1

                max_best = max(v["best"] for v in per_param.values())
                for p, v in per_param.items():
                    if math.isclose(v["best"], max_best, rel_tol=1e-12, abs_tol=1e-12):
                        best_by_best[p] += 1

            for p in params:
                means, medians, bests, stdevs = [], [], [], []
                count_inst = 0

                for inst_file in used_instance_files:
                    v = inst_stats.get(inst_file, {}).get(p)
                    if v is None:
                        continue
                    count_inst += 1
                    means.append(v["mean"])
                    medians.append(v["median"])
                    bests.append(v["best"])
                    stdevs.append(v["stdev"])

                if count_inst == 0:
                    continue

                w.writerow([
                    algo, pname, p,
                    count_inst,
                    mean(means),
                    mean(medians),
                    mean(bests),
                    mean(stdevs),
                    best_by_mean.get(p, 0),
                    best_by_best.get(p, 0),
                ])

    print("\nSaved:")
    print(f"- Detailed runs: {OUT_RUNS_CSV}")
    print(f"- Summary:       {OUT_SUMMARY_CSV}")



if __name__ == "__main__":
    main()
