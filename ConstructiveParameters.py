import os
import re
import csv
import math
import random
from datetime import datetime

from structure import solution

from constructives import cgrasp
try:
    from constructives import cgr2  # si existe
except ImportError:
    cgr2 = None  # fallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# -----------------------
# Configuración
# -----------------------
INST_DIR = "instances"
DATASETS = ["Geo", "Ran"]
NS = [100, 250, 500]

PARAMS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Iteraciones por instancia
ITERS = 5
SEED = 12345

PICK_MODE = "rand"  # "first" o "rand"

EPS = 1e-9

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

def list_instance_paths(dataset, n, base_dir=INST_DIR):
    folder = os.path.join(base_dir, dataset)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

    # Quédate solo con los que contengan el n (100/250/500)
    pat = re.compile(rf"\b{n}\b")
    files = [f for f in files if pat.search(os.path.basename(f))]

    def inst_idx(path):
        base = os.path.splitext(os.path.basename(path))[0]
        nums = re.findall(r"\d+", base)
        return int(nums[-1])  # en "Geo 100 17" -> 17

    files.sort(key=inst_idx)
    return files


def pick_one_from_range(files, start_idx_inclusive, end_idx_exclusive, seed_key):

    block = files[start_idx_inclusive:end_idx_exclusive]
    if not block:
        raise RuntimeError(f"No hay ficheros en el rango [{start_idx_inclusive},{end_idx_exclusive})")
    if PICK_MODE == "first":
        return block[0]
    # aleatorio reproducible
    rng = random.Random(SEED + seed_key)
    return rng.choice(block)

def check_solution(sol):
    inst = sol["instance"]
    n = inst["n"]
    for v in sol["sol"]:
        assert 0 <= v < n
    assert len(sol["sol"]) == inst["p"]
    of_eval = solution.evaluate(sol)
    assert abs(sol["of"] - of_eval) <= 1e-6, (sol["of"], of_eval)

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
# Ejecución 1 iteración (solo constructivo)
# -----------------------
def run_constructive(inst, method, param, it):
    """
    method: "CGR" o "CGR2"
    param: alpha (para CGR) o beta (para CGR2)
    """
    # Semilla por método/param/iteración (para comparaciones justas)
    random.seed(SEED + 100000*it + int(param*1000) + (0 if method == "CGR" else 777777))

    if method == "CGR":
        sol = cgrasp.construct(inst, param)  # alpha
    elif method == "CGR2":
        if cgr2 is not None:
            sol = cgr2.construct(inst, param)  # beta
        else:
            # Si tu CGR2 vive dentro de cgrasp como otra función, cambia esta línea:
            # sol = cgrasp.construct2(inst, param)
            raise ImportError("No encuentro constructives.cgrasp2. Ajusta el ADAPTADOR para CGR2.")
    else:
        raise ValueError(method)

    # check_solution(sol)  # opcional (más lento)
    return sol["of"]

# -----------------------
# Definición exacta de las 12 instancias
# -----------------------
def build_12_instance_plan():
    plan = []
    for dataset in ["Geo", "Ran"]:
        for n in [100, 250, 500]:
            files = list_instance_paths(dataset, n)
            print([os.path.basename(x) for x in files[:5]])
            print([os.path.basename(x) for x in files[10:15]])


            m_first = int(round(0.1 * n))
            path_first = pick_one_from_range(
                files, 0, 10,
                seed_key=hash((dataset, n, "first10")) % 10**6
            )
            plan.append({
                "dataset": dataset, "n": n, "m": m_first,
                "group": "first10", "path": path_first
            })


            m_second = int(round(0.3 * n))
            path_second = pick_one_from_range(
                files, 10, 20,
                seed_key=hash((dataset, n, "second10")) % 10**6
            )
            plan.append({
                "dataset": dataset, "n": n, "m": m_second,
                "group": "second10", "path": path_second
            })

    return plan


# -----------------------
# Experimento principal
# -----------------------
def experiment_cgr_vs_cgr2_12inst():
    log("== Experimento: CGR vs CGR2 (12 instancias, params 0.1..0.9) ==")
    log(f"CWD: {os.getcwd()}")
    os.makedirs("results", exist_ok=True)

    iter_rows = []      # guarda cada iteración (lo que pediste)
    summary_rows = []   # best por instancia/metodo/param

    plan = build_12_instance_plan()
    log(f"Instancias seleccionadas: {len(plan)} (esperadas 12)")


    for inst_idx, item in enumerate(plan, start=1):
        dataset, n, m, group, path = item["dataset"], item["n"], item["m"], item["group"], item["path"]
        inst = load_instance(dataset, path, p=m)
        inst_id = f"{dataset}|n={n}|m={m}|{inst['name']}"

        log(f"[{inst_idx}/12] {inst_id} (grupo={group})")

        for method in ["CGR", "CGR2"]:
            for param in PARAMS:
                vals = []
                for it in range(1, ITERS + 1):
                    ofv = run_constructive(inst, method, param, it)
                    vals.append(ofv)

                    iter_rows.append({
                        "inst_idx": inst_idx,
                        "dataset": dataset,
                        "group": group,
                        "instance": inst["name"],
                        "n": n,
                        "m": m,
                        "method": method,
                        "param": param,
                        "iter": it,
                        "of": ofv,
                    })

                summary_rows.append({
                    "inst_idx": inst_idx,
                    "dataset": dataset,
                    "group": group,
                    "instance": inst["name"],
                    "n": n,
                    "m": m,
                    "method": method,
                    "param": param,
                    "best_of": max(vals),
                    "is_best_param": 0,
                })


            rows_this = [r for r in summary_rows
                         if r["inst_idx"] == inst_idx and r["method"] == method]

            best_val = max(r["best_of"] for r in rows_this)

            for r in rows_this:
                r["is_best_param"] = 1 if abs(r["best_of"] - best_val) < 1e-9 else 0

        # guardado parcial
        write_csv(os.path.join("results", "cgr_cgr2_iterlog.csv"), iter_rows)
        write_csv(os.path.join("results", "cgr_cgr2_summary.csv"), summary_rows)
        log("  [SAVE] CSVs actualizados")

    log("[OK] Guardado final:")
    log(os.path.abspath(os.path.join("results", "cgr_cgr2_iterlog.csv")))
    log(os.path.abspath(os.path.join("results", "cgr_cgr2_summary.csv")))

if __name__ == "__main__":
    experiment_cgr_vs_cgr2_12inst()
