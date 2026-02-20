import os
import re
import math


def _last_number_in_filename(path: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 1


def infer_p_for_geo_ran(path: str, n: int) -> int:
    # Geo/Ran: 20 instancias por n:
    # 1..10  -> p = 0.1n
    # 11..20 -> p = 0.3n
    idx = _last_number_in_filename(path)
    p = int(round((0.1 if idx <= 10 else 0.3) * n))
    return max(2, min(p, n))


def infer_p_for_glover(path: str, n: int) -> int:
    # Glover: 5 instancias por n, m en [0.2n, 0.8n]
    # Mapeo típico en 5 niveles:
    ratios = [0.2, 0.35, 0.5, 0.65, 0.8]  # id 1..5
    idx = _last_number_in_filename(path)
    idx = max(1, min(idx, 5))
    p = int(round(ratios[idx - 1] * n))
    return max(2, min(p, n))


def readInstance(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    lower = path.lower()
    inst = {}

    # ---------------- RAN (n + triples u v dist) ----------------
    if "ran" in lower:
        n = int(lines[0])
        p = infer_p_for_geo_ran(path, n)

        d = [[0.0] * n for _ in range(n)]
        for line in lines[1:]:
            u, v, dist = line.split()
            u = int(u); v = int(v); dist = float(dist)
            d[u][v] = dist
            d[v][u] = dist

        inst["n"] = n
        inst["p"] = p
        inst["d"] = d
        return inst

    # ---------------- GEO / GLOVER (n + k + coords) ----------------
    n = int(lines[0])
    k = int(lines[1])

    if "geo" in lower:
        p = infer_p_for_geo_ran(path, n)      # 0.1n o 0.3n según id
    elif "glover" in lower:
        p = infer_p_for_glover(path, n)       # entre 0.2n y 0.8n según id
    else:
        p = max(2, min(int(round(0.3 * n)), n))

    coords = [None] * n

    # leer exactamente n puntos
    for line in lines[2:2 + n]:
        parts = line.split()
        idx_raw = int(parts[0])

        # índices pueden ser 0..n-1 o 1..n (lo hacemos robusto)
        if idx_raw == n:
            idx = idx_raw - 1
        else:
            idx = idx_raw

        coords[idx] = list(map(float, parts[1:1 + k]))

    if any(c is None for c in coords):
        raise ValueError(f"Faltan coordenadas o índices mal formateados en {path}")

    # construir matriz euclídea
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for t in range(k):
                diff = coords[i][t] - coords[j][t]
                s += diff * diff
            dist = math.sqrt(s)
            d[i][j] = dist
            d[j][i] = dist

    if p < 2 or p > n:
        raise ValueError(f"Instancia inválida {path}: p={p}, n={n}")

    inst["n"] = n
    inst["p"] = p
    inst["d"] = d
    return inst
