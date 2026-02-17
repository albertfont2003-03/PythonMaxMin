import random
from structure import solution

EPS = 1e-9

def improve_imls(sol, k=3, shuffle=False):
    """
    IMLS (Improved Local Search) per al Max-Min Diversity.
    - k: nombre de distàncies més baixes a considerar en e(i)
    - shuffle: si True, desempata aleatòriament (opcional)
    """
    improved = True
    while improved:
        improved = tryImprove_imls(sol, k=k, shuffle=shuffle)


def tryImprove_imls(sol, k=3, shuffle=False):
    """
    Una iteració d'IMLS.
    1) Troba els elements crítics i (d_i == d*)
    2) Tria i* amb menor e(i)
    3) Prova inserir candidats no seleccionats en ordre decreixent de e(s)
       i aplica el primer moviment que siga "millorant"
    """

    Sel = list(sol["sol"])
    if len(Sel) < 2:
        return False

    d_star = sol.get("of", None)
    if d_star is None:
        d_star = solution.evaluate(sol)

    # Computa d_i per a tots els seleccionats (amb "without=i" per excloure'l)
    d_i = {}
    for i in Sel:
        d_i[i] = solution.distanceToSol(sol, i, without=i)

    # Elements crítics: d_i == d*
    critical = [i for i in Sel if abs(d_i[i] - d_star) <= EPS]
    if not critical:
        # per seguretat numèrica, agafa els que tinguen el mínim d_i
        min_di = min(d_i.values())
        critical = [i for i in Sel if abs(d_i[i] - min_di) <= EPS]

    # Funció e(x): suma de les k distàncies més baixes de x a Sel (excloent optionally one element)
    def e_value(x, current_sel_set, exclude=None):
        dmat = sol["instance"]["d"]
        dists = []
        for s in current_sel_set:
            if exclude is not None and s == exclude:
                continue
            if s == x:
                continue
            dists.append(dmat[x][s])
        if not dists:
            return float("inf")
        dists.sort()
        kk = min(k, len(dists))
        return sum(dists[j] / (j + 1) for j in range(kk))

    # Tria i* amb menor e(i) entre crítics
    # (calculat respecte al Sel actual, excloent-se a ell mateix)
    if shuffle:
        random.shuffle(critical)

    e_crit = [(e_value(i, sol["sol"], exclude=i), i) for i in critical]
    e_crit.sort()
    crit_order = [i for (_, i) in e_crit]  # de pitjor a "menys pitjor"

    # Helper: comptar quants elements tenen d_i == d*
    def count_critical(sol_local):
        if len(sol_local["sol"]) < 2:
            return 0
        d_star_local = sol_local.get("of", solution.evaluate(sol_local))
        cnt = 0
        for i in sol_local["sol"]:
            di_local = solution.distanceToSol(sol_local, i, without=i)
            if abs(di_local - d_star_local) <= EPS:
                cnt += 1
        return cnt

    # Intentem eliminar crítics en e(i) creixent fins trobar un swap millorant
    for i_star in crit_order:
        # Treballem amb el conjunt després d'eliminar i*
        # (i* encara està dins sol; però calculem e(s) com si Sel \ {i*})
        sel_minus = set(sol["sol"])
        sel_minus.discard(i_star)

        # Llista de no seleccionats
        n = sol["instance"]["n"]
        unselected = [v for v in range(n) if not solution.contains(sol, v)]

        if shuffle:
            random.shuffle(unselected)

        # Ordena per e(s) decreixent (millors candidats primer)
        scored_unselected = [(e_value(s, sel_minus, exclude=None), s) for s in unselected]
        scored_unselected.sort(reverse=True)
        cand_order = [s for (_, s) in scored_unselected]

        # Guarda estat per comparar "millora"
        of_old = sol.get("of", solution.evaluate(sol))
        crit_old = count_critical(sol)

        # Prova cada candidat i aplica el primer moviment millorant
        for j in cand_order:
            # --- aplica swap temporal ---
            _remove_from_solution(sol, i_star)
            _add_to_solution(sol, j)

            of_new = sol.get("of", solution.evaluate(sol))
            crit_new = count_critical(sol)

            improving = (of_new > of_old + EPS) or (abs(of_new - of_old) <= EPS and crit_new < crit_old)

            if improving:
                return True

            # --- desfés swap si no millora ---
            _remove_from_solution(sol, j)
            _add_to_solution(sol, i_star)

        # si amb aquest i* no hi ha millora, prova el següent crític

    return False


# --- Wrappers per compatibilitat amb signatures antigues/noves ---

def _remove_from_solution(sol, u):
    try:
        # nova signatura: removeFromSolution(sol, u)
        solution.removeFromSolution(sol, u)
    except TypeError:
        # signatura antiga: removeFromSolution(sol, u, something)
        solution.removeFromSolution(sol, u, None)


def _add_to_solution(sol, u):
    try:
        # nova signatura: addToSolution(sol, u)
        solution.addToSolution(sol, u)
    except TypeError:
        # signatura antiga: addToSolution(sol, u, something)
        solution.addToSolution(sol, u, None)
