from structure import solution


def _nn_in_solution(sol):
    """Calcula, per a cada element s en S, les 2 distàncies més menudes a altres
    elements de S (nn1, nn2) i quin era el veí més pròxim (arg1).

    Cost: O(p^2)
    """
    S = list(sol['sol'])
    d = sol['instance']['d']

    nn1 = {s: float('inf') for s in S}
    nn2 = {s: float('inf') for s in S}
    arg1 = {s: None for s in S}

    for a_i in range(len(S)):
        a = S[a_i]
        for b_i in range(a_i + 1, len(S)):
            b = S[b_i]
            dist = d[a][b]

            # update a
            if dist < nn1[a]:
                nn2[a] = nn1[a]
                nn1[a] = dist
                arg1[a] = b
            elif dist < nn2[a]:
                nn2[a] = dist

            # update b
            if dist < nn1[b]:
                nn2[b] = nn1[b]
                nn1[b] = dist
                arg1[b] = a
            elif dist < nn2[b]:
                nn2[b] = dist

    return nn1, nn2, arg1


def _best_two_to_S(sol, candidates):
    """Per a cada candidat j, calcula les 2 distàncies més menudes a la solució
    actual S (best1, best2) i l'element de S que dona best1 (arg1).

    Cost: O(|candidates| * p)
    """
    S = sol['sol']
    d = sol['instance']['d']

    best1 = {}
    best2 = {}
    arg1 = {}

    for j in candidates:
        b1 = float('inf')
        b2 = float('inf')
        a1 = None
        for s in S:
            dist = d[j][s]
            if dist < b1:
                b2 = b1
                b1 = dist
                a1 = s
            elif dist < b2:
                b2 = dist
        best1[j] = b1
        best2[j] = b2
        arg1[j] = a1

    return best1, best2, arg1


def _compute_of_without(current, A, nn1, nn2, argS):
    """Per a cada i en A (elements a traure), calcula OF(S \ {i})
    sense recomputar evaluate() per cada i.

    Cost: O(|A| * p)
    """
    of_without = {}
    size_after = len(current['sol']) - 1

    for i in A:
        # coherent amb solution.evaluate(): si |S|<2 -> 0.0
        if size_after < 2:
            of_without[i] = 0.0
            continue

        best = float('inf')
        for s in current['sol']:
            if s == i:
                continue
            # si el veí més pròxim de s era i, cal usar el 2n més pròxim
            val = nn1[s] if argS[s] != i else nn2[s]
            if val < best:
                best = val
        of_without[i] = best

    return of_without


def findBestSwap(current_sol, sel_initiating_dif, sel_guiding_dif):
    """Troba el millor swap (i -> j) en una iteració de Path Relinking greedy.

    IMPORTANT:
    - NO crea còpies de solució per cada (i,j)
    - NO crida evaluate() per cada candidat

    Retorna:
      - best_swap = (i, j) o None
      - best_of (OF resultant del swap)
    """
    if not sel_initiating_dif or not sel_guiding_dif:
        return None, None

    # 1) info NN dins S
    nn1, nn2, argS = _nn_in_solution(current_sol)

    # 2) OF(S\{i}) per a cada i que podria eixir
    of_without = _compute_of_without(current_sol, sel_initiating_dif, nn1, nn2, argS)

    # 3) per a cada j candidat, 2 millors distàncies a S
    b1, b2, argJ = _best_two_to_S(current_sol, sel_guiding_dif)

    # 4) avaluació O(1) per parella (i,j)
    best_of = -float('inf')
    best_swap = None

    for i in sel_initiating_dif:
        base = of_without[i]
        for j in sel_guiding_dif:
            dist_to_S_wo_i = b1[j] if argJ[j] != i else b2[j]
            cand_of = min(base, dist_to_S_wo_i)

            if cand_of > best_of:
                best_of = cand_of
                best_swap = (i, j)

    return best_swap, best_of


def greedyPathRelinking(initiating_sol, guiding_sol):
    """Greedy Path Relinking (PR) per al Max-Min Diversity.

    - Parteix d'initiating_sol i va transformant-la cap a guiding_sol.
    - En cada pas, prova tots els swaps possibles (i en A, j en B) i tria
      el que maximitza l'OF resultant.
    - Retorna la millor solució trobada al llarg del camí.

    Nota:
    Apliquem el swap utilitzant les funcions de structure.solution (una volta per iteració),
    però la selecció del millor swap es fa amb càlcul incremental eficient.
    """
    sel_initiating_dif = set(initiating_sol['sol']) - set(guiding_sol['sol'])
    sel_guiding_dif = set(guiding_sol['sol']) - set(initiating_sol['sol'])

    r = len(sel_initiating_dif)

    # còpia "lleugera"
    current_sol = {
        'instance': initiating_sol['instance'],
        'sol': set(initiating_sol['sol']),
        'of': initiating_sol['of']
    }

    best_sol_in_path = {
        'instance': current_sol['instance'],
        'sol': set(current_sol['sol']),
        'of': current_sol['of']
    }

    for _ in range(r):
        swap, best_of = findBestSwap(current_sol, sel_initiating_dif, sel_guiding_dif)
        if swap is None:
            break

        i, j = swap

        # aplica swap (una sola volta per iteració)
        solution.removeFromSolution(current_sol, i)
        solution.addToSolution(current_sol, j)

        # per coherència i per a no dependre de recomputacions:
        current_sol['of'] = best_of

        sel_initiating_dif.remove(i)
        sel_guiding_dif.remove(j)

        if current_sol['of'] > best_sol_in_path['of']:
            best_sol_in_path['sol'] = set(current_sol['sol'])
            best_sol_in_path['of'] = current_sol['of']

    return best_sol_in_path
