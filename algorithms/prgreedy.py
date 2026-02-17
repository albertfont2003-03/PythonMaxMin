


def _nn_in_solution(sol):
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

            if dist < nn1[a]:
                nn2[a] = nn1[a]
                nn1[a] = dist
                arg1[a] = b
            elif dist < nn2[a]:
                nn2[a] = dist

            if dist < nn1[b]:
                nn2[b] = nn1[b]
                nn1[b] = dist
                arg1[b] = a
            elif dist < nn2[b]:
                nn2[b] = dist

    return nn1, nn2, arg1


def _best_two_to_S(sol, candidates):
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
    of_without = {}
    size_after = len(current['sol']) - 1
    for i in A:
        if size_after < 2:
            of_without[i] = 0.0
            continue

        best = float('inf')
        for s in current['sol']:
            if s == i:
                continue
            val = nn1[s] if argS[s] != i else nn2[s]
            if val < best:
                best = val
        of_without[i] = best
    return of_without


def findBestSwap_fast(current, A, B, of_without, b1, b2, argJ):
    best_of = -float('inf')
    best_swap = None

    for i in A:
        base = of_without[i]
        for j in B:
            dist_to_S_wo_i = b1[j] if argJ[j] != i else b2[j]
            cand_of = min(base, dist_to_S_wo_i)

            if cand_of > best_of:
                best_of = cand_of
                best_swap = (i, j)

    return best_swap, best_of


def greedyPathRelinking_fast(initiating_sol, guiding_sol):
    A = set(initiating_sol['sol']) - set(guiding_sol['sol'])
    B = set(guiding_sol['sol']) - set(initiating_sol['sol'])
    r = len(A)

    current = {
        'instance': initiating_sol['instance'],
        'sol': set(initiating_sol['sol']),
        'of': initiating_sol['of']
    }

    best_in_path = {
        'instance': current['instance'],
        'sol': set(current['sol']),
        'of': current['of']
    }

    for _ in range(r):
        nn1, nn2, argS = _nn_in_solution(current)
        of_without = _compute_of_without(current, A, nn1, nn2, argS)

        b1, b2, argJ = _best_two_to_S(current, B)

        swap, best_of = findBestSwap_fast(current, A, B, of_without, b1, b2, argJ)
        if swap is None:
            break

        i, j = swap

        # aplica swap (sense recomputar evaluate)
        current['sol'].remove(i)
        current['sol'].add(j)
        current['of'] = best_of

        A.remove(i)
        B.remove(j)

        if current['of'] > best_in_path['of']:
            best_in_path['sol'] = set(current['sol'])
            best_in_path['of'] = current['of']

    return best_in_path

