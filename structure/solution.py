def createEmptySolution(instance):
    sol = {}
    sol['instance'] = instance
    sol['sol'] = set()
    sol['of'] = 0.0          # <-- abans era inf
    return sol



def isFeasible(sol):
    return len(sol['sol']) == sol['instance']['p']


def contains(sol, u):
    return u in sol['sol']


def evaluate(sol):
    if len(sol['sol']) < 2:
        return 0.0          # <-- abans era inf

    dmat = sol['instance']['d']
    items = list(sol['sol'])
    best = float("inf")
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            best = min(best, dmat[items[i]][items[j]])
    return best



def addToSolution(sol, u):
    dmat = sol['instance']['d']

    if len(sol['sol']) == 0:
        sol['sol'].add(u)
        sol['of'] = 0.0      # <-- abans era inf
        return

    min_to_sol = float("inf")
    for s in sol['sol']:
        min_to_sol = min(min_to_sol, dmat[u][s])

    # si ja tenies un valor real, mantens el min
    if sol['of'] == 0.0 and len(sol['sol']) == 1:
        sol['of'] = min_to_sol
    else:
        sol['of'] = min(sol['of'], min_to_sol)

    sol['sol'].add(u)



def removeFromSolution(sol, u):
    sol['sol'].remove(u)
    # easiest safe way: recompute of
    sol['of'] = evaluate(sol)


def distanceToSol(sol, u, without=-1):
    """
    For MaxMin constructive heuristics, a natural greedy score is:
    score(u) = min_{s in sol} d(u,s)
    (or +inf if sol empty)
    """
    dmat = sol['instance']['d']
    if len(sol['sol']) == 0:
        return float("inf")

    best = float("inf")
    for s in sol['sol']:
        if s == without:
            continue
        best = min(best, dmat[s][u])
    return best


def printSolution(sol):
    print("Solution:", sorted(sol['sol']))
    print("Objective Value (max-min):", sol['of'])
