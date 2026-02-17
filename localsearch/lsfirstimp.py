import random

from structure import solution

def improve(sol, max_iter=200):
    improve_flag = True
    it = 0
    while improve_flag and it < max_iter:
        improve_flag = tryImprove(sol)
        it += 1

def tryImprove(sol):
    selected, unselected = createSelectedAndUnselected(sol)
    random.shuffle(selected)
    random.shuffle(unselected)
    for s in selected:
        ds = solution.distanceToSol(sol, s, without=s)
        for u in unselected:
            du = solution.distanceToSol(sol, u, without=s)
            if du > ds:
                solution.removeFromSolution(sol, s)
                solution.addToSolution(sol, u)
                return True
    return False


def createSelectedAndUnselected(sol):
    selected = []
    unselected = []
    n = sol['instance']['n']
    for v in range(n):
        if solution.contains(sol, v):
            selected.append(v)
        else:
            unselected.append(v)
    return selected, unselected