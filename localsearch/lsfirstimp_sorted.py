import random

from structure import solution

def improve(sol):
    improve = True
    while improve:
        improve = tryImprove(sol)


def tryImprove(sol):
    selected = createSelected(sol)
    selected = sorted(selected, reverse=True)
    for s in selected:
        unselected = createUnselected(sol, s)
        if unselected:
            u = unselected[random.randint(0, len(unselected)-1)]
            solution.removeFromSolution(sol, s[1], s[0])
            solution.addToSolution(sol, u[1], u[0])
            return True
    return False


def createSelected(sol):
    selected = []
    for v in sol['sol']:
        dv = solution.distanceToSol(sol, v)
        selected.append((dv, v))
    return selected


def createUnselected(sol, s):
    unselected = []
    n = sol['instance']['n']
    for v in range(n):
        if not solution.contains(sol, v):
            dv = solution.distanceToSol(sol, v, without=s[1])
            if dv > s[0]:
                unselected.append((dv, v))
    return unselected