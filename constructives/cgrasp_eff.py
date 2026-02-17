from structure import solution
import random

def construct(inst, alpha):
    sol = solution.createEmptySolution(inst)
    n = inst['n']
    u = random.randint(0, n-1)
    solution.addToSolution(sol, u)
    cl, gmin, gmax = createCandidateList(sol, u)
    rcl = [0] * n
    alpha = alpha if alpha >= 0 else random.random()
    while not solution.isFeasible(sol):
        threshold = gmax - alpha * (gmax - gmin)
        limit = 0
        for i in range(len(cl)):
            if cl[i][0] >= threshold:
                rcl[limit] = i
                limit += 1
        selIdx = random.randint(0, limit-1)
        cSel = cl[rcl[selIdx]]
        solution.addToSolution(sol, cSel[1], cSel[0])
        del cl[rcl[selIdx]]
        gmin, gmax = updateCandidateList(sol, cl, cSel[1])
    return sol


def createCandidateList(sol, first):
    n = sol['instance']['n']
    cl = []
    gmin = 0x3f3f3f
    gmax = 0
    for c in range(n):
        if c != first:
            d = solution.distanceToSol(sol, c)
            cl.append([d, c])
            gmin = min(gmin, d)
            gmax = max(gmax, d)
    return cl, gmin, gmax


def updateCandidateList(sol, cl, added):
    gmin = 0x3f3f3f
    gmax = 0
    for i in range(len(cl)):
        c = cl[i]
        c[0] += sol['instance']['d'][added][c[1]]
        gmin = min(gmin, c[0])
        gmax = max(gmax, c[0])
    return gmin, gmax

