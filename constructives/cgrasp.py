from structure import solution
import random


def construct(inst, alpha):
    sol = solution.createEmptySolution(inst)
    n = inst['n']

    u = random.randint(0, n-1)
    solution.addToSolution(sol, u)

    cl = createCandidateList(sol, u)
    alpha = alpha if alpha >= 0 else random.random()

    while not solution.isFeasible(sol):


        if not cl:
            raise RuntimeError(
                f"CL vacía antes de ser factible: n={inst['n']} p={inst['p']} |sol|={len(sol['sol'])}"
            )

        gmin, gmax = evalGMinGMax(cl)
        threshold = gmax - alpha * (gmax - gmin)


        rcl = [c for c in cl if c[0] >= threshold - 1e-12]


        if not rcl:
            rcl = cl[:]

        # elegir candidato
        cSel = random.choice(rcl)

        # añadir y actualizar
        solution.addToSolution(sol, cSel[1])
        cl.remove(cSel)
        updateCandidateList(sol, cl, cSel[1])

    return sol



def evalGMinGMax(cl):
    gmin = 0x3f3f3f
    gmax = 0
    for c in cl:
        gmin = min(gmin, c[0])
        gmax = max(gmax, c[0])
    return gmin, gmax


def createCandidateList(sol, first):
    n = sol['instance']['n']
    cl = []
    for c in range(n):
        if c != first:
            d = solution.distanceToSol(sol, c)
            cl.append([d, c])
    return cl


def updateCandidateList(sol, cl, added):
    for i in range(len(cl)):
        c = cl[i]
        c[0] = min(c[0], sol['instance']['d'][added][c[1]])

