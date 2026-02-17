from structure import solution
import random
import math


def construct(inst, beta):
    """
    GRC2 for Max-Min Diversity:
    - Build RCL2 by selecting ceil(beta * |CL|) candidates uniformly at random from CL
    - Pick the best candidate (max min-distance to current solution) within RCL2
    """
    sol = solution.createEmptySolution(inst)
    n = inst['n']

    # first element randomly
    first = random.randint(0, n - 1)
    solution.addToSolution(sol, first)

    # candidate list: each entry is [score, node]
    # score = min distance to current solution (Max-Min greedy score)
    cl = createCandidateList(sol, first)

    # if beta < 0, random beta in (0,1]
    beta = beta if beta > 0 else random.random()

    while not solution.isFeasible(sol):
        if not cl:
            break  # should not happen unless p > n or something inconsistent

        # size of random filtered list
        q = int(math.ceil(beta * len(cl)))
        q = max(1, min(q, len(cl)))

        # choose q candidates uniformly without replacement
        rcl2 = random.sample(cl, q)

        # greedy choice inside rcl2: max score
        best = max(rcl2, key=lambda x: x[0])  # [score, node]

        # add to solution
        solution.addToSolution(sol, best[1])

        # remove from CL and update remaining scores incrementally
        cl.remove(best)
        updateCandidateList(sol, cl, best[1])

    return sol


def createCandidateList(sol, first):
    n = sol['instance']['n']
    cl = []
    for c in range(n):
        if c != first:
            d = solution.distanceToSol(sol, c)  # should be min-distance to S in your MaxMin solution.py
            cl.append([d, c])
    return cl


def updateCandidateList(sol, cl, added):
    # Max-Min: score(c) = min(score(c), d(added, c))
    dmat = sol['instance']['d']
    for i in range(len(cl)):
        cl[i][0] = min(cl[i][0], dmat[added][cl[i][1]])
