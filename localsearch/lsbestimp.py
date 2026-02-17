from structure import solution

def improve(sol, max_iter=50):
    improve = True
    it = 0
    while improve and it < max_iter:
        improve = tryImprove(sol)
        it += 1



def tryImprove(sol):
    sel, ofVarSel, unsel, ofVarUnsel = selectInterchange(sol)
    if ofVarSel < ofVarUnsel:
        solution.removeFromSolution(sol, sel)
        solution.addToSolution(sol, unsel)
        return True
    return False


def selectInterchange(sol):
    n = sol['instance']['n']
    sel = -1
    bestSel = 0x3f3f3f
    unsel = -1
    bestUnsel = 0
    for v in sol['sol']:
        d = solution.distanceToSol(sol, v,without=v)
        if d < bestSel:
            bestSel = d
            sel = v
    for v in range(n):
        d = solution.distanceToSol(sol, v, without=sel)
        if not solution.contains(sol, v):
            if d > bestUnsel:
                bestUnsel = d
                unsel = v
    return sel, bestSel, unsel, bestUnsel