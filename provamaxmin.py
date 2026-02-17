from structure import instance
from constructives import cgrasp as grc



def verify_maxmin(sol):
    inst = sol["instance"]
    sel = list(sol["sol"])
    d = inst["d"]

    real = float("inf")
    for i in range(len(sel)):
        for j in range(i+1, len(sel)):
            real = min(real, d[sel[i]][sel[j]])

    if len(sel) < 2:
        real = 0.0

    print("Stored:", sol["of"], "Real:", real, "OK?", abs(sol["of"]-real) < 1e-9)

# test
inst = instance.readInstance("instances/Ran/Ran 500 9.txt")
sol = grc.construct(inst, 0.5)
verify_maxmin(sol)
print("len sol:", len(sol["sol"]), "p:", inst["p"])
