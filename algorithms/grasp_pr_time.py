from constructives import cgrasp
from algorithms import prgreedy_good
from localsearch import lsfirstimp
import time
import copy
from itertools import combinations

def updateEliteSet(sol, es_size, elite_set):
    # Simple logic to add to set if not full
    if len(elite_set) < es_size:
        elite_set.append(copy.deepcopy(sol))
        return True

    # Check if better than the worst in the set
    worse_solutions = [s for s in elite_set if s['of'] < sol['of']]
    
    if not worse_solutions:
        return False

    # Find the most similar among the worse ones (Diversity preservation)
    most_similar_sol = None
    max_similarity = -1
    new_sol_set = sol['sol']
    
    for s in worse_solutions:
        similarity = len(new_sol_set.intersection(set(s['sol'])))
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_sol = s
    
    elite_set.remove(most_similar_sol)
    elite_set.append(sol)
    return True

def execute(inst, alpha, es_size=10, time_limit=30, time_doing_grasp = 0.4):
    best = None
    elite_set = []
    iterations = 0
    start_time = time.time()
    
    # --- CONFIGURATION ---
    # Note: If the Elite Set isn't full, we ignore the split and keep building.
    GRASP_TIME_LIMIT = time_limit * time_doing_grasp
    mi = 50 if inst["n"] >= 500 else 200

    # --- PHASE 1: GRASP Construction ---
    print(f"Starting GRASP Phase (Limit: {round(GRASP_TIME_LIMIT, 2)}s)...")
    
    while True:
        elapsed = time.time() - start_time
        
        # STOPPING CRITERIA:
        # 1. If we passed the 70% mark AND we have a full Elite Set -> Switch to PR
        # 2. If we are dangerously close to the absolute total limit (leaving 1s buffer) -> Switch/Stop
        if (elapsed > GRASP_TIME_LIMIT and len(elite_set) >= es_size) or (elapsed > time_limit):
            break

        iterations += 1
        
        # 1. Construct
        sol = cgrasp.construct(inst, alpha)
        
        # 2. Improve (Using lsfast for efficiency)
        lsfirstimp.improve(sol, max_iter=mi)

        
        # 3. Update Elite Set & Best
        updateEliteSet(sol, es_size, elite_set)
        
        if best is None or best['of'] < sol['of']:
            best = copy.deepcopy(sol)
            
        # print(f"Iter {iterations}: {round(sol['of'], 2)}", end="\r")

    # --- PHASE 2: Static Path Relinking ---
    print(f"\nStarting PR Phase with {len(elite_set)} elite solutions...")
    
    # Generate all pairs from the Elite Set
    pairs = list(combinations(elite_set, 2))
    p = 0
    while time.time() - start_time < time_limit and len(pairs) > 0:
        s1, s2 = pairs[p % len(pairs)]
        p += 1

        pr_1 = prgreedy_good.greedyPathRelinking(s1, s2)
        if time.time() - start_time >= time_limit:
            break

        pr_2 = prgreedy_good.greedyPathRelinking(s2, s1)
        path_sol = pr_1 if pr_1['of'] > pr_2['of'] else pr_2

        if time.time() - start_time >= time_limit:
            break

        lsfirstimp.improve(path_sol, max_iter=mi)

        updateEliteSet(path_sol, es_size, elite_set)
        pairs = list(combinations(elite_set, 2))  # refresca pares con el elite nuevo

        if best is None or path_sol['of'] > best['of']:
            best = copy.deepcopy(path_sol)

    return best, iterations