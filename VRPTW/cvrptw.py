
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import random, math, os

# ------------------------
# CONFIG
# ------------------------
NUM_CUSTOMERS = 20
NUM_VEHICLES = 3
MAX_ITER = 50000
INITIAL_TEMP = 4000
COOLING_RATE = 0.995
PARALLEL_RUNS = NUM_VEHICLES

# penalties
CAPACITY_PENALTY   = 1000
TIME_PENALTY       = 1000
DUPLICATE_PENALTY  = 20000
MISS_PENALTY       = 2000
IDLE_PENALTY       = 1000
SMOOTHNESS_PENALTY = 0
CROSSING_PENALTY   = 20000

RANDOM_SEED = 91
os.makedirs("graphs", exist_ok=True)

# ------------------------
# DATA GENERATION
# ------------------------
def generate_problem(num_customers, num_vehicles, seed=42):
    np.random.seed(seed)
    customers = pd.DataFrame({
        "CustomerID": range(1, num_customers + 1),
        "x": np.random.randint(0, 100, num_customers),
        "y": np.random.randint(0, 100, num_customers),
        "Demand": np.random.randint(1, 20, num_customers),
        "ReadyTime": np.random.randint(0, 50, num_customers),
        "DueTime": np.random.randint(60, 120, num_customers)
    })
    vehicles = pd.DataFrame({
        "VehicleID": range(1, num_vehicles + 1),
        "Capacity": np.random.randint(50, 100, num_vehicles)
    })
    depot = {"x": 50, "y": 50}

    customers.to_csv("customers.csv", index=False)
    vehicles.to_csv("vehicles.csv", index=False)
    return customers, vehicles, depot

# ------------------------
# GEOMETRY HELPERS
# ------------------------
def distance(p1, p2):
    return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])

def angle_between(p1, p2, p3):
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(c, -1, 1))

def ccw(A,B,C):
    return (C["y"]-A["y"])*(B["x"]-A["x"]) > (B["y"]-A["y"])*(C["x"]-A["x"])

def segments_intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def check_route_crossing(route, customers, depot):
    pts = {"Depot": depot}
    for _, r in customers.iterrows():
        pts[r.CustomerID] = {"x": r.x, "y": r.y}

    penalty = 0
    edges = []

    prev = "Depot"
    for c in route:
        edges.append((prev, c))
        prev = c
    edges.append((prev, "Depot"))

    for i in range(len(edges)):
        A, B = edges[i]
        pA, pB = pts[A], pts[B]
        for j in range(i+1, len(edges)):
            C, D = edges[j]
            pC, pD = pts[C], pts[D]
            if B == C or A == D: continue
            if segments_intersect(pA, pB, pC, pD):
                penalty += CROSSING_PENALTY
    return penalty

# ------------------------
# COST FUNCTION
# ------------------------
def evaluate_cost(solution, customers, vehicles, depot):
    total_distance = 0
    capacity_penalty = 0
    time_penalty = 0
    duplicate_penalty = 0
    miss_penalty = 0
    idle_penalty = 0
    smoothness_penalty = 0
    crossing_penalty = 0

    visited = []

    for v_idx, route in enumerate(solution):
        if len(route) == 0:
            idle_penalty += IDLE_PENALTY
            continue

        cap = vehicles.iloc[v_idx]["Capacity"]
        load = 0
        time = 0
        prev_node = depot

        crossing_penalty += check_route_crossing(route, customers, depot)

        for cust_id in route:
            cust = customers.iloc[cust_id - 1]
            visited.append(cust_id)
            dist = distance(prev_node, cust)
            time += dist
            if time < cust["ReadyTime"]:
                time = cust["ReadyTime"]
            elif time > cust["DueTime"]:
                time_penalty += TIME_PENALTY * (time - cust["DueTime"])
            load += cust["Demand"]
            total_distance += dist
            prev_node = cust

        if load > cap:
            capacity_penalty += CAPACITY_PENALTY * (load - cap)

        total_distance += distance(prev_node, depot)

        if len(route) >= 3:
            for i in range(1, len(route)-1):
                p1 = customers.iloc[route[i-1] - 1]
                p2 = customers.iloc[route[i] - 1]
                p3 = customers.iloc[route[i+1] - 1]
                ang = angle_between(p1,p2,p3)
                if ang < math.pi/2:
                    smoothness_penalty += SMOOTHNESS_PENALTY*(math.pi/2-ang)

    for cid in customers.CustomerID:
        cnt = visited.count(cid)
        if cnt==0:
            miss_penalty += MISS_PENALTY
        elif cnt>1:
            duplicate_penalty += DUPLICATE_PENALTY*(cnt-1)

    return (total_distance + capacity_penalty + time_penalty +
            duplicate_penalty + miss_penalty + idle_penalty +
            smoothness_penalty + crossing_penalty)

# ------------------------
# NEIGHBOR GENERATION
# ------------------------
def generate_neighbor(solution):
    neighbor = [r.copy() for r in solution]
    if random.random() < 0.5:
        r = random.randint(0, len(neighbor)-1)
        if len(neighbor[r])>1:
            i,j = random.sample(range(len(neighbor[r])),2)
            neighbor[r][i], neighbor[r][j] = neighbor[r][j], neighbor[r][i]
    else:
        r1,r2 = random.sample(range(len(neighbor)),2)
        if neighbor[r1]:
            i = random.randint(0,len(neighbor[r1])-1)
            cust = neighbor[r1].pop(i)
            pos = random.randint(0,len(neighbor[r2]))
            neighbor[r2].insert(pos,cust)
    return neighbor

# ------------------------
# SIMULATED ANNEALING
# ------------------------
def sa_run(seed, customers, vehicles, depot):
    random.seed(seed)
    cust_ids = list(customers.CustomerID)
    random.shuffle(cust_ids)
    routes = [[] for _ in range(len(vehicles))]
    for i,cid in enumerate(cust_ids):
        routes[i%len(vehicles)].append(cid)

    current = routes
    current_cost = evaluate_cost(current, customers, vehicles, depot)
    best = [r.copy() for r in current]
    best_cost = current_cost
    temp = INITIAL_TEMP

    for it in tqdm(range(MAX_ITER), desc=f"SA Run {seed}", ncols=100):
        neigh = generate_neighbor(current)
        neigh_cost = evaluate_cost(neigh, customers, vehicles, depot)
        delta = neigh_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta/temp):
            current = neigh
            current_cost = neigh_cost

        if current_cost < best_cost:
            best = [r.copy() for r in current]
            best_cost = current_cost

        temp *= COOLING_RATE

    return best, best_cost

def parallel_sa(customers, vehicles, depot):
    seeds = list(range(PARALLEL_RUNS))
    results = Parallel(n_jobs=-1)(delayed(sa_run)(s, customers, vehicles, depot) for s in seeds)
    return min(results,key=lambda x:x[1])

# ------------------------
# IMPROVEMENT HEURISTIC
# ------------------------
def improvement_heuristic(solution, customers, vehicles, depot, iterations=2000):
    best = [r.copy() for r in solution]
    best_cost = evaluate_cost(best, customers, vehicles, depot)

    for _ in tqdm(range(iterations), desc="Improvement", ncols=100):
        new = [r.copy() for r in best]
        if random.random() < 0.5:
            r = random.randint(0,len(new)-1)
            if len(new[r])>=4:
                i,j = sorted(random.sample(range(len(new[r])),2))
                new[r][i:j] = reversed(new[r][i:j])
        else:
            r1,r2 = random.sample(range(len(new)),2)
            if new[r1] and new[r2]:
                i = random.randint(0,len(new[r1])-1)
                j = random.randint(0,len(new[r2])-1)
                new[r1][i], new[r2][j] = new[r2][j], new[r1][i]

        c = evaluate_cost(new, customers, vehicles, depot)
        if c < best_cost:
            best = [r.copy() for r in new]
            best_cost = c

    return best, best_cost

# ------------------------
# PLOTTING
# ------------------------
def plot_routes(solution, customers, depot, title, filename):
    G = nx.DiGraph()
    pos = {row.CustomerID:(row.x,row.y) for _,row in customers.iterrows()}
    pos["Depot"] = (depot["x"],depot["y"])
    G.add_node("Depot")
    for c in customers.CustomerID:
        G.add_node(c)

    colors = ['red','blue','green','orange','purple','brown','pink','cyan']

    for v_idx,route in enumerate(solution):
        prev = "Depot"
        for c in route:
            G.add_edge(prev,c,color=colors[v_idx%len(colors)])
            prev = c
        G.add_edge(prev,"Depot",color=colors[v_idx%len(colors)])

    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    plt.figure(figsize=(12,8))
    nx.draw(G,pos,with_labels=True,node_size=400,node_color="skyblue",
            edge_color=edge_colors,arrows=True)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__=="__main__":
    customers, vehicles, depot = generate_problem(NUM_CUSTOMERS, NUM_VEHICLES, RANDOM_SEED)
    print("\n--- Parallel SA ---")
    best_sa, cost_sa = parallel_sa(customers, vehicles, depot)
    print("SA Best Cost:", cost_sa)
    plot_routes(best_sa, customers, depot, "After SA", "graphs/sa.png")
    pd.DataFrame([{"Vehicle":v+1,"Customer":cid} for v,r in enumerate(best_sa) for cid in r]).to_csv("routes_sa.csv", index=False)

    print("\n--- Improvement Heuristic ---")
    best_imp, cost_imp = improvement_heuristic(best_sa, customers, vehicles, depot)
    print("Improved Cost:", cost_imp)
    plot_routes(best_imp, customers, depot, "After Improvement", "graphs/improved.png")
    pd.DataFrame([{"Vehicle":v+1,"Customer":cid} for v,r in enumerate(best_imp) for cid in r]).to_csv("routes_improved.csv", index=False)

    print("\nAll outputs saved to CSV and graphs/")
