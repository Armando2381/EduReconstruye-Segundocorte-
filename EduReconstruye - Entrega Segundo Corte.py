"""
EduReconstruye - Entrega Segundo Corte
Implementación de BFS, DFS y A* sobre un grafo experimental.
Eduardo Murcia Castro Codigo 200876 
Armando Ortega Codigo 200626

Consideraciones: 
- El grafo es sintético/demostrativo por lo cual no representa rutas geográficas verificadas.
- BFS minimiza número de aristas si todas tienen el mismo costo.
- DFS no garantiza ruta óptima.
- A* usa h(n) = distancia euclidiana al objetivo. La heurística es admisible
  y consistente porque cada costo de arista es >= a la distancia geométrica
  entre sus extremos y la distancia euclidiana satisface la desigualdad triangular.
"""


from collections import deque
from heapq import heappush, heappop
from math import hypot
from time import perf_counter_ns
from statistics import median

# Coordenadas abstractas (unidades de mapa, NO coordenadas geográficas).
COORDS = {
    "Q": (0, 0),   # Centro de coordinación
    "A": (1, 1),   # Nodo educativo A
    "B": (2, 0),   # Nodo educativo B
    "C": (2, 2),   # Nodo educativo C
    "D": (3, 1),   # Nodo de conexión D
    "E": (4, 0),   # Nodo educativo E
    "F": (4, 2),   # Nodo de conexión F
    "H": (5, 1),   # Nodo de conexión H
    "G": (6, 0),   # Sede priorizada objetivo
}

# Cada arista es bidireccional. El costo incorpora distancia abstracta
# más sobrecostos operativos no negativos (logística, riesgo y disponibilidad).
EDGE_LIST = [
    ("Q", "A", 2.0), ("Q", "B", 5.0),
    ("A", "C", 2.0), ("A", "D", 2.5),
    ("B", "D", 2.0), ("B", "E", 4.0),
    ("C", "D", 1.8), ("C", "F", 2.2),
    ("D", "F", 1.8), ("D", "H", 2.5),
    ("E", "H", 1.5), ("E", "G", 4.0),
    ("F", "H", 1.6), ("H", "G", 1.5),
]

def build_graph():
    graph = {node: [] for node in COORDS}
    for u, v, cost in EDGE_LIST:
        graph[u].append((v, float(cost)))
        graph[v].append((u, float(cost)))
    # Orden estable para que DFS sea reproducible.
    for node in graph:
        graph[node].sort(key=lambda item: item[0])
    return graph

GRAPH = build_graph()
START = "Q"
GOAL = "G"

def path_cost(path, graph=GRAPH):
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += next(cost for nxt, cost in graph[u] if nxt == v)
    return total

def reconstruct(parent, goal):
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return list(reversed(path))

def bfs(start=START, goal=GOAL, graph=GRAPH):
    """Búsqueda en anchura; encuentra ruta con menor número de aristas."""
    frontier = deque([start])
    visited = {start}
    parent = {}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        if node == goal:
            path = reconstruct(parent, goal)
            return {"path": path, "cost": path_cost(path, graph),
                    "expanded": expanded, "found": True}
        expanded += 1
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                frontier.append(neighbor)
    return {"path": [], "cost": float("inf"), "expanded": expanded, "found": False}

def dfs(start=START, goal=GOAL, graph=GRAPH):
    """Búsqueda en profundidad iterativa con control de visitados."""
    stack = [start]
    visited = {start}
    parent = {}
    expanded = 0
    while stack:
        node = stack.pop()
        if node == goal:
            path = reconstruct(parent, goal)
            return {"path": path, "cost": path_cost(path, graph),
                    "expanded": expanded, "found": True}
        expanded += 1
        # Invertir para conservar el orden alfabético al extraer de la pila.
        for neighbor, _ in reversed(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                stack.append(neighbor)
    return {"path": [], "cost": float("inf"), "expanded": expanded, "found": False}

def heuristic(node, goal=GOAL):
    """Distancia euclidiana abstracta: cota inferior del costo restante."""
    x1, y1 = COORDS[node]
    x2, y2 = COORDS[goal]
    return hypot(x2 - x1, y2 - y1)

def astar(start=START, goal=GOAL, graph=GRAPH):
    """A* con reapertura de nodos si se encuentra un g(n) mejor."""
    frontier = []
    counter = 0
    heappush(frontier, (heuristic(start, goal), counter, start))
    g_score = {start: 0.0}
    parent = {}
    closed = set()
    expanded = 0

    while frontier:
        _, _, node = heappop(frontier)
        if node in closed:
            continue
        if node == goal:
            path = reconstruct(parent, goal)
            return {"path": path, "cost": g_score[goal],
                    "expanded": expanded, "found": True}
        closed.add(node)
        expanded += 1

        for neighbor, edge_cost in graph[node]:
            tentative_g = g_score[node] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                parent[neighbor] = node
                if neighbor in closed:
                    closed.remove(neighbor)
                counter += 1
                f_score = tentative_g + heuristic(neighbor, goal)
                heappush(frontier, (f_score, counter, neighbor))

    return {"path": [], "cost": float("inf"), "expanded": expanded, "found": False}

def benchmark(repetitions=500):
    """Mediana de tiempo en ns; las expansiones y el costo son deterministas."""
    rows = []
    for name, algorithm in (("BFS", bfs), ("DFS", dfs), ("A*", astar)):
        result = algorithm()
        samples = []
        for _ in range(repetitions):
            t0 = perf_counter_ns()
            algorithm()
            samples.append(perf_counter_ns() - t0)
        rows.append({
            "algorithm": name,
            "path": " -> ".join(result["path"]),
            "expanded": result["expanded"],
            "cost": round(result["cost"], 3),
            "median_time_us": round(median(samples) / 1000, 2),
            "repetitions": repetitions,
        })
    return rows

if __name__ == "__main__":
    print("EduReconstruye - Fase 2 | Grafo experimental")
    for row in benchmark():
        print(row)