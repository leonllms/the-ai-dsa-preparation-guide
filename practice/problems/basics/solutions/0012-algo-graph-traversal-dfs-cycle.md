# Graph traversal: Cycle detection in DAG

## 12. Graph Traversal – DFS

**Problem – Detect Cycle in Directed Graph**  
Given a directed graph `G(V, E)`, determine whether it contains a cycle.  

```python
def cycle_detections_gtdfs(graph):

    """

    Algorithm:
    
    For each vertext in the graph and perform a DFS keeping a list of 
    visited nodes, to-visit nodes, those in the current path, and the number of nodes 
    per level. 

    Specifically:

    1 - Take a vertex from the graph and put it in the to-visit list. 
    2 - Loop in reverse over the levels and remove all levels and path nodes
        that have level count zero, stopping at the first non-zero.

    3 - Take the last added node from the tovisit list
    4 - Since we took one node from the tovisit, in this level there is
        one less, so reduce the last element of levels ( check first for 
        proper initialization )
    5 - Add it to the visited , and to the path, and append a level,
        base count 0
    6 - explore the visited node, by adding all its neighbours if they are
        nor visited nor in path in the tovisit list. If some is in path terminate. Set the current
        level to the number of neighbours added in tovisit.
    7 - Repeat from step 2

    """

    visited = set()

    for vertex in graph.keys():
        if vertex not in visited:
            tovisit = [vertex]
            levels = []
            path = []

            while tovisit:

                while levels and levels[-1]==0:
                    path.pop()
                    levels.pop()

                if levels:
                    levels[-1] -= 1


                current = tovisit.pop()
                visited.add(current)

                path.append(current)
                levels.append(0)

                ref = graph.get(current, [])
                for neighbour in ref:
                    if neighbour in path:
                        return True

                    if neighbour not in visited:
                        tovisit.append(neighbour)
                        levels[-1]+=1

    return False


```
## Simplified algorithm description

    Algorithm:

    For every vertex in the graph:

    Check if it is visited other wise add it in the tovisit list.

    Process: 
    For every node in the tovisit list:

    Backtrack: 
    To the highest non-zero level
    
    Visit:
    Take the last added element from the tovisit list, and add it in visited,
    in path, and add a new level with zero count of nodes in that level.

    Explore the node:

    Count and add all its neighbours in the level and tovisit . If any of these
    is in the path then we have found a cycle and we can terminate and return true

    If there are no more vertices to process no cycle has been detected.


# AI suggestions

## 2.  School‑book implementations

Below are two classic ways to detect a cycle in a directed graph.  
Both are easy to read, fully correct, and use the same asymptotic bounds.

### 2.1  Recursive DFS (colour/white‑gray‑black method)

```python
def has_cycle_recursive(graph):
    """
    Returns True iff the directed graph contains a cycle.
    Uses the classic 3‑colour DFS:
        WHITE – not visited yet
        GRAY  – currently on the recursion stack
        BLACK – completely processed
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = {v: WHITE for v in graph}

    def dfs(v):
        colour[v] = GRAY                     # vertex entered
        for nb in graph.get(v, []):
            if colour[nb] == GRAY:           # back‑edge → cycle
                return True
            if colour[nb] == WHITE:
                if dfs(nb):
                    return True
        colour[v] = BLACK                    # vertex left
        return False

    for vertex in graph:
        if colour[vertex] == WHITE:
            if dfs(vertex):
                return True
    return False
```

*Complexity*: `O(V+E)` time, `O(V)` recursion depth (stack) + colour table.

---

### 2.2  Iterative DFS with an explicit stack

```python
def has_cycle_iterative(graph):
    """
    Detects a cycle using an explicit stack (no recursion).
    The stack stores pairs (vertex, iterator over its neighbours)
    and a separate set `on_stack` acts as the recursion‑stack marker.
    """
    visited   = set()
    on_stack  = set()          # vertices currently on the DFS path

    for start in graph:
        if start in visited:
            continue

        stack = [(start, iter(graph.get(start, [])))]
        visited.add(start)
        on_stack.add(start)

        while stack:
            v, nbrs = stack[-1]            # look at the top of the stack
            try:
                w = next(nbrs)             # next neighbour of v
                if w not in visited:
                    visited.add(w)
                    on_stack.add(w)
                    stack.append((w, iter(graph.get(w, []))))
                elif w in on_stack:       # back‑edge
                    return True
                # else: w already finished → ignore
            except StopIteration:
                # all neighbours processed → pop v
                stack.pop()
                on_stack.remove(v)

    return False
```


*Complexity*: `O(V+E)` time, `O(V)` auxiliary space (the `stack` and the two sets).

---

## 4.  Quick sanity‑check

```python
if __name__ == "__main__":
    # 1️⃣  acyclic graph (diamond)
    G1 = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: []
    }

    # 2️⃣  simple cycle
    G2 = {
        0: [1],
        1: [2],
        2: [0]
    }

    # 3️⃣  two components, one cyclic, one acyclic
    G3 = {
        0: [1],
        1: [2],
        2: [],         # acyclic component
        3: [4],
        4: [5],
        5: [3]         # cycle component
    }

    for f in (cycle_detection_gtdfs,
              has_cycle_recursive,
              has_cycle_iterative):
        print(f.__name__, "G1:", f(G1), "G2:", f(G2), "G3:", f(G3))
```

All three functions print

```
... G1: False G2: True G3: True
```

confirming that the corrected minimal‑change version and the two textbook versions behave identically.


