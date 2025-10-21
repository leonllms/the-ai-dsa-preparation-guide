## 13. Shortest Path – Dijkstra

**Problem – Single‑Source Shortest Paths (Positive Weights)**  
Given a directed graph with non‑negative edge weights and a source vertex `s`, compute the minimum distance from `s` to every other vertex.


Outline of the algorithm in steps to solve the problem :

**Dijkstra’s algorithm – step by step**

1. **Input**  
   * Directed graph `G = (V , E)`  
   * Non‑negative weight function `w : E → ℝ≥0`  
   * Source vertex `s ∈ V`

2. **Data structures**  
   * `dist[v]` – current best distance from `s` to `v` (array or map)  
   * `prev[v]` – predecessor of `v` on the current shortest‑path tree (optional, for path reconstruction)  
   * A **min‑priority queue** `Q` that stores vertices keyed by `dist`.  
   * A boolean set `visited` (or a flag inside `dist`) to mark vertices whose final distance is known.

3. **Initialisation**  
   ```
   for each vertex v in V:
       dist[v] = ∞
       prev[v] = NIL
   dist[s] = 0
   insert s into Q with key 0
   ```

4. **Main loop** – repeat while the queue is not empty  

   a. **Extract‑min**  
      ```
      u = vertex in Q with smallest dist[u]
      remove u from Q
      ```
      At this point `dist[u]` is the final shortest‑path distance from `s` to `u`.

   b. **Mark u as visited** (optional, because once extracted it will never be inserted again).

   c. **Relax all outgoing edges of u**  
      For each edge `(u , v) ∈ E` :

      ```
      if dist[u] + w(u,v) < dist[v]:
          dist[v] = dist[u] + w(u,v)
          prev[v] = u
          if v is already in Q:
               decrease‑key of v to new dist[v]
          else:
               insert v into Q with key dist[v]
      ```

5. **Termination**  
   When `Q` becomes empty, `dist[v]` holds the length of the shortest path from `s` to every vertex `v`.  
   If a vertex never entered the queue, its distance remains `∞` (unreachable).

6. **Optional – retrieve a path**  
   To obtain the actual path from `s` to a target `t`, follow `prev` pointers backwards:

   ```
   path = empty list
   while t ≠ NIL:
        prepend t to path
        t = prev[t]
   ```

7. **Complexity**  
   * Using a binary heap for `Q`: `O((|V|+|E|) log |V|)` time, `O(|V|)` space.  
   * With a Fibonacci heap the time improves to `O(|E| + |V| log |V|)`.

8. **Correctness sketch**  
   The algorithm maintains the invariant that every vertex removed from the queue has its final shortest‑path distance, because all edge weights are non‑negative. Relaxation never overestimates a distance, and the min‑priority queue always selects the currently closest unreached vertex, guaranteeing optimality by induction on the number of extracted vertices.


## Naive implementation without using a priority queue


```python3

def shortest_path_weighted_graph(graph, weights, s):

    """
    Algorithm: Maintain a list of distances from the source vertex `s` to all
    other vertices, initially set to infinity except for the source which is 0.
    Also maintain a set or list of visited nodes. While there are unvisited
    nodes, iteratively select and remove the unvisited node with the smallest
    known distance. For each neighbor of this node, calculate the tentative
    distance from `s` through the current node. If this tentative distance is
    less than the currently recorded distance, update the distance. Mark the
    processed node as visited. Repeat until all reachable nodes are visited.
    The distance list then contains the shortest path distances from `

    """

    infinity = 100000000
    dist = {}

    # Go over all vertices and set everything to infinity except source, adding
    # all to unvisited set.
    for v, edges in graph.items():
        dist[v] = infinity if v != s else 0
        for e in edges:
            dist[e] = infinity if e != s else 0

    def minval_idx(prio: dict):
        minval = infinity
        for k,v in prio.items():
            if v < minval:
                minval = v 
                idx = k
        return idx

    unvisited = {**dist}

    def relax(i):
        edges = graph[i]
        for j in edges:
            w_ij = weights[(i,j)]
            dist[j] = min(dist[j], dist[i]+w_ij)
            unvisited[j] = dist[j]

    while unvisited:
        # Get the minimum value index and remove it from the unvisited
        top = minval_idx(unvisited)

        # If it has outgoing connections then relax it, visiting each 
        # connections and updating the distance if it is smaller
        if top in graph:
            relax(top)

        del unvisited[top]

    return dist
```


## Naive implementation assuming a priority Queue


```python

class heap(ABC):

    @abstractmethod
    def heapify(self):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def push(self, kv_pair):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    def __contains__(self, item):
        pass



def shortest_path_weighted_graph_heapq(graph, weights, s):

    infinity = 100000000

    def init(distances: dict, priorities: heap):
        
        # Go over all edges and vertices to find all the vertices in the graph
        # and initialize the list of distances and the priority queue
        for v, edges in graph.items():
            toadd = [v,*edges]
            for e in toadd:
                if e not in distances:
                    priorities.push((infinity, e))
                    distances[e] = infinity

        # Set the distance of the source vertex to zero (ofc)
        distances[s] = 0
        priorities.push((0,s))

    dist = dict()
    prio = heap()
    
    init(dist, prio)
    
    prio.heapify()

    def relax(i):
        """
        For a provided vertex relax the edges:

        Go over all its neighbours and recompute their distance for path
        going through vertex i. Store the new distance if it is less than 
        the current.

        """

        connections = graph[i]
        for j in connections:
            w_ij = weights[(i,j)]
            dist[j] = min(dist[i]+w_ij,dist[j])
            if j in prio:
                prio.set(j,dist[j])

    while prio:
        dist, top = prio.pop() # Assume pops the top ( least distance )
        if top in graph:
            relax(top)
        prio.heapify()

    return dist

```

