"""


## 11. Graph Traversal – BFS

**Problem – Shortest Path in Unweighted Graph**  
Given an undirected, unweighted graph `G(V, E)` and two vertices `s` and `t`, find the length (number of edges) of the shortest path from `s` to `t`.  

"""

def shortest_path_undirected(graph , s, t):

    """
    Assume graph as an edge list , i->[j], depicting the vertices you can go to from i
    
    Breadth first search on a graph. 

    Algorithm:

    Start from s, and visit the neighbours, repeat for each neighbour until have found t


    Special cases:

        1. Disconnected components ( starting from s if not found then no path)
        2. Watchout for cycles (don't go to anything visited)
    
    """

    l = 0
    visited = set()
    tovisit = [s]
    level_boundary = s
    while tovisit:

        i = tovisit.pop(0)
        visited.add(i)

        if i == t:
            return l
        else:
            for j in graph[i]:
                if j not in visited and j not in tovisit:
                    tovisit.append(j)

            if i == level_boundary and tovisit:
                level_boundary = tovisit[-1]
                l+=1
    return -1

