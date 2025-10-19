"""

## 8. Dynamic Programming – Knapsack

**Problem – 0/1 Knapsack (Weight/Value)**  
You have `n` items, each with weight `w[i]` and value `v[i]`, and a knapsack capacity `W`. Compute the maximum total value you can obtain by selecting a subset of items (each item at most once).  

"""


"""

Most basic approach to a applying dynamic programming to a problem, no implementation or generated code, or pseudocode, just words.


**Basic steps for applying dynamic programming**

1. **Identify the sub‑problems** – Break the original problem into smaller, overlapping sub‑problems that can be solved independently.  
2. **Define the state** – Choose a variable (or set of variables) that uniquely represents each sub‑problem.  
3. **Formulate the recurrence** – Express the solution of a sub‑problem in terms of solutions to even smaller sub‑problems.  
4. **Set the base cases** – Determine the simplest sub‑problems whose answers are known directly.  
5. **Choose the computation order** – Decide whether to fill the table bottom‑up (iterative) or top‑down with memoization (recursive).  
6. **Store intermediate results** – Use a table, array, or map to save each sub‑problem’s answer so it is computed only once.  
7. **Extract the final answer** – After all needed sub‑problems are solved, read the result for the original problem from the stored table.


"""


def knapsack_value_max(W: int, weights: list[int],values: list[int], n) -> int:

    """

    Return the maximum total value that you can fit in a knapsack with capacity W.
    Weights and values are provided as equal length lists, where n is the lenght. 

    """ 

    VMAP = {}

    def V(cap):
        ref = VMAP.get(cap, None)
        if ref is None:
            ref = VMAP[cap] = [0]*(n+1)
        return ref

    for c in range(1,W+1):
        for i in range(1,n+1):
            w = weights[i-1]
            v = values[i-1]

            if c-w == 0:
                candidate = v
            elif c-w > 0:
                candidate = V(c-w)[i-1] + v
            else:
                candidate = 0 

            V(c)[i] = max( V(c)[i-1], candidate )

    return V(W)[n]


def knapsack_value_max_so(W: int, weights: list[int],values: list[int], n) -> int:

    """

    Return the maximum total value that you can fit in a knapsack with capacity W.
    Weights and values are provided as equal length lists, where n is the lenght. 

    Space optimized solution

    """ 

    VMAP = {}

    def V(cap):
        ref = VMAP.get(cap, None)
        if ref is None:
            ref = VMAP[cap] = [0,0]
        return ref

    for i in range(1,n+1):
        for c in range(1,W+1):
            w = weights[i-1]
            v = values[i-1]

            if c == w:
                candidate = v
            elif c > w:
                candidate = V(c-w)[0] + v
            else:
                candidate = 0 

            V(c)[1] = max( V(c)[0], candidate )
        
        for c in VMAP.keys():
            V(c)[0] = V(c)[1]            

    return V(W)[1]


def kp(weights, values, maxweight):


    """

    Backtracking to get the indices of the items providing the solution

    """

    n = len(weights)

    V = [[0]*(n+1) for _ in range(maxweight+1)]

    for i in range(1,n+1):

        for c in range(maxweight+1):

            w = weights[i-1]
            v = values[i-1]

            if c>=w:
                V[c][i] = max(V[c-w][i-1]+v, V[c][i-1])
            else:
                V[c][i] = V[c][i-1]


    # Get the indices that produce the solution

    c = maxweight
    i = n
    indices = []
    while i>0 and c>0:

        if V[c][i] != V[c][i-1]:
            indices.append(i-1)
            c -= weights[i-1]
        i -= 1


    return V[maxweight][n] , indices




"""

How to solve the problem : 

**Problem – 0/1 Knapsack (Weight/Value)**  
You have `n` items, each with weight `w[i]` and value `v[i]`, and a knapsack capacity `W`. Compute the maximum total value you can obtain by selecting a subset of items (each item at most once).  

No pseudocode or implementation , just explanation

**Idea – dynamic programming on the prefix of items**

The knapsack problem has two properties that make a DP solution possible.

1. **Optimal‑substructure** –  
   Suppose we already know the best value that can be obtained from the first *i‑1* items when the remaining capacity of the knapsack is *c*.  
   When we consider the *i‑th* item we have only two possibilities:

   * **skip it** – the best value stays the same as for the first *i‑1* items with capacity *c*;
   * **take it** – we must pay its weight *w[i]*, so the remaining capacity becomes *c‑w[i]* (if this is non‑negative) and we can add its value *v[i]* to the best value that can be obtained from the first *i‑1* items with that reduced capacity.

   The optimum for the first *i* items and capacity *c* is the better of those two choices.

2. **Overlapping sub‑problems** –  
   The same sub‑problem “best value using the first *k* items with capacity *x*” appears many times while we examine different items. Storing the result once and re‑using it avoids recomputation.

---

### State definition  

Let  

```
DP[i][c] = maximum total value achievable using only the first i items
           when the knapsack capacity that is still available is c
```

* *i* ranges from 0 … n (0 means “no items considered”).
* *c* ranges from 0 … W (the whole capacity of the knapsack).

The answer we need is `DP[n][W]`.

---

### Recurrence  

For each item *i* (1‑based) and each capacity *c*:

*If the item does not fit* (`w[i] > c`) we cannot take it, therefore  

```
DP[i][c] = DP[i‑1][c]
```

*If the item fits* (`w[i] ≤ c`) we have a choice:

```
DP[i][c] = max( DP[i‑1][c] ,                // skip the item
                DP[i‑1][c‑w[i]] + v[i] )   // take the item
```

The first term is the value when the item is omitted, the second term is the value obtained by putting the item into the knapsack (its value plus the best we can do with the remaining capacity).

---

### Base cases  

* `DP[0][c] = 0` for every capacity *c* – with no items we can obtain no value.
* `DP[i][0] = 0` for every *i* – with zero capacity we cannot put anything.

These initialise the first row and first column of the table.

---

### Computing the table  

We fill the table row by row (or column by column).  
When we are at row *i* we already have row *i‑1* completely computed, so the recurrence can be applied directly.  

The total number of entries is *(n+1)*(W+1)*, and each entry is filled in O(1) time, giving a **time complexity of O(n·W)**.

---

### Space optimisation  

The recurrence only needs the previous row (`i‑1`) to compute the current row (`i`).  
Therefore we can keep a single one‑dimensional array of size *W+1* and update it in decreasing order of capacity (so that the value `DP[c‑w[i]]` still refers to the previous row).  
This reduces the memory consumption to **O(W)** while preserving the O(n·W) running time.

---

### Why greedy does not work  

A greedy rule such as “pick items with highest value‑to‑weight ratio first” can be optimal for the *fractional* knapsack (where items may be split) but fails for the 0/1 version because the decision to include an item is binary and the best combination may require taking a lower‑ratio item to fill the capacity more efficiently. The DP explores all feasible combinations implicitly, guaranteeing optimality.

---

### When the DP over weight is unsuitable  

If the capacity *W* is huge (e.g., up to 10⁹) but the total value of all items is modest, a DP **over total value** is preferable:

*Define* `DP[i][v] = minimum weight needed to achieve total value v using the first i items`.  
The recurrence is analogous, and the table size becomes O(n·Vmax) where Vmax is the sum of all values. The answer is the largest v whose required weight ≤ W.

---

### Alternative exact methods  

* **Branch‑and‑bound** – explores a search tree of decisions, pruning sub‑trees whose best possible value cannot beat the current best. Useful when *n* is moderate (≈30‑40) and capacities are large.
* **Meet‑in‑the‑middle** – split the items into two halves, enumerate all subsets of each half (2^{n/2} each), sort one list by weight, and combine to find the best feasible pair. Works well for *n* up to ~40.

Both are exponential in the worst case, whereas the DP described above is pseudo‑polynomial (polynomial in *n* and *W*).

---

### Summary of the DP solution  

1. **Define** `DP[i][c]` as the best value using the first *i* items with capacity *c*.  
2. **Initialize** the first row/column to 0.  
3. **Iterate** over items *i = 1 … n* and capacities *c = 0 … W*:  
   *If w[i] > c* → copy the previous row value.  
   *Else* → take the maximum of “skip” and “take”.  
4. **Result** is `DP[n][W]`.  
5. **Complexities** – O(n·W) time, O(W) space (with the 1‑D optimisation).  

This method guarantees the maximum total value for the 0/1 knapsack problem.ion


"""