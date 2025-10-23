# 31. Dynamic Programming – Edit Distance

**Problem – Minimum Edit (Levenshtein) Distance**  
Given two strings `A` (length `m`) and `B` (length `n`), compute the minimum number of single‑character operations required to transform `A` into `B`. The allowed operations are  

* **Insert** a character  
* **Delete** a character  
* **Replace** a character  

Each operation has a cost of 1. Return the minimum total cost (the edit distance).  

*Typical constraints:* `0 ≤ m, n ≤ 10⁴` (or larger for a challenge).  

*Expected complexity:* `O(m·n)` time and `O(min(m,n))` or `O(m·n)` space using the classic DP formulation.

# Briefly describe the solution in simple words , do not implement.

**Idea in plain language**

Think of the two strings as being written one above the other.  
We want to line them up by possibly inserting, deleting or changing characters so that the upper line becomes exactly the lower line.  
While we scan the strings from left to right we keep, for every prefix of the two strings, the cheapest way to transform one prefix into the other.  
The cheapest cost for a longer prefix can be built from the cheapest costs of the three smaller prefixes that precede it – exactly the three operations we are allowed to use.

---

### 1. Sub‑problems

Let  

*`dp[i][j]`* = minimum number of operations needed to turn the first **i** characters of `A` into the first **j** characters of `B`.

The answer we need is `dp[m][n]` (the whole strings).

---

### 2. Recurrence (how a bigger sub‑problem is solved)

* If the last characters already match (`A[i‑1] == B[j‑1]`) we do not need any extra work, so  

```
dp[i][j] = dp[i‑1][j‑1]
```

* Otherwise we have three possibilities, each costing one extra operation:

| Operation | What it does | Resulting sub‑problem |
|-----------|--------------|----------------------|
| **Insert** a character into `A` (so that it matches `B[j‑1]`) | we keep the same `i` characters of `A` and have already built the first `j‑1` characters of `B` | `dp[i][j‑1] + 1` |
| **Delete** a character from `A` | we drop `A[i‑1]` and still need to build the first `j` characters of `B` | `dp[i‑1][j] + 1` |
| **Replace** `A[i‑1]` by `B[j‑1]` | we consume both characters and pay one replace | `dp[i‑1][j‑1] + 1` |

We take the cheapest of the three:

```
dp[i][j] = 1 + min( dp[i‑1][j] ,          // delete
                    dp[i][j‑1] ,          // insert
                    dp[i‑1][j‑1] )        // replace
```

If the characters are equal we simply use `dp[i‑1][j‑1]` (no +1).

---

### 3. Base cases

* Transforming an empty prefix of `A` into a prefix of `B` needs only insertions:

```
dp[0][j] = j      (j characters must be inserted)
```

* Transforming a prefix of `A` into an empty prefix of `B` needs only deletions:

```
dp[i][0] = i      (i characters must be deleted)
```

---

### 4. Order of filling

We fill the table row‑by‑row (or column‑by‑column) because each entry depends only on the cell directly above, the cell to the left, and the diagonal‑up‑left cell, all of which are already known when we move forward.

The total work is the number of cells, i.e. `m·n`, giving **O(m·n) time**.

---

### 5. Reducing memory

Only the previous row (or previous column) is needed to compute the current row, because the recurrence never looks farther back than one step.  
Thus we can keep two 1‑dimensional arrays of length `min(m,n)+1` and swap them after each row, achieving **O(min(m,n)) space**.

---

### 6. Result

After the table (or the rolling arrays) is filled, the entry that corresponds to the full lengths, `dp[m][n]`, holds the minimum number of insert, delete and replace operations required – the Levenshtein edit distance.

---


```python 

def levenshtein(alpha,beta):

    """
    
    Subproblem : Given prefix of the string alpha up to index i and prefix of 
    string beta up to j there are number edits[i][j] to go from alpha to beta. 

    Increment: Considering the next character in alpha here is the logic:

    Insert: We can insert a character in A to make it equal to prefix of B
    Delete: We can remove a character from A 
    Replace: We can replace a character in A from a character in B

    """

    n = len(alpha)
    m = len(beta)

    edits = [[0]*(m+1) for _ in range(n+1)]
    edits[0] = [ j for j in range(0,m+1) ]
    for i in range(1,n+1):
        edits[i][0] = i
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            if alpha[i-1] == beta[j-1]:
                edits[i][j] = edits[i-1][j-1]
            else:
                edits[i][j] = 1 + min(
                    edits[i-1][j],
                    edits[i-1][j-1],
                    edits[i][j-1]
                    )

    return edits[-1][-1]


```
