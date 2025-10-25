# 17. String Matching – Naïve

**Problem – Find All Occurrences**  
Given a pattern string `P` and a text string `T`, output all starting indices in `T` where `P` occurs (allow overlapping matches).  

Outline in simple words how to solve this problem, and do not provide an implementation.


**Naïve string‑matching outline**

1. **Read the two strings**  
   *Pattern* `P` of length `m` and *text* `T` of length `n`.

2. **Loop over every possible start position** in the text where the pattern could fit:  
   for each index `i` from `0` to `n‑m` (inclusive) do the following.

3. **Compare the characters one by one**  
   * Set a flag `match = true`.  
   * For `j = 0 … m‑1` compare `P[j]` with `T[i + j]`.  
   * If any pair differs, set `match = false` and break out of the inner loop.

4. **Record a hit**  
   *If `match` stayed true after the inner loop, the pattern occurs at position `i`.  
   Store (or output) `i`.*

5. **Continue** the outer loop until all start positions have been examined.  
   Overlapping occurrences are automatically handled because the loop advances only by one position each time.

6. **Finish** by outputting the list of recorded indices (or an empty list if none were found).

**Why it works** – The algorithm checks every possible alignment of `P` against `T`. Whenever all `m` characters line up, the alignment is a genuine occurrence, so the corresponding index is reported.  

**Complexity** – In the worst case the inner comparison runs `m` times for each of the `n‑m+1` start positions, giving a time bound of `O(n·m)`. The algorithm uses only a few extra variables, so the space usage is `O(1)` besides the output list.

```python

def match_string(source_str: str, match_str: str) -> int :

    n = len(source_str)
    m = len(match_str)

    if n == 0 or m == 0:
        return -1


    for i in range(n):

        j = 0

        while j<m and i+j<n and source_str[i+j] == match_str[j]:
            j += 1

        if j==m:
            return i

    return -1


```

