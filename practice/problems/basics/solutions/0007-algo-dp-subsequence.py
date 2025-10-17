"""

## 7. Dynamic Programming – Subsequence

**Problem – Longest Increasing Subsequence (LIS)**  
Given an integer array `A` of length `n`, find the length of the longest strictly increasing subsequence.  

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




def longest_increasing_subsequence(A:list[int]) -> int:


    """

    Each prefix ending at index i has a maximum strictly increasing subsequence . 
    The new maximum can be computed by checking if the new element is larger than any element j in the prefix. 
    Checking for that element the LIS[j] and among all candidates taking the max.

    """

    n = len(A)
    LIS = [0]*n

    for i in range(n):
            candidates = [ LIS[j]+1 for j in range(i) if A[i]>A[j] ]
            if candidates:
                LIS[i] = max(candidates)
            else:
                LIS[i] = 1 # If smaller than all other elements, then new subsequence may start here

    return max(LIS)



"""

How to solve the longest strictly increasing subsequence, no pseudocode or implementation , just explanation


**Idea behind the problem**  
You are given a sequence of numbers. A *subsequence* is obtained by deleting some elements without changing the order of the remaining ones.  
The task is to find the longest subsequence whose elements are strictly increasing from left to right.

---

### 1.  Dynamic‑programming view ( O(n²) )

For each position *i* in the original array consider the best increasing subsequence that **ends** at that element.  
If we already know the best length for every earlier position *j* ( j < i ), we can extend any of those subsequences whose last element is smaller than the element at *i*.  
The longest such extension gives the optimal length for position *i*.

*Why it works* – any optimal increasing subsequence must end at some index *i*. Removing the last element leaves an optimal subsequence that ends at a previous index *j* and satisfies the same “last‑value‑smaller‑than‑current” condition. Therefore the recurrence captures all possibilities.

*How to obtain the answer* – after processing all indices, the maximum of the stored lengths is the length of the longest increasing subsequence. To reconstruct the actual subsequence, keep a back‑pointer from each index to the previous index that gave the best extension, then follow the pointers from the index that achieved the maximum length.

*Complexity* – two nested loops over the n positions give O(n²) time, and O(n) extra space for the length and back‑pointer arrays.


What i don't understand is what are conditions when i am at index i to compute the value wrt to previously computed values


When you reach position **i** you already know, for every earlier position **j** ( 0 ≤ j < i ), the length of the longest increasing subsequence that ends at **j**.  
To compute the value for **i** you only have to look at those earlier positions whose element is **smaller** than the element at **i**:

```
consider every j < i
    if  a[j] < a[i]          // the subsequence can be continued
        candidate length = length[j] + 1
```

Among all candidates you keep the largest one; if no such **j** exists the subsequence consisting of the single element a[i] is the best, so its length is 1.

Formally  

```
len[i] = 1 + max{ len[j] | j < i and a[j] < a[i] }   if the set is non‑empty
len[i] = 1                                            otherwise
```

The condition `a[j] < a[i]` guarantees strict increase, and `j < i` guarantees that the original order is preserved.  

After processing every index, the overall answer is `max_i len[i]`.  
If you also store, for each **i**, the index **j** that gave the maximal candidate, you can later follow those back‑pointers from the index that achieved the global maximum to reconstruct the actual subsequence.


---

### 2.  “Patience sorting” view ( O(n log n) )

Imagine playing a solitaire game with piles of cards. Scan the input from left to right and maintain a set of piles, each pile holding a single top card. The rule for placing a new card (value *x*) is:

* Find the leftmost pile whose top card is **greater than** *x* and put *x* on top of that pile (replacing the previous top).  
* If no such pile exists, start a new pile on the right.

Because piles are kept in increasing order of their top cards, the search for the proper pile can be done with binary search, giving O(log n) per element.

*Why the piles give the answer* – the number of piles after the whole scan equals the length of the longest strictly increasing subsequence. This follows from a classic theorem: the minimal number of piles needed in this game is exactly the length of a longest increasing subsequence, while the greedy placement never creates more piles than necessary.

*Recovering an actual subsequence* – while scanning, store for each element the index of the pile it was placed on and a pointer to the element that was the top of the previous pile (the predecessor in an increasing subsequence). After the scan, start from any element that ended on the last pile and follow the predecessor links backwards; the collected elements, read in reverse order, form a longest increasing subsequence.

*Complexity* – each element requires a binary search among at most n piles, so O(n log n) time, and O(n) space for the predecessor information.

---

### 3.  Choosing a method

* If the input size is modest (a few thousand) the O(n²) dynamic‑programming approach is simple to understand and implement.  
* For large inputs (tens of thousands or more) the O(n log n) patience‑sorting method is preferred because it scales much better while still allowing reconstruction of an optimal subsequence.

Both approaches rely on the same fundamental observation: an optimal increasing subsequence can be built by extending a shorter optimal subsequence that ends earlier, and the greedy pile‑building rule captures the necessary ordering information efficiently.

"""