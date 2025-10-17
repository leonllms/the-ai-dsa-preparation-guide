"""

## 1. Searching

**Problem – Linear Search**  
Given an integer array `A` of length `n` and a target value `x`, return the index of `x` in `A` (or `-1` if `x` is not present).  

"""

def linear_search(x, A):
    for i in range(len(A)):
        if A[i] == x:
            return i
    return -1        


