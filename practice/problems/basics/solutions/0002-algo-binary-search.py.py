"""

## 2. Searching (Sorted)

**Problem – Binary Search**  
Given a **sorted** integer array `A` (ascending) of length `n` and a target value `x`, return the index of `x` (or `-1` if `x` is not present).  

"""

def binary_search(x,A):

    n = len(A)

    if n==0:
        return -1

    lb=0
    ub=n-1

    while ub>=lb:
        pivot=(ub+lb)//2

        e=A[pivot]

        if e==x:
            return pivot
        elif e<x:
            lb=pivot+1
        else:
            ub=pivot-1

    return -1

