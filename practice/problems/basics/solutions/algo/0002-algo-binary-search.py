"""

## 2. Searching (Sorted)

**Problem – Binary Search**  
Given a **sorted** integer array `A` (ascending) of length `n` and a target value `x`, return the index of `x` (or `-1` if `x` is not present).  

"""

def bsearch_direct(A, x):


    """

    Binary search works by selecting a pivot in the sorted array and then
    checking for equality of the the element at the pivot and returning the
    pivot if they are equal. Otherwise, checks if the pivot element is larger
    or smaller than the target value and selects the left or right partition
    respectively. This is done by tracking the lower and upper boundaries of
    the partition, set as the pivot+1 of the pivot-1 respectively. Once the
    boundaries are set the pivot is computed as the middle integer element of
    the partition and the loop continues from the target value comparison with
    the value at the pivot.

    The loop terminates when the length of the partition is either zero or one,
    containing a single element in which case the final check for equality
    needs be performed.

    """
    n = len(A)

    lb = 0
    ub = n-1
    l = ub-lb+1

    while l>1:
        
        m = lb + l//2

        if x == A[m]:
            return m

        if x < A[m]:
            ub = m-1
        if x > A[m]:
            lb = m+1

        l = ub-lb+1


    if l==1 and A[lb]==x:
        return lb
    else:
        return -1

