"""
## 5. Sorting (Linear Time)

**Problem – Counting Sort**  
Given an integer array `A` where each element is in the range `[0, k)`, sort `A` in `O(n + k)` time.  

"""

"""

Go over the array and count how many times each integer is found keeping an 0,k index based array.
Then go over the index and overwrite the original array with the index as many times it has been counted.

"""


def count_sort(A:list[int], k: int):

    n = len(A)
    lookup = [0]*k

    for i in range(n):
        lookup[A[i]] += 1

    pos=0
    for num in range(k):
        j=0
        while j < lookup[num]:
            A[pos]=num
            pos+=1
            j+=1

    return

