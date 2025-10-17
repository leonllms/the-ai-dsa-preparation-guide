"""
## 4. Sorting (Divide & Conquer)

**Problem – Merge Sort**  
Implement merge sort to sort an integer array `A` of length `n` in non‑decreasing order.  

"""

"""
Description of the merge sort algorithm:

Merge sort is a divide‑and‑conquer sorting method.  
1. **Divide**: Split the input list into two halves of (roughly) equal size.  
2. **Conquer**: Recursively apply merge sort to each half until the sub‑lists contain a single element (or are empty), which are trivially sorted.  
3. **Combine**: Merge the two sorted halves into a single sorted list by repeatedly taking the smallest remaining element from the fronts of the two halves.  

The merge step runs in linear time, and the recursion depth is logarithmic, giving an overall time complexity of O(n log n). The algorithm is stable and requires O(n) additional auxiliary space for the merging process.

"""

def merge(a: list, b: list) -> list:
    n = len(a)
    m = len(b)

    o = [None]*(m+n)

    i=0
    j=0
    k=0
    while i<n and j<m:

        if a[i] == b[j]:
            o[k]=a[i]
            k+=1
            o[k]=b[j]
            i+=1
            j+=1
        elif a[i] < b[j]:
            o[k]=a[i]
            i+=1
        else:
            o[k]=b[j]
            j+=1

        k+=1

    while i<n:
        o[k]=a[i]
        i+=1
        k+=1

    while j<m:
        o[k]=b[j]
        j+=1
        k+=1

    return o



def merge_sort(A):

    n = len(A)

    if n==0:
        return

    if n==1:
        return

    pivot = n//2

    leftpart = A[0:pivot]
    rightpart= A[pivot:n]

    merge_sort(leftpart)
    merge_sort(rightpart)
    
    merged = merge(leftpart,rightpart)

    for i in range(n):
        A[i] = merged[i]

    return



