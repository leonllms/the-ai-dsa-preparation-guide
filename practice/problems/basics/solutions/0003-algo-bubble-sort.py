"""

## 3. Sorting

**Problem – In‑Place Bubble Sort**  
Write a function that sorts an integer array `A` of length `n` in non‑decreasing order using the bubble‑sort algorithm.  

"""

"""
Algorithm: 

Bubble sort repeatedly steps through the list, compares each pair of adjacent items and swaps them if they are in the wrong order. After one full pass, the largest (or smallest) element “bubbles” to the end of the list. The process is then repeated for the remaining unsorted portion, stopping when a pass makes no swaps. The algorithm has O(n²) time complexity in the average and worst cases and O(1) extra space.

"""

def bubble_sort(A: list) -> list:
    B = [ x for x in A ] 
    bubble_sort_inplace(B)
    return B


def bubble_sort_inplace(A: list):

    n = len(A)

    def swap(i,j):
        t=A[j]
        A[j] = A[i]
        A[i] = t

    swaps = True

    while swaps:
        swaps=False
        for i in range(n-1):
            if A[i] > A[i+1]:
                swap(i,i+1)
                swaps=True

    return
