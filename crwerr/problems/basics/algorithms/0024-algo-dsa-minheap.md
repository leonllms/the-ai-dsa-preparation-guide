
# 24. Data Structures – Heap

**Problem – K‑th Largest Element**  
Given an unsorted array `A` of length `n` and an integer `k`, find the `k`‑th largest element using a min‑heap of size `k`.  


# Write the solution briefly to this problem in simple terms without implementing it in code.

**Idea**  
Keep only the k largest numbers seen so far in a min‑heap.  
The smallest element in that heap (its root) is the current k‑th largest.  

**Steps**  

1. Insert the first k elements of the array into a min‑heap.  
2. For each remaining element x in the array:  
   * If x is larger than the heap’s root, remove the root and insert x.  
   * Otherwise ignore x.  
3. After all elements are processed, the root of the heap is the k‑th largest element.

**Why it works**  
The heap always contains the k largest values seen; any value smaller than the smallest of those cannot be the k‑th largest, so it is discarded.

**Complexity**  
Building the initial heap takes O(k).  
Each of the remaining n‑k elements may cause a delete‑and‑insert, each costing O(log k).  
Total time O(n log k), extra space O(k).


## Build a heap

Build a min-heap that can insert, delete, and do all other necessary operations, maintaining the heap property.


**Idea**  
Store the heap in an array so that for a node at index *i* its children are at indices `2*i+1` and `2*i+2` and its parent is at index `(i-1)//2`. This layout lets us move up or down the tree using only index arithmetic.

**Operations**

1. **Insert (push)**  
   * Append the new key at the end of the array.  
   * While the key is smaller than its parent, swap it with the parent ( “bubble‑up” ).  
   * Stop when the heap property (parent ≤ child) holds or the node becomes the root.

2. **Extract‑min (pop)**  
   * The minimum is at index 0. Save it to return.  
   * Move the last element of the array to index 0 and delete the last slot.  
   * Restore the heap by “sifting‑down”: repeatedly swap the element with the smaller of its two children while it is larger than that child.  
   * Stop when the element is ≤ both children or it becomes a leaf.

3. **Delete arbitrary element** (optional)  
   * Replace the element to delete with the last element and shrink the array.  
   * Compare the new value with its parent; if it is smaller, bubble‑up, otherwise sift‑down.  
   * One of these two passes will restore the heap property.

4. **Heapify (build from an unsorted list)**  
   * Place all items into the array.  
   * Starting from the last non‑leaf node (`(n//2)-1`) and moving backwards to index 0, apply the sift‑down operation.  
   * After processing all internal nodes the whole array satisfies the min‑heap property.  

**Complexities**  
* Insert – O(log n) time, O(1) extra space.  
* Extract‑min – O(log n) time, O(1) extra space.  
* Delete – O(log n) time.  
* Build heap – O(n) time (bottom‑up heapify).  

**Why it works**  
The array representation preserves the complete‑binary‑tree shape, so every level is filled left‑to‑right. The bubble‑up and sift‑down steps only move a node along a single root‑to‑leaf path, guaranteeing at most the height of the tree (`⌊log₂ n⌋`) swaps. After each operation the parent‑child ordering (parent ≤ child) holds for every edge, which is exactly the min‑heap property.

```python

class node:
    def __init__(self, key, data=None):
        self.key = key
        self.data = data

    def __lt__(self, other: 'node'):
        return self.key < other.key

    def __le__(self, other: 'node'):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: 'node'):
        return not self.__le__(other)

    def __ge__(self, other: 'node'):
        return not self.__lt__(other)

    def __eq__(self, other: 'node'):
        return self.key == other.key

    def __ne__(self, other: 'node'):
        return not self.__eq__(other)

# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.




class minheap:

    def __init__(self):
        self.heaparray=[]

    def top(self):
        if self.heaparray:
            return self.heaparray[0]
        else:
            return None

    def insert(self,node):
        """

        Insert the element at the bottom of the heap and continuously swap it
        with its parent until it is smaller than its parent. If it has no parent
        it is at the root and we can stop. 

        """

        self.heaparray.append(node)
        n = len(self.heaparray)
        
        cursor = n-1
        pidx = self._parent_idx_(cursor)

        while pidx != -1:
            parent = self.heaparray[pidx]
            if parent > node:
                self._swap_(pidx,cursor)
                cursor = pidx
            else:
                return cursor
            pidx = self._parent_idx_(cursor)
        return cursor


    def pop(self) -> node:

        n = len(self.heaparray)

        # Move last element to root (actually swap)
        self._swap_(0,n-1)

        # Delete last element ( which was the root )
        item = self.heaparray.pop() 

        # shift down until settled
        idx = 0
        n = n-1
        first_idx, second_idx = self._child_idx_(idx)
        while first_idx < n:

            switch_to = None

            cursor = self.heaparray[idx]
            first = self.heaparray[first_idx]

            if cursor > first:
                switch_to = first_idx

            if second_idx < n:
                second = self.heaparray[second_idx]
                if second < first and cursor > second:
                    switch_to = second_idx            
            
            if switch_to:
                self._swap_(idx, switch_to)
                idx = switch_to
                first_idx, second_idx = self._child_idx_(idx)
            else:
                return item

        return item

    def _parent_idx_(self,i):
        if i==0:
            return -1
        if i==1:
            return  0
        return (i-1)//2


    def _child_idx_(self,i):
        return 2*i+1, 2*i+2


    def _swap_(self,i,j):

        """
        Swap node at position i with node at position j
        """

        temp = self.heaparray[i]
        self.heaparray[i] = self.heaparray[j]
        self.heaparray[j] = temp


def k_smallest(numbers, k):

    """

    Fill the minheap with the first k elements 

    Then for each remaining element insert in the heap if it is larger than 
    the root, after removing the root. 

    The root is going to be the k-th largest 

    """

    n = len(numbers)

    if n==0 or k==0:
        return None

    k_heap = minheap()
    for c in range(min(k,n)):
        newnode = node(numbers[c])
        k_heap.insert(newnode)

    newstart = c+1
    for c in range(newstart,n):
        newnode = node(numbers[c])
        root = k_heap.top()

        if newnode > root:
            k_heap.pop()
            k_heap.insert(newnode)

    return k_heap.top()

```

## Gemini-2.5-pro becomes overconfident

# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.


**Errors found**

1. **`k_smallest` – wrong value inserted in the second loop**  
   ```python
   newnode = node(c)          # uses the index instead of the array value
   ```  
   It should be `node(numbers[c])`.

2. **`pop` – incorrect length handling after removal**  
   ```python
   n = len(self.heaparray)          # original length
   ...
   self._swap_(0, n-1)
   item = self.heaparray.pop()
   n = n-1                          # length is reduced again
   ```  
   After `pop()` the heap size is already `n‑1`. Decrementing `n` a second time makes `n` too small and can cause the loop to miss children or index out‑of‑range errors. Use `n = len(self.heaparray)` after the pop (or remove the extra `n = n-1`).

I fixed the first , thank you.

   you are wrong on the second count , n is not updated after the pop . where did you get that ?


   You are correct. My apologies, I misread the flow.

Let's trace your code:
1.  `n = len(self.heaparray)` stores the original size.
2.  `self.heaparray.pop()` shrinks the array.
3.  `n = n-1` then correctly updates the variable `n` to match the new, smaller size of the array.

The logic is sound and there is no error. The variable `n` correctly represents the heap's size within the sift-down loop.


## The previous code had an error

```python

    for c in range(newstart,n):
        newnode = node(c) # instead of having node(numbers[c])
        root = k_heap.top()

```

