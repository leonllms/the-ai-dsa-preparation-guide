# All Basic problems for data structures

Below is a concise “exercise bank” – one core problem for each of the most‑important data‑structure topics that a software‑engineer should master.  
Each prompt is deliberately kept **minimal** (no hints, no solution, no code‑skeleton) so you can focus on the underlying concept and design the solution yourself.

---

## 1. **Array / Dynamic Array**
**Problem:**  
Given an unsorted integer array `A[ ]` of size *n*, design an algorithm that returns the **k‑th smallest** element in **O(n)** expected time without sorting the whole array.

---

## 2. **String (character array)**
**Problem:**  
Write a function that determines whether a given string `s` is a **permutation of a palindrome** (i.e., can be rearranged to form a palindrome). The solution must run in **O(|s|)** time and use **O(1)** extra space.

---

## 3. **Singly Linked List**
**Problem:**  
Given the head of a singly linked list, remove **all duplicate nodes** so that each value appears only once. The list is **not** sorted. Aim for **O(n)** time and **O(n)** auxiliary space.

---

## 4. **Doubly Linked List**
**Problem:**  
Implement a **LRU (Least Recently Used) cache** with capacity *C* that supports `get(key)` and `put(key,value)` in **O(1)** time. Use a doubly‑linked list together with a hash map.

---

## 5. **Stack**
**Problem:**  
Design a stack that, in addition to `push` and `pop`, supports `getMin()` – returning the minimum element in the stack – all in **O(1)** time and **O(1)** extra space per operation.

---

## 6. **Queue**
**Problem:**  
Implement a **queue** that supports `enqueue`, `dequeue`, and `max` (return the maximum element currently in the queue) all in **amortized O(1)** time.

---

## 7. **Circular Buffer (Ring Queue)**
**Problem:**  
Given a fixed‑size circular buffer of capacity *k*, write the logic to **detect overflow** when trying to enqueue an element and **underflow** when trying to dequeue from an empty buffer.

---

## 8. **Priority Queue / Binary Heap**
**Problem:**  
Given a stream of *n* integers, maintain a **min‑heap** that can return the **median** of all numbers seen so far in **O(log n)** time per insertion.

---

## 9. **Binary Search Tree (BST)**
**Problem:**  
Write a function that, given the root of a BST, returns the **in‑order successor** of a node containing value *x*. Assume each node has a parent pointer.

---

## 10. **Self‑Balancing BST (AVL / Red‑Black)**
**Problem:**  
Insert a sequence of *m* keys into an empty **AVL tree** and report the **height** of the tree after all insertions. The algorithm must maintain AVL balance after each insertion.

---

## 11. **Segment Tree**
**Problem:**  
Given an array of *n* integers, build a segment tree that supports **range sum queries** and **point updates** in **O(log n)** time each.

---

## 12. **Fenwick Tree (Binary Indexed Tree)**
**Problem:**  
Implement a Fenwick tree that can compute the **prefix sum** of the first *i* elements and also support **add(value, index)** updates, both in **O(log n)** time.

---

## 13. **Trie (Prefix Tree)**
**Problem:**  
Given a dictionary of *d* words, build a trie and then write a routine that, for any query string *q*, returns **all words** in the dictionary that have *q* as a prefix.

---

## 14. **Hash Table**
**Problem:**  
Design a hash table that stores **key‑value pairs** with **open addressing** (linear probing). Implement `insert`, `search`, and `delete` operations, handling **clustering** and **rehashing** when the load factor exceeds 0.7.

---

## 15. **Disjoint Set (Union‑Find)**
**Problem:**  
Given *e* pairs of elements representing **equivalence relations**, implement a Union‑Find structure with **path compression** and **union by rank** to answer **connected?** queries in near‑constant amortized time.

---

## 16. **Graph – Adjacency List**
**Problem:**  
Write a function that performs **topological sorting** on a directed acyclic graph (DAG) represented by adjacency lists. Return any valid ordering of the vertices.

---

## 17. **Graph – Adjacency Matrix**
**Problem:**  
Given an undirected weighted graph stored as an adjacency matrix, implement **Prim’s algorithm** to compute the **minimum spanning tree** (MST) in **O(V²)** time.

---

## 18. **Binary Heap (as Priority Queue) – Decrease‑Key**
**Problem:**  
Extend a binary min‑heap to support a `decreaseKey(node, newWeight)` operation that reduces the key of an existing element and restores the heap property in **O(log n)** time.

---

## 19. **Interval Tree**
**Problem:**  
Given a set of *n* intervals \([l_i, r_i]\), build an interval tree that can answer **all intervals overlapping a query interval** \([L,R]\) in **O(log n + k)** time, where *k* is the number of reported intervals.

---

## 20. **Skip List**
**Problem:**  
Implement a skip list that supports **search**, **insert**, and **delete** in expected **O(log n)** time. Include a method to **display** the current levels for debugging.

---

### How to Use This List
1. **Pick a topic** you’re less comfortable with.  
2. **Read the problem statement** carefully – it isolates the core operation you need to implement.  
3. **Sketch the data‑structure layout** (nodes, pointers, arrays, etc.).  
4. **Write the algorithm** on paper or in your IDE, then code it.  
5. **Test** with edge cases (empty structures, single element, maximum capacity, duplicate keys, etc.).  

These exercises cover the foundational data‑structure concepts that appear repeatedly in interviews, system design discussions, and everyday engineering work. Happy coding!