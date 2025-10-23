# All Basic problems for algorithms


Below is a curated set of **exercises** – one for each of the classic algorithmic problem types that every software‑engineer should be comfortable with.  
Each prompt is written in its **most basic form** (no hints, no solution, no code). Feel free to adjust input sizes, constraints, or language‑specific details to match your learning goals.

---

## 1. Searching

**Problem – Linear Search**  
Given an integer array `A` of length `n` and a target value `x`, return the index of `x` in `A` (or `-1` if `x` is not present).  

---

## 2. Searching (Sorted)

**Problem – Binary Search**  
Given a **sorted** integer array `A` (ascending) of length `n` and a target value `x`, return the index of `x` (or `-1` if `x` is not present).  

---

## 3. Sorting

**Problem – In‑Place Bubble Sort**  
Write a function that sorts an integer array `A` of length `n` in non‑decreasing order using the bubble‑sort algorithm.  

---

## 4. Sorting (Divide & Conquer)

**Problem – Merge Sort**  
Implement merge sort to sort an integer array `A` of length `n` in non‑decreasing order.  

---

## 5. Sorting (Linear Time)

**Problem – Counting Sort**  
Given an integer array `A` where each element is in the range `[0, k]`, output a sorted version of `A` in `O(n + k)` time.  

---

## 6. Recursion

**Problem – Fibonacci (Naïve Recursion)**  
Write a recursive function `fib(n)` that returns the `n`‑th Fibonacci number (with `fib(0)=0`, `fib(1)=1`).  

---

## 7. Dynamic Programming – Subsequence

**Problem – Longest Increasing Subsequence (LIS)**  
Given an integer array `A` of length `n`, find the length of the longest strictly increasing subsequence.  

---

## 8. Dynamic Programming – Knapsack

**Problem – 0/1 Knapsack (Weight/Value)**  
You have `n` items, each with weight `w[i]` and value `v[i]`, and a knapsack capacity `W`. Compute the maximum total value you can obtain by selecting a subset of items (each item at most once).  

---

## 8b. Dynamic Programming – Subset Sum

**Problem – Subset Sum**  
Given a set of positive integers `S` and a target sum `T`, determine whether any subset of `S` adds up exactly to `T`.  

---

## 9. Greedy – Activity Selection

**Problem – Maximum Non‑Overlapping Intervals**  
You are given `n` intervals `[s_i, f_i]` (start and finish times). Choose the maximum number of intervals such that no two chosen intervals overlap.  

---

## 10. Greedy – Fractional Knapsack

**Problem – Maximize Value with Fractional Items**  
Given `n` items each with weight `w[i]` and value `v[i]`, and a knapsack capacity `W`, compute the maximum value you can achieve if you are allowed to take fractions of items.  

---

## 11. Graph Traversal – BFS

**Problem – Shortest Path in Unweighted Graph**  
Given an undirected, unweighted graph `G(V, E)` and two vertices `s` and `t`, find the length (number of edges) of the shortest path from `s` to `t`.  

---

## 12. Graph Traversal – DFS

**Problem – Detect Cycle in Directed Graph**  
Given a directed graph `G(V, E)`, determine whether it contains a cycle.  

---

## 13. Shortest Path – Dijkstra

**Problem – Single‑Source Shortest Paths (Positive Weights)**  
Given a directed graph with non‑negative edge weights and a source vertex `s`, compute the minimum distance from `s` to every other vertex.  

---

## 14. Shortest Path – Bellman‑Ford

**Problem – Detect Negative Cycle**  
Given a directed graph that may contain negative edge weights, compute the shortest distances from a source `s` to all vertices, and report if a negative‑weight cycle is reachable from `s`.  

---

## 15. Minimum Spanning Tree – Kruskal

**Problem – MST Weight**  
Given an undirected, weighted graph `G(V, E)`, compute the total weight of a Minimum Spanning Tree using Kruskal’s algorithm.  

---

## 16. Minimum Spanning Tree – Prim

**Problem – MST Construction (Adjacency Matrix)**  
Given a weighted undirected graph represented by an adjacency matrix, construct a Minimum Spanning Tree using Prim’s algorithm and output its total weight.  

---

## 17. String Matching – Naïve

**Problem – Find All Occurrences**  
Given a pattern string `P` and a text string `T`, output all starting indices in `T` where `P` occurs (allow overlapping matches).  

---

## 18. String Matching – KMP

**Problem – Efficient Pattern Search**  
Implement the Knuth‑Morris‑Pratt (KMP) algorithm to find all occurrences of a pattern `P` in a text `T`.  

---

## 19. String Processing – Longest Common Subsequence (LCS)

**Problem – LCA Length**  
Given two strings `A` and `B`, compute the length of their longest common subarray.  

---

## 20. String Processing – Longest Palindromic Substring

**Problem – Expand‑Around‑Center**  
Given a string `S`, return the longest substring of `S` that is a palindrome.  

---

## 21. Data Structures – Stack

**Problem – Balanced Parentheses**  
Given a string consisting only of characters `(`, `)`, `{`, `}`, `[`, `]`, determine whether the parentheses are balanced.  

---

## 22. Data Structures – Queue

**Problem – First‑Come‑First‑Served Simulation**  
Simulate a queue of customers where each customer arrives at time `t_i` and requires service time `s_i`. Compute the total waiting time for all customers (FIFO discipline).  

---

## 23. Data Structures – Binary Search Tree

**Problem – Validate BST**  
Given the root of a binary tree, determine whether it satisfies the binary search tree property.  

---

## 24. Data Structures – Heap

**Problem – K‑th Largest Element**  
Given an unsorted array `A` of length `n` and an integer `k`, find the `k`‑th largest element using a min‑heap of size `k`.  

---

## 25. Divide & Conquer – Closest Pair of Points

**Problem – Minimum Euclidean Distance**  
Given `n` points in the 2‑D plane, compute the smallest Euclidean distance between any pair of points.  

---

## 26. Geometry – Convex Hull

**Problem – Graham Scan**  
Given a set of points in the plane, output the vertices of the convex hull in counter‑clockwise order using the Graham scan algorithm.  

---

## 27. Bit Manipulation

**Problem – Count Set Bits**  
Given a non‑negative integer `x`, return the number of `1` bits in its binary representation.  

---

## 28. Number Theory

**Problem – Greatest Common Divisor (Euclidean Algorithm)**  
Implement the Euclidean algorithm to compute `gcd(a, b)` for two non‑negative integers `a` and `b`.  

---

## 29. Number Theory – Modular Exponentiation

**Problem – Fast Power (Modular)**  
Given integers `a`, `b`, and `m` (`m > 0`), compute `(a^b) mod m` efficiently.  

---

## 30. Randomized Algorithms

**Problem – Reservoir Sampling**  
Given a stream of unknown length, design an algorithm that, after reading the stream once, returns a uniformly random element from the stream.  

---

### How to Use These Exercises

1. **Pick a topic** – start with the area you’re weakest in.  
2. **Write test cases** – create edge cases (empty input, single element, large input, etc.).  
3. **Implement** – code the solution from scratch (no copy‑pasting).  
4. **Analyze** – after you’re done, determine the time and space complexity and compare it with the optimal bound.  
5. **Iterate** – once comfortable, try to improve the solution (e.g., replace a naïve \(O(n^2)\) approach with an \(O(n \log n)\) one).  

Happy coding! 🚀