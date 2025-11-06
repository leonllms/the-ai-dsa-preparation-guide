# Depth first search

There are two possible approaches to DFS , recursive and stack implementation.
Almost in all possible scenarios the stack based implementation is better. Reason
for this is that a recursion is also a stack based operation, as it just moves
the pointer along the call stack copying over the arguments. In the stack
based implementation the user can choose to place the datatypes, objects, or
pointers to either a statically allocated array on the stack or on an dynamically
allocated array. The first case is purely equivalent to making the recursion
per se.

## Recursive Implementation

For the purpose of algorithmic clarity the recursive implementation is pretty
straightforward and included here for completeness.

```python
def dfs_r(T, v):
    if T is None:
        return False
    if T == v:
        return True
    for t in T.childs():
        if dfs_r(t,v):
            return True
    return False
```

## Stack Implementation

```python
def dfs(T,v):
    if T is None or v is None:
        return False
    tovisit = [T]
    while tovisit:
        t = tovisit.pop()
        if t == v:
            return True
        tovisit.extend(t.childs())            
    return False
```
