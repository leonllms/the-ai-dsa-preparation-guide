# Breadth first search

Breadth first search (BFS) can be implemented recursively or with a Queue. 

## Recursive implementation

Below is a resursive implementation of BFS on a Tree

```python
def bfs_recursive(T, v):
    if T == []:
        return False

    childs = []
    for t in T:
        if t == v:
            return True
        childs.extend(t.childs())
    return bfs_recursive(childs, v)
```

## Queue based implementation

```python
def bfs_queue(T, v):
    tovisit = [T]
    while tovisit:
        t = tovisit.pop(0)
        if t == v:
            return True
        for n in t.childs():
            tovisit.append(n)
    return False
```
