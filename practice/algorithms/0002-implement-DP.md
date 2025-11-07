# Dynamic programming

Dynamic programming is a method for solving complex problems by breaking them into simpler sub‑problems, solving each sub‑problem just once, storing its result (memoization), and reusing those results to build up the solution to the original problem. It is especially useful for optimization problems with overlapping sub‑problems and optimal‑substructure properties.


## Top down recursive

```python 
from typing import Callable, Iterable, Any

def dynamic_program(x: Any, terminate: Callable[[Any], Any | None], 
    combine: Callable[..., Any], predecessors: Callable[..., Any]) -> Any:

    T = terminate(x)
    if T is not None: return T 
    
    states = ( dynamic_program(u, terminate, combine, predecessors) 
        for u in predecessors(x) )

    return combine(*states)
```

**Example**:

```python
# Example: number of ways to climb n stairs with 1‑ or 2‑step moves
def term(n):               # base case
    return n == 0

def preds(n):
    return (n-1, n-2) if n >= 2 else (n-1,)

def combine(vals):
    return sum(vals)        # sum of the ways from the sub‑states

print(dynamic_program(5, term, combine, preds))   # → 8
```


### Cache based recursive

```python
from functools import lru_cache
from typing import Callable, Iterable, Any

def dynamic_program(x: Any, terminate: Callable[[Any], Any | None], 
    combine: Callable[..., Any], predecessors: Callable[..., Any]) -> Any:
    
    @lru_cache
    def _dp(v):
        T = terminate(v)
        if T is not None: return T 
        
        states = ( _dp(u) for u in predecessors(v) )

        return combine(*states)
    return _dp(x)
```

## Bottom up generic DP

1. Perform a topological sorting based on the predecessors function
2. Loop over the defined order computing the value

```python
from typing import Callable, Any, Iterable, Set, List

def bottom_up_dp(
    start: Any,
    terminate: Callable[[Any], Any | None],
    combine: Callable[..., Any],
    predecessors: Callable[[Any], Iterable[Any]],
) -> Any:
    """
    Compute the DP value for *start* without recursion.
    The algorithm first builds a topological order of all states that can
    reach *start* (by following the predecessor relation) and then evaluates
    the states in that order.

    Parameters
    ----------
    start : Any
        The state whose value we want.
    terminate : Callable[[Any], Any | None]
        Returns the base‑case value for a state or ``None`` if the state is not
        a base case.
    combine : Callable[..., Any]
        Combines the values of the predecessor states.
    predecessors : Callable[[Any], Iterable[Any]]
        Yields the predecessor states of a given state.

    Returns
    -------
    Any
        The DP value for *start*.
    """

    # ----- 1. collect reachable states and a topological order -----
    visited: Set[Any] = set()
    order: List[Any] = []          # will contain states in post‑order

    def dfs(v: Any) -> None:
        if v in visited:
            return
        visited.add(v)
        for u in predecessors(v):
            dfs(u)
        order.append(v)           # after all predecessors → post‑order

    dfs(start)

    # ----- 2. evaluate states in topological order (bottom‑up) -----
    dp: dict[Any, Any] = {}

    for v in order:                # predecessors are already in dp
        base = terminate(v)
        if base is not None:
            dp[v] = base
        else:
            vals = [dp[u] for u in predecessors(v)]
            dp[v] = combine(*vals)

    return dp[start]
```

### Example – ways to climb n stairs

```python
def term(n):
    # base case: 0 stairs → 1 way (empty sequence)
    return 1 if n == 0 else None

def preds(n):
    # from n we can come from n‑1 or n‑2 (if they exist)
    return (n - 1, n - 2) if n >= 2 else (n - 1,)

def combine(*vals):
    return sum(vals)

print(bottom_up_dp(5, term, combine, preds))   # → 8
```

**Alternative with a stack implementation**

```python 
from collections import deque
from typing import Callable, Any, Iterable, Set, List

def bottom_up_dp(
    start: Any,
    terminate: Callable[[Any], Any | None],
    combine: Callable[..., Any],
    predecessors: Callable[[Any], Iterable[Any]],
) -> Any:

    order = deque()
    visited = set()
    DP = {}

    def dfs(first):
        tovisit = [first]
        while tovisit:
            u = tovisit.pop()
            if u in visited:
                continue
            order.appendleft(u)
            visited.add(u)
            tovisit.extend([ x for x in predecessors(u)])

    dfs(start)

    for u in order:
        base = terminate(u)
        if base is not None:
            DP[u] = base
        else:
            vals = [DP[x] for x in predecessors(u)]
            DP[u] = combine(*vals)

    return DP[start]

```

