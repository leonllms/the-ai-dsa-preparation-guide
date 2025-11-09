```python
from functools import lru_cache

def dynamic_program(start, termination, combine, dependencies):
    @lru_cache
    def dp(state):
        T = termination(state)
        if T is not None:
            return T
        d = [ dp(x) for x in dependencies(state)]
        return combine(*d)
    return dp(start)
```

```python
from collections import deque

def dynamic_program(first, termination, combine, previous):
    order = deque()
    DP = {}
    visited = set()
    def bfs(start):
        tovisit = deque()
        tovisit.append(start)
        while tovisit:
            u = tovisit.pop()
            if u not in visited:
                visited.add(u)
                order.appendleft(u)
                for v in previous(u):
                    tovisit.appendleft(v)
    
    bfs(first)

    for u in order:
        base = termination(u)
        if base is not None:
            DP[u] = base
        else:
            states = [ DP[v] for v in previous(u) ]
            DP[u] = combine(*states)
    return DP[first]


def wildcard_match(inputstr, pattern):

    n = len(inputstr)
    m = len(pattern)

    def term(state):
        i, j = state

        if i<0 and j<0:
            return True

        if j<0:
            return False    # Empty pattern doesn't match anything

        return None


    def deps(state):
        i,j = state
        if j<0:
            return []
        p = pattern[j] 
        # Case 1: p =='*', then is can match zero characters in the input string
        # for example 'aa*' matching 'aa' or it can match more than one, e.g. 
        # 'aa*' matching 'aabcd'. In the first case we need to ignore '*' so
        # a possible path is (i,j-1) whereas in the second case we move back
        # one character because it was matched.
        if p=='*':
            if i>=0:
                return [(i-1,j), (i,j-1)]
            else:
                return [(i,j-1)]

        # Case 2: p=='?' or literal. 'aa?' matches 'aab' but not 'aa', and the
        # literal case matches 'aab' but not 'aac'. In the literal case if they
        # match then the previous dependant state if (i-1,j-1) . In the '?' case
        # it is the same.
        if i>=0 and ( p=='?' or p==inputstr[i]):
            return [(i-1,j-1)]

        return []



    def comb(*states):
        return any(states)
        
    start=(n-1,m-1)

    return dynamic_program(start, term, comb, deps)

```

# Optimized

This is a membership problem so we only need to keep those states that are truthful

```python
from collections import deque

def dynamic_program(first, termination, combine, previous):
    order = deque()
    DP = set()
    visited = set()
    def bfs(start):
        tovisit = deque()
        tovisit.append(start)
        while tovisit:
            u = tovisit.pop()
            if u not in visited:
                visited.add(u)
                order.appendleft(u)
                for v in previous(u):
                    tovisit.appendleft(v)
    bfs(first)

# Just need the previous state in the bottom-up

    for u in order:
        base = termination(u)
        if base is not None:
            if base == True:
                DP.add(u)
        else:
            preds = previous(u)
            for p in preds:
                if p in DP:
                    DP.add(u)
    return first in DP


def wildcard_match(inputstr, pattern):

    n = len(inputstr)
    m = len(pattern)

    def term(state):
        i, j = state

        if i<0 and j<0:
            return True

        if j<0:
            return False    # Empty pattern doesn't match anything

        return None


    def deps(state):
        i,j = state
        if j<0:
            return []
        p = pattern[j] 
        # Case 1: p =='*', then is can match zero characters in the input string
        # for example 'aa*' matching 'aa' or it can match more than one, e.g. 
        # 'aa*' matching 'aabcd'. In the first case we need to ignore '*' so
        # a possible path is (i,j-1) whereas in the second case we move back
        # one character because it was matched.
        if p=='*':
            if i>=0:
                return [(i-1,j), (i,j-1)]
            else:
                return [(i,j-1)]

        # Case 2: p=='?' or literal. 'aa?' matches 'aab' but not 'aa', and the
        # literal case matches 'aab' but not 'aac'. In the literal case if they
        # match then the previous dependant state if (i-1,j-1) . In the '?' case
        # it is the same.
        if i>=0 and ( p=='?' or p==inputstr[i]):
            return [(i-1,j-1)]

        return []



    def comb(*states):
        return any(states)
        
    start=(n-1,m-1)

    return dynamic_program(start, term, comb, deps)

```

# Raw optimized

Uses O(n) space 

```python
def wildcard_match(s, p):
    n, m = len(s), len(p)

    # prev[j] = dp[0][j]  (empty string vs pattern prefix)
    prev = [False] * (m + 1)
    prev[0] = True                     # empty pattern matches empty string
    for j in range(1, m + 1):          # only '*' can keep it True
        prev[j] = prev[j - 1] and p[j - 1] == '*'

    for i in range(1, n + 1):
        curr = [False] * (m + 1)
        # dp[i][0] stays False because a non‑empty string cannot match an empty pattern
        for j in range(1, m + 1):
            if p[j - 1] == '*':
                # * matches empty (dp[i][j‑1]) or one more char (dp[i‑1][j])
                curr[j] = curr[j - 1] or prev[j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                # ordinary match uses diagonal value
                curr[j] = prev[j - 1]
            else:
                curr[j] = False
        prev = curr

    return prev[m]
```

