# 19. String Processing – Longest Common Subsequence (LCS)

**Problem – LCS Length**  
Given two strings `A` and `B`, compute the length of their longest common subsequence.  


A **subsequence** of a string is any sequence that can be obtained by deleting zero or more characters from the original string while keeping the remaining characters in their original order.  
The deletions do **not** have to be contiguous; only the relative order matters.

**Example**

- Original string: `ABCD`
- Subsequences: `ABCD`, `ABC`, `ABD`, `ACD`, `BCD`, `AB`, `AD`, `A`, `B`, … (including the empty string).

Thus, for two strings `A` and `B`, a *common subsequence* is a sequence that appears as a subsequence in both `A` and `B`. The **longest common subsequence (LCS)** is the common subsequence with the greatest possible length.


**Idea**

The longest common subsequence (LCS) of two strings can be found with a classic dynamic‑programming table.

*If the last characters of the two strings are the same, that character can be part of the LCS; otherwise we have to drop one of the two last characters and see which choice gives a longer subsequence.*



## Naive schoolbook implementation

```python

def lcs(alpha, beta):

    """

    Subproblem structure, say i have prefixes up to i and j from alpha and beta 
    respectively and the lcs has lengh k.

    There are three options: 

    1. Take the next character from both strings because it is common, and increase
    the length by one
    2. Take the character from the one or the other because they are different
    which makes then taking either identical to the maximum of either subsequence

    """


    n = len(alpha)
    m = len(beta)

    if m==0 or n==0:
        return 0

    length = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1,n+1):
        for j in range(1,m+1):

            if alpha[i-1] == beta[j-1]:
                length[i][j] = length[i-1][j-1] + 1
            else:
                length[i][j] = max(length[i-1][j], length[i][j-1])

    return length[-1][-1]

```

## Space optimization

At every iteration it is clear that only the last row is needed to build the new

```python

def lcs_so(alpha, beta):

    n = len(alpha)
    m = len(beta)

    if m==0 or n==0:
        return 0

    current = [0]*(m+1)
    previous = [0]*(m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):

            if alpha[i-1]==beta[j-1]:
                current[j] = previous[j-1] + 1
            else:
                current[j] = max(previous[j],current[j-1])

        previous = current
        current = [0]*(m+1)

    return previous[-1]
```


## Getting the longest subsequence

```python


def lcs_path(alpha,beta):

    n = len(alpha)
    m = len(beta)

    if n == 0 or m == 0:
        return 0,[]


    length = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1,n+1):
        for j in range(1,m+1):
            if alpha[i-1] == beta[j-1]:
                length[i][j] = length[i-1][j-1]+1
            else:
                length[i][j] = max(length[i-1][j],length[i][j-1])

    i = n
    j = m

    path = []
    while i>0 and j>0:
        if length[i][j]>length[i-1][j] and length[i][j]>length[i][j-1]:
            path.append(i)
            i-=1
            j-=1
        elif length[i][j]==length[i-1][j]:
            i-=1
        else:
            j-=1


    for c in range(len(path)//2):
        t = path[c]
        path[c] = path[-c-1]
        path[-c-1] = t

    return length[-1][-1], path


```
