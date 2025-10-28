"""

## 8b. Dynamic Programming – Subset Sum

**Problem – Subset Sum**  
Given a set of positive integers `S` and a target sum `T`, determine whether any subset of `S` adds up exactly to `T`.  


"""


def subset_sum_validate(numbers, target):

    """

    Problem statement:

    Determine if a subset of numbers adds up exactly to the target value 

    Solution:
    
    Subproblem structure: Assume that for some target value smaller than 
    the sought after value there is a subset that adds up exactly to that. 
    Then for some other value larger than previously but still smaller, 
    taking an integer would imply that there should be a subset without that 
    integer that exactly adds up to the new value subtracting the integer.

    Alternatively tracking the state (true/false) would be sufficient. 

    """

    n = len(numbers)

    vals = [[0]*(n+1) for _ in range(target+1)]

    for i in range(1,n+1):

        for t in range(target+1):

            x = numbers[i-1]

            if t>=x and vals[t-x][i-1] == t-x:
                    vals[t][i] = t
            else:
                vals[t][i] = vals[t][i-1] 

    return True if vals[target][n] == target else False
