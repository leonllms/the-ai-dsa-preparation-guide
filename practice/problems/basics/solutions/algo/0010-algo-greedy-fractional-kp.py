"""

## 10. Greedy – Fractional Knapsack

**Problem – Maximize Value with Fractional Items**  
Given `n` items each with weight `w[i]` and value `v[i]`, and a knapsack capacity `W`, compute the maximum value you can achieve if you are allowed to take fractions of items.  


"""


def fractional_kp(weights, values, maxweight):

    """

    Find the maximum value fitting in a knapsack of a fixed capacity, given
    items with weights and values that can be taken fractionally.

    Observation: There are infinite fractions for every item

    Algorithm:
        Take the highest value items vs weights first, and then pick the rest if they fit 
        in decreasing fractions

    """


    n = len(weights)

    if n == 0:
        return 0

    if n == 1:
        return values[0]

    ratios = [ v/w for v,w in zip(values,weights) ]
    
    idx = [ i for _,i in sorted((v,j) for j,v in enumerate(ratios) , reverse=True) ]

    space = maxweight
    value = 0
    for i in range(n):

        j = idx[i]

        w = weights[j]
        v = values[j]

        if space >= w:
            value += v
            space -= w
        else:
            value += space*v/w
            space = 0

        if space == 0:
            return value


    return value



    