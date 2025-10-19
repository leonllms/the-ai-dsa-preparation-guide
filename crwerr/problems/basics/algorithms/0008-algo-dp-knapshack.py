"""

Errors:

Two indexing errors , one in assigning weights and values, another in the max computation.

"""


def kp_naive(weights, values, bagsize):

    """
    
    if there is a bag of size c and we add one element then capacity will be c-w and value v[c-w] + v 

    """

    n = len(weights)

    V=[[0]*(n+1) for _ in range(0,bagsize+1)]

    for i in range(1,n+1):
        for c in range(0,bagsize+1):
            
            v=values[i]
            w=weights[i]

            if c >= w:                
                V[c][i]=max(V[c-w][i-1]+v, V[c-w][i-1])
            else:
                V[c][i]=V[c][i-1]

    return V[bagsize][n]



# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.




"""

Errors below:

Space optimized solution with error in copying previous to current state,
having an extra parameter, and indexing error in max calculation. On top
there is a loop sequence error having the outer loop as an inner loop causing
miscomputation of the levels.

"""


def kp_so(weights,values,bagsize,target):

    n = len(weights)

    V = [[0,0] for _ in range(0,bagsize+1)]

    for c in range(0,bagsize+1):
        for i in range(n):

            v = values[i]
            w = weights[i]

            if c>=w:
                V[c][1] = max(V[c-w][0]+v, V[c-w][0])
            else:
                V[c][1] = V[c][0]

        V[:][0] = V[:][1]

    return V[bagsize][1]




# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.


