# 704. Binary Search


""" 

Write a function that find a number in a sequence of sorted integers and return its index. 

"""


def bin_search(arr : list[int], target: int) -> int : 

    """

    General idea about the algorithm , take the midpoint. 

    if the midpoint is equal to target , return the midpoint. 

    if it is larger than take the lower split of the array ( don't include the midpoint)

    else take the higher split ( don't include the midpoint )


    if the array split is odd the above algorithm work like a charm. 

    two ways to work around this , the nice and the ugly. let's do the nice: 

    if the length of the split is even, then check the end of the array for equality .  

    given that from odd splits we get even splits it's easy to do this simply every time to avoid getting even splits.

    so it boils down to 

    if the array is even , check the last element and if it is not targer reduce the split-end by one . 

    find the midpoint , and if it is larger , take the lower split , check the end of the split, and midpoint computation on the rest

    otherwise find the midpoint of the highest split , check the first element and do midpoint computation on the rest. 

    """

# []
    n = len(arr)

# [1,2,3,5, 7], n=0, i=0, j=0, 

    if n==0:
        return -1

    if n==1:
        if arr[0] == target:
            return 0
        else:
            return -1


    i = 0
    j = n-1

    # if the array is even , make it odd by checking the last element for the target.

    while not i>j:

        n = j-i+1
        n = i+n//2

        if arr[n] == target:
            return n

        elif arr[n] > target:
                j = n-1
        else:
                i = n+1

    return -1

