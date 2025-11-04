# 0001. Two Sum

Find two the indices of numbers in an array of integers that add up to a
target value.

Naive solution go over all numbers , holding one and then check with all the
other numbers if any adds up to target.

```python
from typing import List, Tuple

def naive_two_sum(numbers: List[int, int], target: int) -> Tuple[int, int] | Tuple[None,None]: 

    n = len(numbers)
    if n == 0:
        return None, None

    for i in range(n):
        for j in range(i,n):
            if numbers[i] + numbers[j] == target:
                return i,j

    return None
```

Go over the array , if a number would add-up then is must be equal to 
target - numbers[i] . We just need to look it up in a datastructure of seen
numbers

Complexity n steps going over all , log(i) steps to add each seen number, 
log(i) steps to look the difference. Immediately we can improve by dropping 
numbers that are larger than target value. 

```python
from typing import List, Tuple

def lut_two_sum(numbers: List[int], target: int) -> Tuple[int, int] | Tuple[None, None]:

    # Get the range of numbers, then get the difference required to obtain the
    # target number. If the current number is not larger equal to target, check
    # if there is an already seen number that equals that difference. If there
    # is found one then return the associatted index. Otherwise, add it the
    # current number as a candidate to seen numbers and store its index. If the
    # entire range of numbers is exhausted then just exit.

    n = len(numbers)
    if n==0:
        return None, None
    seen = {}

    for i in range(n):
        diff = target - numbers[i]

        if diff > 0:
            candidate = seen.get( diff, None)
            if candidate is not None:
                return candidate, i
            else:
                seen[numbers[i]] = i
        # else continue with the next one

    return None, None
```
