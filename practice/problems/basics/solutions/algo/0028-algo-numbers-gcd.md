# 28. Number Theory

**Problem – Greatest Common Divisor (Euclidean Algorithm)**  
Implement the Euclidean algorithm to compute `gcd(a, b)` for two non‑negative integers `a` and `b`.  

Here is a brief and simple explanation of the Euclidean algorithm.

The Euclidean algorithm finds the greatest common divisor (GCD) of two numbers, `a` and `b`, using a simple, repetitive process:

1.  **Divide** `a` by `b` and find the remainder.
2.  **Replace** `a` with `b`, and replace `b` with the remainder from the previous step.
3.  **Repeat** this process until the remainder is 0.

The GCD is the **last non-zero remainder** you calculated.

---
**Example: Find gcd(48, 18)**

1.  Divide 48 by 18: Remainder is **12**. (New pair is 18, 12)
2.  Divide 18 by 12: Remainder is **6**. (New pair is 12, 6)
3.  Divide 12 by 6: Remainder is **0**. Stop.

The last non-zero remainder was **6**, so the GCD of 48 and 18 is **6**.


```python
def gcd_euc(a: int,b: int) -> int:

    # If either is zero by convention return 0, could be None as well. Implying 
    # that the given pair is not admissible.

    if a == 0 or b == 0:
        return 0

    if a > b:
        lhs = a
        rhs = b
    else:
        lhs = b
        rhs = a

    remainder = -1
    while remainder:
        remainder = lhs % rhs
        lhs = rhs
        rhs = remainder

    return lhs

```
