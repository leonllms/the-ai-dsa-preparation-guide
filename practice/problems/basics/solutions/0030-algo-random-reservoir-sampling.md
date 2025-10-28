## 30. Randomized Algorithms

**Problem – Reservoir Sampling**  
Given a stream of unknown length, design an algorithm that, after reading the stream once, returns a uniformly random element from the stream.  


**Idea**  
Keep only one element in memory while you scan the stream.  
When the *i*‑th item arrives (the first item has i = 1) decide whether to keep it or to keep the element that is already stored.


The strategy is to maintain a single "candidate" element. As you process the stream, you decide with a decreasing probability whether to replace your candidate with the new element.

Here is the step-by-step process:

1.  **See the first element:** Take the very first element from the stream and save it. This is your initial chosen element.

2.  **See the second element:** Now, there's a **1 in 2** (50%) chance that you will replace your saved element with this new one.

3.  **See the third element:** There's a **1 in 3** chance you will replace your saved element with this new, third element.

4.  **Continue this pattern:** For the *N*-th element you see in the stream, you give it a **1 in *N*** chance to replace the element you currently have saved.

When the stream ends, the single element you have saved is the result. This method guarantees that any element from the entire stream has an equal probability of being the final one chosen, no matter how long the stream is.


**Algorithm**  

1. Read the first element and store it as the current answer.  
2. For each subsequent element with index *i* = 2,3,…  
   * generate a random number in the range [0,1).  
   * with probability **1 / i** replace the stored answer by the new element; otherwise keep the old one.  
3. After the whole stream has been processed, output the stored element.

**Why it works**  

We prove by induction that after processing the first *i* items each of them is kept with probability 1/i.

*Base*: after the first item (i = 1) it is stored with probability 1, which equals 1/1.

*Inductive step*: assume after the first *i‑1* items each is stored with probability 1/(i‑1).  
When the *i*‑th item arrives we keep it with probability 1/i.  
For any earlier item *j* (j < i) to remain stored we need two things: it was stored after step *i‑1* (probability 1/(i‑1)) **and** we do **not** replace it at step *i* (probability 1‑1/i).  
Thus  

```
P(item j is final) = (1/(i‑1)) * (1‑1/i) = (1/(i‑1)) * ((i‑1)/i) = 1/i
```

The new item *i* is kept with probability 1/i by construction. Hence after *i* steps every item seen so far has probability 1/i of being the stored one.

When the stream ends, let *n* be its length; each of the *n* elements is kept with probability 1/n, i.e. uniformly at random.

**Properties**  

* One pass over the data – suitable for streams of unknown length.  
* O(1) extra memory (just the current candidate).  
* Works with any source of uniform random numbers.  

That is the reservoir‑sampling algorithm for selecting a single random element from a stream.


```python
import random

def stream_sample_uniform(inputstream):

    try:
        cursor = next(inputstream)
    except StopIteration:
        return None

    counter = 1
    range_bound = 1000
    candidate = cursor

    while True:
        try:
            cursor=next(inputstream)
        except StopIteration:
            return candidate
    
        counter+=1
        if counter > range_bound // 10:
            range_bound *= 10

        boundary = range_bound//counter
        randvar = random.randint(1, range_bound)

        if randvar < boundary:
            candidate=cursor


```

