# 2. **String (character array)**
**Problem:**  
Write a function that determines whether a given string `s` is a **permutation of a palindrome** (i.e., can be rearranged to form a palindrome). The solution must run in **O(|s|)** time and use **O(1)** extra space.


**Idea**

A string can be rearranged into a palindrome iff each character appears an even number of times, except possibly one character that may appear an odd number of times (the middle character of an odd‑length palindrome).  
So we only need to know how many characters have odd counts.

**How to get the odd‑count information in O(1) space**

* If the alphabet is known and small (e.g., only lowercase English letters), keep a 26‑bit integer.  
  * For each character `c` in the string, flip the bit that corresponds to `c` (`mask ^= 1 << (c‑'a')`).  
  * After processing the whole string, a bit set to 1 means that character has been seen an odd number of times.  

* If the alphabet is larger but still bounded (e.g., all ASCII characters), use a 128‑bit (or 256‑bit) mask – still constant space.

* If the alphabet is unbounded, use a single hash set (or a boolean array) that stores only the characters that currently have odd counts.  
  * When a character is read, if it is already in the set, remove it (its count becomes even); otherwise, insert it (its count becomes odd).  
  * The set never holds more than the number of distinct characters, which is bounded by the size of the character set, so it is still O(1) extra space.

**Decision step**

After the scan, examine the mask (or the set size):

* If the number of bits set (or the set size) is **0 or 1**, the string can be permuted into a palindrome.  
* If it is **greater than 1**, it cannot.

**Complexity**

* One pass over the string → **O(|s|)** time.  
* Only a fixed‑size bit mask (or a bounded‑size set) → **O(1)** extra space.


## Naive dictionary implementation

```python

def check_palindrome(candidate: str) -> bool:

    """

    Keep a dictionary that flips value every time a character is seen, and process
    the entire string. At the end count the non-zero characters ( present odd times ). 
    If there string length is odd at most one non-zero character can exist. For
    the even case no non-zero characters can exist for the string to be transformed
    into a palindrome. 

    """


    n = len(candidate)
    if n == 0:
        return False

    if n == 1:
        return True

    counters = {}
    for c in candidate:
        if c in counters.keys():
            counters[c] = int(not counters[c])
        else:
            counters[c] = 1

    non_zero = sum(counters.values())

    if n%2 == non_zero:
        return True

    return False

``` 

## Bitset based implementation in c++

```cpp
#include <bitset>
#include <cstddef>

template <typename T>
bool isPermutationPalindrome(const T& s) {
    std::bitset<256> bits;
    for (auto ch : s) {
        bits.flip(static_cast<unsigned char>(ch));
    }
    return bits.count() <= 1;
}
```