# 20. String Processing – Longest Palindromic Substring

## Naive solution

**Problem – Expand‑Around‑Center**  
Given a string `S`, return the longest substring of `S` that is a palindrome.  


# Write the solution briefly to this problem in simple terms without implementing it in code.

**Idea in plain language**

A palindrome reads the same forward and backward.  
If we pick any character (or the gap between two characters) as the “middle” of a palindrome, the whole palindrome is determined by how far we can stretch equally to the left and to the right while the characters keep matching.  

So the longest palindrome in the whole string can be found by trying every possible middle and expanding outward as far as the match holds.

**Step‑by‑step description**

1. **Treat every position as a possible centre**  
   * For odd‑length palindromes the centre is a single character (e.g., the “c” in “racecar”).  
   * For even‑length palindromes the centre is the gap between two consecutive characters (e.g., the centre between the two “e”s in “deed”).

2. **Expand from the centre**  
   * Starting with the chosen centre, move one step: look at the character just left of the centre and the character just right of the centre.  
   * If they are the same, the substring between them (including the centre) is still a palindrome – keep expanding outward.  
   * Stop when the characters differ or one side runs out of the string.

3. **Record the longest one seen so far**  
   * After each expansion we know the length (or the start‑and‑end indices) of the palindrome that was built from that centre.  
   * Compare it with the best palindrome we have kept; if it’s longer, replace the stored answer.

4. **Continue for all centres**  
   * Walk through the string from left to right, performing the two expansions (odd and even) at each index.

5. **Return the stored answer** – the longest palindromic substring found.

**Why it works**

*Any* palindrome has a centre: either a single middle character (odd length) or a middle gap (even length). By expanding from that centre we will exactly recreate the palindrome, because the definition of a palindrome is “mirrored characters on both sides”. Since we examine every possible centre, the longest palindrome cannot be missed.

**Complexity**

* **Time** – For each of the `n` positions we may expand up to `n` steps in the worst case (when the whole string is a palindrome). That gives an overall time of `O(n²)`.
* **Space** – Only a few integer variables are needed to keep indices and the best answer, so the extra space is `O(1)`.

**Summary**

1. Loop over every index `i` in the string.  
2. For each `i` run two expansions:  
   * one with centre at `i` (odd length)  
   * one with centre between `i` and `i+1` (even length)  
3. While the characters on the left and right match, grow the window.  
4. Keep the longest window found.  
5. After the loop, return the substring defined by that window.

That’s the “expand‑around‑center” method for the longest palindromic substring.


```python

def naive_longest_palindrome(astring):

    n = len(astring)


    def match_check(k,offset=0):
        return astring[c-k-offset]==astring[c+k]

    def odd_len(c):
        """
        Check if at position c in the string there exists an odd length palindrome

        In that case the center is c and all that is needed is to iteratively
        check the next and previous characters until either there is a mismatch
        or either of the boundaries is reached. 

        Edge cases: 
            - Empty : ''
            - Single: 'x'
            - 3-len : 'xxx' / 'xxa'
            - hit right: 'xxxa'
            - hit left: 'axxx'

        Assume that we never get an empty string since it would lead to index error

        Assume that a single character string is not a palindrome because there
        are not repeating characters since there is only one.

        """

        # Establish initialization

        if c<1:
            return 0

        if c+1>=n:
            return 0

        if astring[c-1] != astring[c+1]:
            return 0

        j=1
        while c-j-1>=0 and c+j+1<n:
            if astring[c-j-1] == astring[c+j+1]:
                j+=1
            else:
                return 2*j-1

        # If begin here we have exited the loop due to hitting either boundary
        return 2*j+1



    def even_len(c):
        """
        
        Check if at position c in the string there exists an even length palindrome.

        In that case the center is between c and c+1. All that is needed is to iteratively
        check characters expanding outwards from this center until either there is a mismatch
        or either of the boundaries is reached.

        Edge cases:
            - Empty: ''
            - 2-len: 'xx'/ 'xa'
            - 3-len: 'axx' / 'xxa'
            - 4-len: 'xxxx' 
            - 5-len: 'xxxxa' / 'axxxx'
            - 6-len: 'axxxxa'

        """

        if c == 0:
            return 0

        if astring[c-1] != astring[c]:
            return 0

        j=1
        while c-j-1>=0 and c+j<n:
            if astring[c-j-1]==astring[c+j]:
                j+=1
            else:
                return 2*j

        return 2*j


    final_len = 0
    for i in range(1,n):
        final_len=max(final_len, odd_len(i))
        final_len = max(final_len, even_len(i))

    return final_len


```



## Effient Algorithm

  
The linear‑time solution is **Manacher’s algorithm**.

**Idea**  
Treat the string as if every character were separated by a special marker (for example ‘#’) so that both odd‑ and even‑length palindromes become odd‑length.  
While scanning the transformed string keep, for each position, the radius of the longest palindrome centred there.  
Use the previously computed palindrome that reaches farthest to the right (its centre and right‑most boundary).  
If the current position lies inside that boundary, its radius can be at least the mirror’s radius; otherwise start expanding from the current centre.  
Whenever a palindrome extends beyond the known right‑most boundary, update the centre and boundary.

**Steps**

1. Build `T = "^#a#b#c#…#$"` (adding sentinel characters to avoid bounds checks).  
2. Create array `P` of length `|T|` to store radii.  
3. Initialise `center = 0`, `right = 0`.  
4. For each `i` from 1 to `|T|-2`  
   * `mirror = 2*center - i`.  
   * If `i < right` set `P[i] = min(right‑i, P[mirror])`.  
   * Expand while `T[i + P[i] + 1] == T[i - P[i] - 1]` and increment `P[i]`.  
   * If `i + P[i] > right` update `center = i`, `right = i + P[i]`.  
5. The maximum value in `P` gives the longest palindrome radius; its centre maps back to the original string to obtain the answer.

**Complexity**  
*Time* – each character is visited a constant number of times, so `O(n)`.  
*Space* – the transformed string and the radius array need `O(n)` extra space.

Thus Manacher’s algorithm finds the longest palindromic substring in linear time, which is more efficient than the `O(n²)` expand‑around‑center method.