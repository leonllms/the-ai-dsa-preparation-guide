# 18. String Matching – KMP

**Problem – Efficient Pattern Search**  
Implement the Knuth‑Morris‑Pratt (KMP) algorithm to find all occurrences of a pattern `P` in a text `T`. 

**Solution Overview (in plain language)**  

**In a nutshell**

1. Pre‑process the pattern to compute the LPS (failure) function.  
2. Walk through the text, advancing both indices on matches.  
3. On a mismatch, jump the pattern index back to the LPS value instead of restarting from the beginning.  
4. Whenever the pattern index reaches its length, record a match and continue using the LPS value to allow overlapping matches.

This yields all pattern occurrences in linear time without any explicit code.

---

1. **Goal** – We need to locate every position in the text `T` where the pattern `P` appears, and we want to do it in linear time `O(|T| + |P|)`.

2. **Key Idea – “Partial‑match” table**  
   The KMP algorithm avoids re‑examining characters that we already know match.  
   It does this by pre‑computing, for each prefix of the pattern, the length of the longest proper prefix that is also a suffix.  
   This table is usually called **LPS** (Longest Proper Prefix which is also a Suffix) or **π‑function**.

3. **Step 1 – Build the LPS table for the pattern**  

   *Initialize* `lps[0] = 0`.  
   Walk through the pattern from left to right (index `i = 1 … m‑1`, where `m = |P|`).  
   Keep a pointer `len` that tells how long the current longest prefix‑suffix is.  

   - If `P[i] == P[len]`, we can extend the prefix‑suffix: set `len = len + 1` and `lps[i] = len`.  
   - If they differ and `len > 0`, we “fallback” to the previous shorter prefix‑suffix: set `len = lps[len‑1]` and repeat the comparison.  
   - If they differ and `len == 0`, there is no proper prefix‑suffix ending at `i`; set `lps[i] = 0` and move to the next character.

   After this pass we have, for every position `i` in the pattern, the amount we can safely shift the pattern when a mismatch occurs at `i`.

4. **Step 2 – Scan the text using the LPS table**  

   Keep two indices: `i` for the text (0 … n‑1) and `j` for the pattern (0 … m‑1).  

   - While `i < n`:
     *If `T[i] == P[j]`* → characters match, advance both (`i++`, `j++`).  
     *If `j` reaches `m`* → we have matched the whole pattern; record the occurrence at position `i‑m` (0‑based) and then set `j = lps[j‑1]` to look for overlapping matches.  

     *If a mismatch occurs (`T[i] != P[j]`)*:
       - If `j > 0`, we do **not** move `i`; instead we shift the pattern by setting `j = lps[j‑1]`. This uses the pre‑computed information to align the longest possible prefix of the pattern with the suffix already matched.
       - If `j == 0`, there is no prefix to reuse, so we simply move to the next text character (`i++`).

   This loop never backs up `i`; each character of the text is examined at most once, and `j` only moves forward or jumps back using the LPS values.

5. **Result** – Every time `j` becomes equal to `|P|` we output the current start index (`i‑|P|`). After the scan finishes we have a complete list of all occurrences.

6. **Complexities**  

   - **Time**:  
     *Building LPS* takes `O(m)`.  
     *Scanning the text* takes `O(n)`.  
     Overall `O(n + m)`.  

   - **Space**:  
     Only the LPS array of size `m` and a few integer variables are needed → `O(m)` auxiliary space.

7. **Why it works** – The LPS table tells exactly how far the pattern can be shifted without losing any already‑matched characters. When a mismatch happens, we reuse the longest prefix that is also a suffix of the matched part, guaranteeing that no possible match is skipped and that we never re‑compare characters that we already know to be equal.

---

    
```python

def string_match_advanced(source, match):

    """
    The core idea is to construct an array that when we find a character mismatch,
    tells us how far back we should go in order to find the same candidate
    prefix . 
                                                    [0110011]
    For example: aacbaakak , when is matched against aacbaad at index 6 there is a
    mismatch but we shouldn't go back at index 0 of the pattern, instead we should
    go at index 1

    """

    n = len(source)
    m = len(match)

    if m == 0 or n == 0:
        return []

    lps=[0]*m
    offset = 0
    i = 1

    while i<m:
        if match[i] == match[offset]:
            # Advance if pattern repetition
            offset += 1
            lps[i]=offset
            i+=1
        elif offset > 0:
            # Check previous character occurence for subsequent match
            offset = lps[offset-1]
        else:
            # Reset if no previous character in the string
            offset = 0
            lps[i]=offset
            i+=1

    i = 0
    j = 0

    matches = []


    while i < n:
        
        if source[i] == match[j]:
            i += 1
            j += 1
        else:
            if j==0:
                i += 1
            else:
                j = lps[j-1]

        if j==m:
            matches.append(i-m)
            j = lps[m-1]

    return matches

```

