# 21. Data Structures – Stack

**Problem – Balanced Parentheses** 

Given a string consisting only of characters `(`, `)`, `{`, `}`, `[`, `]`, 
determine whether the parentheses are balanced.  



## Naive

**Solution Idea (in plain English)**  

1. **Use a stack** – think of it as a pile where you can only add or remove the topmost item.  
2. **Scan the string from left to right**.  
   * When you see an opening bracket `(`, `{` or `[`, **push** it onto the stack.  
   * When you see a closing bracket `)`, `}` or `]`:
     * If the stack is empty, there is no matching opening bracket → the string is **unbalanced**.  
     * Otherwise, look at the bracket on the top of the stack.  
       * If it is the *corresponding* opening bracket (e.g., `(` matches `)`), **pop** it off – the pair is matched.  
       * If it is a different kind of bracket, the ordering is wrong → the string is **unbalanced**.  
3. **After you have processed every character**:  
   * If the stack is empty, every opening bracket found a matching closing one, so the string is **balanced**.  
   * If the stack still contains any brackets, some openings never got closed → the string is **unbalanced**.  

**Why it works**  
The stack always keeps the *most recent* unmatched opening bracket at the top. A closing bracket can only correctly close the bracket that was opened most recently; any other situation would break the nesting rules. By pushing on openings and popping only when a correct closing appears, we enforce the proper nesting and order. When the scan finishes, an empty stack guarantees that all brackets were paired correctly.

**Key steps to remember**

| Step | Action |
|------|--------|
| 1    | Create an empty stack. |
| 2    | For each character `c` in the string: |
|      | • If `c` is an opening bracket → push `c`. |
|      | • If `c` is a closing bracket → if stack empty → *unbalanced*; else if top of stack matches `c` → pop; else → *unbalanced*. |
| 3    | After the loop, if stack is empty → *balanced*; otherwise → *unbalanced*. |

That’s the whole algorithm—no code needed, just the logical steps.

```python

def check_parantheses(somestr) -> bool :

    if len(somestr) == 0:
        return True

    store = []
    map = {
    '[' : ']',
    '{' : '}',
    '(' : ')'
    }

    for c in somestr:

        if c in '[{(':
            store.append(map[c])

        if c in ']})':
            if store:
                last = store.pop()
                if last != c:
                    return False
            else:
                return False
    
    return True if len(store) == 0 else False

```

