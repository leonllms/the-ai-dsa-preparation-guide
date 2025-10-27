## 23. Data Structures – Binary Search Tree

**Problem – Validate BST**  
Given the root of a binary tree, determine whether it satisfies the binary search tree property.  


**Idea**  
A binary search tree (BST) must satisfy: for every node, all keys in its left subtree are smaller than the node’s key and all keys in its right subtree are larger.  
We can verify this by traversing the tree while keeping the allowed range of values for each node.

**Algorithm (recursive range check)**  

1. Call a helper with the root node and the initial range `(-∞, +∞)`.  
2. If the current node is null, return true (an empty subtree is valid).  
3. Let `val` be the node’s key.  
   - If `val` is not greater than the lower bound **or** not less than the upper bound, the BST property is violated → return false.  
4. Recursively check the left child with the range `(lower, val)`.  
5. Recursively check the right child with the range `(val, upper)`.  
6. Return true only if both recursive calls return true.

**Why it works**  
The range passed to a child node represents all values that are still allowed for that subtree, given the ancestors already examined. If any node falls outside its permitted range, the ordering required by a BST is broken.

**Complexity**  
Each node is visited once, so the time complexity is **O(n)** where *n* is the number of nodes. The recursion depth is at most the height of the tree, giving **O(h)** auxiliary space (worst‑case O(n) for a degenerate tree).

**Alternative view**  
An inorder traversal of a BST yields keys in strictly increasing order. Traversing the tree in inorder and checking that each key is larger than the previously seen key is another way to validate the BST, also O(n) time and O(h) space.

```python

class node:

    def __init__(self,value, left=None, right=None):

        self.left = left
        self.right = right
        self.value = value

class infinity:

    def __init__(self):
        self._positive_ = True

    def __lt__(self,other):
        if self._positive_:
            # Nothing is larger than infinity
            # inf < 100 -- is False
            return False
        else:
            # Everything is larger than -infinity
            # -inf < -100 -- is true
            return True

    def __le__(self,other):
        return self.__lt__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self,other):
        return not self.__lt__(other)

    def __eq__(self, other):
        # Nothing is equal to infinity ( or -infinity)
        return False

    def __ne__(self,other):
        return True

    def __neg__(self):
        new_inf = infinity()
        new_inf._positive_ = False
        return new_inf


def validate_bst_stack(root: node) -> bool:
    
    """
    
    All keys left off current must be less and those on the right must be larger

    Evidently we must traverse the tree, validating the condition every time. This
    can be done in a DFS manner because a BFS wouldn't validate an entire branch
    at every level. 

    At every moving down the tree branch we must keep a range of values that 
    shouldn't be violated by lower level nodes. 

    Initialization, start with [-inf,inf] and then split left or right, setting
    the left or right limit according to either having moved left or right
    respectively.

    If there is no violation terminate when all nodes have been processed.

    """

    work = [root]
    bounds = [ (-infinity(), infinity()) ]

    while work:

        current = work.pop()
        low, high = bounds.pop()

        # Validate 
        if current.left is not None:
            left_val = current.left.value
            if left_val>=current.value or left_val<=low or left_val>=high :
                return False
            else:
                work.append(current.left)
                bounds.append((low,current.value))
                        

        if current.right is not None:
            right_val = current.right.value
            if right_val <= current.value or right_val<=low or right_val>=high:
                return False
            else:
                work.append(current.right)
                bounds.append((current.value, high))

    return True
```


# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.

GPT-OSS-120b ( but also QWEN3-235B-A22 )

The code has a few issues:

1. **Root None** – `validate_bst_stack` does not handle `root` being `None`; it will raise an exception when trying to access `current.left`.

2. **Bounds list mismatch** – `work` and `bounds` are popped in the same order, but when a node has both children the code pushes the left child first and then the right child, while the corresponding bounds are also pushed left‑first then right‑first. Because `pop()` removes the last element, the right child will be processed with the left child's bounds, which is incorrect.

3. **Redundant checks** – The conditions `left_val>=current.value` and `right_val<=current.value` already guarantee the BST ordering; the extra `left_val<=low` / `right_val>=high` checks are unnecessary but not harmful.

4. **Custom infinity** – The `infinity` class implements only a subset of comparison operators; using it with Python’s built‑in numeric types may lead to unexpected results in edge cases (e.g., when a node value is also an instance of `infinity`).

5. **Missing return for empty tree** – If the tree is empty the function should return `True`, but currently it will error out as mentioned in point 1.

I don't get point 2. bounds list mismatch. They are both pushed in the same order so the left node it pushed with its own bounds and the right with its own bounds , which is correct. Then they are popped also in the same order. Are you sure. 


When a node has two children you do:

```
work.append(left)
bounds.append((low, node.val))

work.append(right)
bounds.append((node.val, high))
```

`list.append` adds to the end, and `pop()` removes from the end.  
So after the two pushes the stack looks like (top on the right):

```
work   = [..., left, right]
bounds = [..., (low, node.val), (node.val, high)]
```

The next iteration executes `pop()` on both lists, therefore it takes **right** as the current node but pairs it with the **last** bounds entry, which is `(low, node.val)` – the bounds intended for the left subtree. The left child will later be paired with `(node.val, high)`. This swaps the bounds and can let an invalid value pass the check.

To keep the node and its bounds aligned you must push them together (e.g., as a tuple) or push the right child first so that the left child is popped first.


You are absolutely wrong, the (node.val, high) is popped with right.

The two stacks stay **in sync** because you push a child and its own bound together (first left, then right) and you pop from the end of each list. After the two pushes the situation is  

```
work   = … , left , right        # right is on top
bounds = … , (low, node.val) , (node.val, high)   # (node.val, high) is on top
```

A `pop()` removes the top element, so the next iteration gets **right** together with **(node.val, high)** – the correct bound for the right subtree. The left child will later be paired with **(low, node.val)**. Therefore the bounds are not mismatched; the earlier comment was mistaken.

