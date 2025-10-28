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


