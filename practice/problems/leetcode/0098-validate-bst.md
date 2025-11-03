# 98. Validate binary search tree


You are give the root of a binary tree, and you need to validate if it a binary search tree or not.

conditions:

- left subtree contains only nodes with keys strictly less than the key
- right subtree contains only nodes with keys strictly greater than the key
- both left and right subtrees are binary search trees


There are two ways to approach this, one is with recursion the other with
stack based recursion which is most effiecient especially for very deep trees.


## Recursive solution

Start from the root, and perform a DFS , at every step validating the assumptions.

```python
class treenode:
    def __init__(self, value , left=None, right=None ):
        self.value = value
        self.left = left
        self.right = right 

def validate_bst(t : treenode, lower: int | None = None, upper: int | None = None) -> bool:

    if t is None:
        return True

    def validate_current():
        # If both bounds are none is the case -inf , -inf
        if upper is None and lower is None:
            return True

        if (upper is not None and lower is not None and \
             lower < t.value and t.value < upper ) :
            return True

        if upper is not None and t.value < upper :
            return True

        if lower is not None and lower < t.value:
            return True

        return False

    if validate_current():
        return validate_bst(t.left, lower, t.value) and \
        validate_bst(t.right, t.value, upper)

    return False
```


## Stack based approach

```python
class cursor:
    def __init__(self,node, lower, upper):
        self.node = node
        self.lower = lower
        self.upper = upper


def validate_bst_full_stack( root: treenode ) -> bool:

    """

    Visit a node , check membership , generate the bounds for the childs, append them with the bounds in the que.

    """
    tovisit = [cursor(root, None, None)]

    def invalidate_node( t, lower, upper):
        if t is None:
            return False

        if lower is not None and t.val <= lower:
            return True

        if upper is not None and t.val >= upper:
            return True

        return False


    while tovisit:

        current = tovisit.pop() # check later for order of pop

        if invalidate_node(current.node, current.lower, current.upper):
            return False


        if current.node is not None:
            left_node = current.node.left
            left_lower = current.lower
            left_upper = current.node.val

            right_node = current.node.right
            right_lower = current.node.val
            right_upper = current.upper

            tovisit.append(cursor(left_node,left_lower, left_upper))
            tovisit.append(cursor(right_node,right_lower,right_upper))

    return True

```


## Advanced Recursive Solution

The recursive solution is based on holding passing checking the root of each
subtree is within bounds and childs are stricly smaller and larger for left and
right respectively. All that is needed is to pass down in a DFS manner the root
and bounds of each subtree.

### Helper datastructures

The absolute minimum class required is a node that has two childs left and right
and a value to be compared. Implementing the comparison dunders ( operators )
makes the code more readable and less prone to errors.

```python
from typing import Tuple
import sys

class node:
    """ 
    Models a binary tree node

    - Only the value is needed for the root node. 
    - Childs can either be assigned at construction or by modifying child refs.

    """

    def __init__(self, value, left: node | None = None, right: node | None = None):
        self.left = left
        self.right = right
        self.value = value

    def __eq__(self, other: 'node') -> bool:
        if isinstance(other, node):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: 'node') -> bool:
        if isinstance(other, node):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: 'node') -> bool:
        return self < other or self == other

    def __ne__(self, other: 'node') -> bool:
        return not self == other

    def __gt__(self, other: 'node') -> bool:
        return not self <= other

    def __ge__(self, other: 'node') -> bool:
        return not self < other

    def __neg__(self) -> 'node':
        newnode = self.__copy__()
        newnode.value = -newnode.value
        return newnode

    def __copy__(self):
        return node(self.value, self.left, self.right)

class infinite(node):
    """
    Models a node that is infinite

    The core functionality of the node is by returning always larger than all others.

    """

    def __init__(self):
        super().__init__(sys.maxsize)

```

### Implementation

First check if the root is defined; otherwise this is a valid bst but empty.
Unpack the lower and upper bounds from the tuple ( we could have just passed
them as argumented in the first place ). Reaching a leaf results in the root
of subtree being None so this is a valid branch. Check first if the root of
this subtree is within bounds, otherwise return false, and recurse on the
left and right side by setting the upper for left child and lower for right
child recursions to be the root of this subtree.

```python
from typing import Tuple

def validate_bst_recursive(root: node | None, bounds: Tuple[node, node] = (-infinite(), infinite())) -> bool :
    """
    Validate that a binary tree satisfies the binary search tree (BST) property.

    The function checks each node to ensure its value lies within the
    inclusive lower and exclusive upper bounds supplied in ``bounds``.
    For the root call the default bounds are (-infinite, infinite).

    Parameters
    ----------
    root : node | None
        The current subtree root. ``None`` represents an empty subtree,
        which is considered valid.
    bounds : Tuple[node, node], optional
        A pair ``(lower, upper)`` defining the allowed range for ``root.val``.
        The left child is validated with ``(lower, root.val)`` and the right
        child with ``(root.val, upper)``. The default uses negative and
        positive infinity.

    Returns
    -------
    bool
        ``True`` if the subtree rooted at ``root`` is a valid BST, otherwise
        ``False``.

    Notes
    -----
    * The function is recursive; it returns ``True`` for an empty tree.
    * ``infinite()`` must return a value that compares greater/less than any
      node value in the tree.
    """

    if root == None:
        return True

    lower, upper = bounds

    if lower < root and root < upper:
        return validate_bst_recursive(root.left, (lower, root)) and \
                validate_bst_recursive(root.right, (root, upper))

    return False

```