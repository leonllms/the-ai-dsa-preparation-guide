# 98. Validate binary search tree


""" 

You are give the root of a binary tree, and you need to validate if it a binary search tree or not.

conditions:

- left subtree contains only nodes with keys strictly less than the key
- right subtree contains only nodes with keys strictly greater than the key
- both left and right subtrees are binary search trees

"""


# There are two ways to approach this, one is with recursion the other with
# stack based recursion which is most effiecient. Since this is python 
# investigate whether recursion could be defined with a stack on the functions
# themselves and what would then be the advantages


# naive recursive solution# 98. Validate binary search tree


""" 

You are give the root of a binary tree, and you need to validate if it a binary search tree or not.

conditions:

- left subtree contains only nodes with keys strictly less than the key
- right subtree contains only nodes with keys strictly greater than the key
- both left and right subtrees are binary search trees

"""


# There are two ways to approach this, one is with recursion the other with
# stack based recursion which is most effiecient. Since this is python 
# investigate whether recursion could be defined with a stack on the functions
# themselves and what would then be the advantages


# naive recursive solution

"""

Start from the root, and perform a BFS , at every step validating the assumptions, if at any point the fail , return false.

Worst case logn calls should be less than roughly 100 , which implies 2^100 maximum elements is BST is balanced

"""

class treenode:


    def __init__(self, val =0 , left=None, right=None ):

        self.val = val
        self.left = left
        self.right = right 



def validate_bst(t : treenode, lower: int | None=None, upper: int | None ) -> bool:

    """ 

    0. if node is none return true ( finished )
    1. check left is not none and less than value of the node, and in the boundary
    2. update the boundary for the left node to [ -- , left]
    3. visit the node 
    4. check right is not none and greater than the value of the node and in the current boundary
    5. update the boundary for the right node to [right, -- ] 
    6. visit the node

    """

    """
        Tests:
            

    """

    # provide a natural language algorithm that solves the problem.

    if t is None:
        return True

    # Check if value of left node is within bounds and strictly smaller 
    # than current
    if t.left is not None:

        if t.left.val >= t.val:
            return False

        if upper is not None and t.left.val >= upper:
            return False

        if lower is not None and t.left.val <= lower:
            return False

    # Check if value of right node is strictly larger than value and
    # strictly within bounds 

    if t.right is not None:

        if t.right.val <= t.val: 
            return False

        if lower is not None and t.right.val <= lower:
            return False

        if upper is not None and t.right.val >= upper:
            return False 

    return validate_bst(t.left, lower, t.val) and validate_bst(t.right, t.val, upper)


# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.

"""

Yes, there are a few errors.

1.  The `upper` parameter is missing a default value of `None` in the function signature.
2.  The function does not check if the current node's value (`t.val`) is valid within the provided `lower` and `upper` bounds. This check is essential for the recursion to work correctly.

want's it checked in the previous check of the recursion dumbo?


You've correctly identified the core issue in your comment. The standard recursive approach validates the *current* node, not the children of the current node.

Yes, there are a few errors in the provided implementation:

1.  A syntax error: The `upper` parameter in the function signature is missing a default value of `None`.
2.  A logical error: The function checks the values of the children (`t.left.val`, `t.right.val`) but it never validates the **current node's value** (`t.val`) against the `lower` and `upper` bounds that are passed into the function. Each recursive call should first validate the node it receives before checking its children.o?

"""

# [ 1<-2->3 ] , []  , current = (1,none,2), ln=none, ll=none,lu=1, rn=3,rl=1,ru=none

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


# Check if there are any errors , don't suggest improvements or judge the efficiency of the algorithm , this is a naive solution, be very brief.


# There are no logical errors. The code correctly validates a strict Binary Search Tree. The bounds for child nodes are generated correctly, and the `invalidate_node` function properly checks if a node's value violates those bounds. The handling of null nodes is also correct.

