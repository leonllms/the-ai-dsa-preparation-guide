# 235. Lowest common ancestor of a binary search tree


"""

Find the lowest common ancestor of two nodes in a binary search tree.

The lowest common node is the lowest node in a tree that has as descendants both (all) nodes.

"""

def bst(root: treenode, x: treenode) -> list[treenode]:

    tovisit = [root]
    path = []

    while tovisit:

        cursor = tovisit.pop()

        if cursor is not None:

            if cursor.val == x:
                return path

            path.append(cursor)

            if cursor.val < x:
                tovisit.append(cursor.right)
            else:
                tovisit.append(cursor.left)

    return path


def compare(lhs: list[treenode], rhs: list[treenode]) -> treenode:

    if lhs == [] or rhs == []:
        raise ValueError("Invalid none empty list input")

    i = 1
    while lhs[i-1].val == rhs[i-1].val:
        i += 1

    return lhs[i-1]


def find_lca_bst(root: treenode, p: treenode, q: treenode) -> treenode:

    """

    1. find the first , recording its ancestors 
    2. find the second, recording its ancestors
    3. compare the two lists, where each level is an element

    """


    p_list = bst(root, p)
    q_list = bst(root, q)

    return compare(p_list,q_list)


"""

Effective solution , search for any of the two at the same time and keep a list of the paths 

"""



def find_lca_bst_promise(root:treenode, p:treenode, q:treenode) -> treenode:

    """

    How it works:

    Perform BST , recording the last

    stop where the search diverges

    """

    cursor = root

    while ( cursor.val < p.val and cursor.val < q.val ) \
            or ( cursor.val > p.val and cursor.val > q.val )  :

        if cursor.val < p.val and cursor.val < q.val:
            cursor = cursor.right
        elif cursor.val > p.val and cursor.val > q.val :
            cursor = cursor.lefl

    return cursor




