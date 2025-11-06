# Binary search tree

A binary search tree (BST) is a binary tree data structure in which each node contains a key (and optionally associated data) and satisfies the following property:  

- All keys in the left subtree of a node are less than the node’s key.  
- All keys in the right subtree of a node are greater than the node’s key.  

This ordering allows efficient operations such as search, insertion, and deletion, typically in O(log n) time for a balanced tree.

Here is a schoolbook implementation of a binary search tree

```python

class BST_node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self, root = None):
        self.root = root


    def _remove(self, node, parent):

        def set_parent(replacement):
            nonlocal parent
            if parent is None:
                self.root = replacement
            else:
                if parent.left == node:
                    parent.left = replacement
                else:
                    parent.right = replacement

        if node.left is None and node.right is None:
            set_parent(None)
            return None
        if node.left is None and node.right is not None:
            set_parent(node.right)
            return node.right
        if node.right is None and node.left is not None:
            set_parent(node.left)
            return node.left

        successor, suc_parent = self.min(node.right)
        if suc_parent:
            suc_parent.left = None
        successor.right = node.right
        successor.left = node.left
        set_parent(successor)

        node.left = None
        node.right = None

        return successor


    def insert(self, key, value):
        newnode = BST_node(key,value)
        if self.root is None:
            self.root = newnode
            return newnode
        cursor = self.root
        previous = None
        while cursor:
            if cursor.key == key:
                cursor.value = value
                return cursor
            if key < cursor.key:
                previous = cursor
                cursor = cursor.left
            if key > cursor.key:
                previous = cursor
                cursor = cursor.right
        
        if cursor is None:
            if key < previous.key:
                previous.left = newnode
            else:
                previous.right = newnode
        return newnode


    def delete(self, key):
        """
        Deletion is an elaborate algorithm. 

        Search for the key to delete. The replacement of its position with the
        appropriate key is a non simple process. If it has only a left or right 
        child replace with that. Otherwise, search for the inorder successor. 
        That is the next largest smallest node. Since it has a left child, the 
        inorder successor is going to be on the min of the right subtree.
        """
        cursor = self.root
        if cursor is None:
            return None
        previous = None
        while cursor:
            if cursor.key == key:
                return self._remove(cursor, previous)
            previous = cursor
            if key > cursor.key:
                cursor = cursor.right
            else:
                cursor = cursor.left
        return None


    def min(self,ref):
        """
        Find the min of a subtree
        """
        if ref is None:
            return None
        previous = None
        cursor = ref
        while cursor.left:
            previous = cursor
            cursor = cursor.left
        return cursor, previous

    def search(self, key):
        cursor = self.root
        while cursor is not None:
            if cursor.key == key:
                return cursor
            if cursor.key < key:
                cursor = cursor.right
            else:
                cursor = cursor.left
        return None
```

A binary search tree can also be implemented with an array for better memmory 
locality 

Check if there are any errors , don't suggest improvements or judge the
efficiency of the algorithm , this is a naive solution, be very brief.


## Binary search

Binary search is usually performed on a binary search tree. 

```python
def binary_search_tree(root, key):
    if root is None:
        return False
    if root.key == key:
        return True
    if root.key < key:
        return binary_search_tree(root.right, key)
    else:
        return binary_search_tree(root.left, key)
```

And it can be berformed with and without recursion:

```python
def binary_search_tree_nr(root,key):
    cursor = root
    while cursor is not None:
        if cursor.key == key:
            return True
        if cursor.key < key:
            cursor = cursor.right
        else:
            cursor = cursor.left
    return False
```


but it can also be performed on any sorted array, because satisfies the binary
tree property for any key.


```python
def binary_search_sorted(array,value):
    """
    In a sorted array of type(value) find the element equal to the value
    """

    # Find the mid-point of the array, if the element is equal exit returning
    # the index. Otherwise, if it is smaller, move to left side, or to the right
    # side when it is larger.

    n = len(array)
    if n == 0:
        return n        # Return past the end index

    if n == 1:
        return 0 if array[0]==value else 1

    i = n//2 - 1 
    lb = 0
    ub = n
    while ub>lb:
        if array[i] == value:
            return i
        # Move to the right side, adjusting boundaries
        if array[i] < value:
            lb = i+1
        # Move to the left side adjusting boundaries
        else:
            ub = i
        i = lb + (ub-lb) // 2
    return n
```

