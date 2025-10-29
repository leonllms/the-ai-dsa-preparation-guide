# 4. **Doubly Linked List**

**Problem:**  
Implement a **LRU (Least Recently Used) cache** with capacity *C* that supports `get(key)` and `put(key,value)` in **O(1)** time. Use a doubly‑linked list together with a hash map.


## LRU Implementation

```python
class DLIST:

    class node:
        def __init__(self, key=None, data=None, nextnode: 'DLIST.node'= None, 
            previousnode: 'DLIST.node' = None):
            self._key_ = key
            self._data_ = data
            self._next_ = None
            self._previous_ = None

        def value(self):
            return self._data_

        def key(self):
            return self._key_


    class iter:

        def __init__(self, first: DLIST.node):
            self._cursor_ = first

        def __iter__(self) -> 'DLIST.iter':
            return self

        def __next__(self) -> DLIST.node :
            if self._cursor_ == None:
                raise StopIteration
            final_cursor = self._cursor_
            self._cursor_ = self._cursor_._next_
            return final_cursor

    class riter:

        def  __init__(self, last: DLIST.node):
            self._cursor_ = last

        def __iter__(self) -> 'DLIST.riter':
            return self

        def __next__(self):
            if self._cursor_==None:
                raise StopIteration
            final_cursor = self._cursor_
            self._cursor_ = self._cursor_._previous_
            return final_cursor


    def __init__(self, root: DLIST.node = None):

        self._head_ = root
        self._tail_ = root

        # Make certain that pointers are set appropriately
        if root is not None:
            root._next_ = None
            root._previous_ = None


    def _find_by(self, attr_val: list, attr_name: list =['_key_'], k=1) -> DLIST.node:
        """
        Retrieve first k matching nodes by key
        """
        found = []
        for node in self:
            matches = len(attr_val)
            for attribute, value in zip(attr_name,attr_val):
                if getattr(node,attribute, None) != value:
                    break
                matches -=1

            if matches == 0:
                found.append(node)
                k-=1

            if k==0:
                return found

    def head(self):
        return self._head_


    def tail(self):
        return self._tail_


    def drop(self, at: DLIST.node) -> DLIST.node | None :
        """
        Remove a node form the list, invalidates iterator references point to 'at'
        """

        if at is not None:
            previous = at._previous_
            nextnode = at._next_
            # Check edge cases 1. Head , 2. Tail
            if at != self._head_ and at != self._tail_:
                # Check if node is within ( not at the tail limits ) 
                previous._next_ = at._next_
                nextnode._previous_ = at._previous_

            elif at == self._head_:
                nextnode._previous_ = None
                self._head_ = nextnode
            else:
                previous._next_ = None
                self._tail_ = previous

            # Remove the node by removing references to others, this makes any
            # iterators pointing to 'at' invalidated
            at._next_ = None
            at._previous_ = None

        return at


    def pop(self, at: DLIST.node = None):
        
        if at is not None:
            # Just get the node and then drop it
            self.drop(at)
            return at
        else:
            return self.pop(self._tail_)

    def swap(self, first: DLIST.node, second: DLIST.node):

        # If the same node nothing to swap
        if first == second:
            return

        # Set head and tail references if needed
        if first == self._head_:
            self._head_ = second
        elif first == self._tail_:
            self._tail_ = second

        if second == self._head_:
            self._head_ = first
        elif second == self._tail_:
            self._head_ = first

        # Swap first with second
        second_next = second._next_
        second_previous = second._previous_

        second._next_ = first._next_
        second._previous_ = first._previous_

        first._next_ = second_next
        first._previous_ = second_previous


    def find_key(self, key,k=1):
        return self._find_by(['_key_'],[key],k)


    def push_back(self, xnode: DLIST.node):
        
        if self._head_ is None and self._tail_ is None:
            self._head_=xnode
            self._head_._next_=None
            self._head_._previous_=None
            self._tail_=self._head_
        else:
            self._tail_._next_ = xnode
            xnode._previous_ = self._tail_
            self._tail_ = xnode


    def push_front(self, xnode: DLIST.node):
        
        if self._head_ is None and self._tail_ is None:
            self._head_ = xnode
            self._head_._next_ = None
            self._head_._previous_ = None
            self._tail_=self._head_
        else:
            xnode._next_=self._head_
            self._head_._previous_=xnode
            self._head_=xnode


    def __iter__(self) -> DLIST.iter:
        return DLIST.iter(self._head_)


    def __reversed__(self) -> DLIST.riter:
        return DLIST.riter(self._tail_)

```

Below is the LRU implementation simplest version.

```python

class LRU:
    """
    
    Functionally this class model an least recently used cache by keeping a 
    dictionary for O(1) lookups during put and get, and a doubly linked list 
    to push most recently used item back at the front in O(1) time. When adding
    an item exceeds capacity then the cache is expired droping items from the
    back. 

    """    
    def __init__(self,capacity=100):
        self._LUT_={}
        self._STORE_=DLIST()
        self.remaining_space=capacity

    def get(self,key):
        if key not in self._LUT_:
            return None

        current_node = self._LUT_[key]

        # Move the current_node to the front
        self._STORE_.drop(current_node)
        self._STORE_.push_front(current_node)

        return current_node.value()


    def put(self,key,value):
        """

        Checks first if a node with this key already exists. If it exists it
        gets replaced, otherwise a new item is added. The relevant node is
        first removed from the list if it exists and then in either case pushed
        to the front in O(1) time.
        
        The counter is reduced to check for remaining capacity in order to trigger
        cache expiration function.

        """

        if key not in self._LUT_:
            newnode = DLIST.node(key,value)
            self._LUT_[key] = newnode
            self._STORE_.push_front(newnode)
            self.remaining_space -= 1
        else:
            newnode = DLIST.node(key,value)
            current_node = self._LUT_[key]
            self._STORE_.drop(current_node)
            self._STORE_.push_front(newnode)
        
        self._expire_()


    def _expire_(self):
        while self.remaining_space < 0:

            # Drop from store and reduce space because the capacity should alway
            # track the size of the store
            removed_node = self._STORE_.pop()
            if removed_node is not None:
                self.remaining_space +=1

                key = removed_node.key()

                # Check if the last node was also in the dictionary and pop it
                #
                # It is possible it was set to none e.g. via direct access
                node = self._LUT_.get(key,None)
                if node is not None :
                    self._LUT_.pop(key)
```

It is also possible to add the asynchronous version

```python
import asyncio

class LRU:
    """
    Async LRU cache.  Look‑ups are O(1) via a dict and a doubly linked list.
    """

    def __init__(self, capacity: int = 100):
        self._LUT_ = {}
        self._STORE_ = DLIST()
        self.remaining_space = capacity
        self._lock = asyncio.Lock()          # protect all mutations

    def _register_node(self, key, value):
        node = DLIST.node(key, value)
        self._LUT_[key] = node
        self._STORE_.push_front(node)
        self.remaining_space -= 1

    def _replace_node(self, key, value, current: DLIST.node):
        newnode = DLIST.node(key, value)
        self._STORE_.drop(current)
        self._STORE_.push_front(newnode)
        self._LUT_[key] = newnode

    def _unregister_node(self, removed: DLIST.node):
        self.remaining_space += 1
        key = removed.key()
        if key in self._LUT_:
            self._LUT_.pop(key)        

    def _set_most_recent(self, node):
        """
        Move node to front as most‑recently used
        """
        self._STORE_.drop(node)
        self._STORE_.push_front(node)

    def _expire_(self):
        while self.remaining_space < 0:
            removed = self._STORE_.pop()
            if removed is None:
                break
            self._unregister_node(removed)

    async def get(self, key):
        async with self._lock:
            node = self._LUT_.get(key, None)
            if node is not None:
                self._set_most_recent(node)
                return node.value()
            return None

    async def put(self, key, value):
        async with self._lock:
            node = self._LUT_.get(key, None)
            if node is None:
                self._register_node(key, value)
            else:
                self._replace_node(key, value, node)
            self._expire_()

```


