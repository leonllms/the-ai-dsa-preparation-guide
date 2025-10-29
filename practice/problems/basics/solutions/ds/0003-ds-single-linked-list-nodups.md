# 3. **Singly Linked List**
**Problem:**  
Given the head of a singly linked list, remove **all duplicate nodes** so that each value appears only once. The list is **not** sorted. Aim for **O(n)** time and **O(n)** auxiliary space.


**Schoolbook implementation of Singly Linked List**

```python


class slist:

    class node:
        def __init__(self, value):
            self._next = None
            self._value = value

    def __init__(self, root: slist.node = None):
        self._head = root
        self._tail = root
        
        if self._head != None:
            # Make sure new node points in the right direction
            self._tail._next = None

    def append(self,newnode: slist.node):
        # If there is a head there is also a tail, otherwise it is None
        if self._tail:
            self._tail._next = newnode
            self._tail = newnode
            self._tail._next = None
        else:
            # There is no head so set it.
            self.__init__(newnode)


    def pop(self):

        if self._tail:
            current = self._head
            previous = current
            while current != self._tail:
                previous = current
                current = current._next
            # Set tail referece to previous to tail, in practice dropping the
            # last element from the queue while preserving the reference to via
            # current to it.             
            self._tail = previous
            self._tail.next = None
        
        else:
            current = self._head
            self.__init__()

        # Return current value
        return current._value

    def drop(self, at: node):
        """

        Drops the next element from the linked list by moving forward the next
        pointer. Iterators of the dropped element are not affected and 
        will still point to it. So advancing an iterator will work as expected
        from where it was. 

        """
        if at._next != None:

            if at._next != self._tail:
                to_drop = at._next
                at._next = to_drop._next

            else:
                at._next = None
                self._tail = at
        

    class iter:

        def __init__(self, start: slist.node):
            self.current = start

        def __iter__(self):
            return self

        def __next__(self):
            nextnode = self.current 
            # Stop iteration when reached the last element
            if nextnode == None:
                raise StopIteration
            self.current = nextnode._next
            return nextnode

    def __iter__(self):
        return slist.iter(self._head)

    def __bool__(self):
        return self._head is not None

``` 

```python

def nodups(userlist: slist):
    
    seen = set()    

    if userlist:
        it = iter(userlist)
        try:
            previous = next(it)
            cursor = previous
            while True:
                if cursor._value in seen:
                    userlist.drop(previous)
                else:
                    seen.add(cursor._value)
                    previous = cursor
                cursor = next(it)
        except StopIteration:
            return
    return 

```

