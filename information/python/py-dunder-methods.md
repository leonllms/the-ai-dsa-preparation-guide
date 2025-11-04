# Python class magic methods


Below is the set of “magic” (dunder) methods that the Python data model defines.  
They are the names that the interpreter looks for when you use the corresponding
operator, syntax construct, or built‑in function.

---

### 1. Object construction / destruction
```
__new__(cls, …)          # called to create a new instance
__init__(self, …)         # called to initialise the instance
__del__(self)             # called when the object is about to be destroyed
```

### 2. Representation
```
__repr__(self)            # official string representation
__str__(self)             # informal / user‑friendly string representation
__format__(self, format_spec)
__bytes__(self)           # bytes() conversion
```

### 3. Attribute access
```
__getattr__(self, name)          # called when attribute not found normally
__setattr__(self, name, value)   # called for every attribute assignment
__delattr__(self, name)          # called for attribute deletion
__getattribute__(self, name)    # called for *all* attribute accesses
```

### 4. Callable objects
```
__call__(self, *args, **kwargs)
```

### 5. Container protocol
```
__len__(self)                # len()
__getitem__(self, key)       # indexing, slicing, iteration
__setitem__(self, key, value)
__delitem__(self, key)
__contains__(self, item)     # membership test (in)
__iter__(self)               # iterator object
__reversed__(self)           # reversed()
__next__(self)               # next() for iterator objects
```

### 6. Numeric / arithmetic operators
```
__add__(self, other)         __radd__(self, other)          __iadd__(self, other)
__sub__(self, other)         __rsub__(self, other)          __isub__(self, other)
__mul__(self, other)         __rmul__(self, other)          __imul__(self, other)
__matmul__(self, other)      __rmatmul__(self, other)       __imatmul__(self, other)
__truediv__(self, other)     __rtruediv__(self, other)      __itruediv__(self, other)
__floordiv__(self, other)    __rfloordiv__(self, other)     __ifloordiv__(self, other)
__mod__(self, other)         __rmod__(self, other)          __imod__(self, other)
__pow__(self, other, modulo=None)  __rpow__(self, other, modulo=None)  __ipow__(self, other, modulo=None)
__lshift__(self, other)      __rlshift__(self, other)       __ilshift__(self, other)
__rshift__(self, other)      __rrshift__(self, other)       __irshift__(self, other)
__and__(self, other)         __rand__(self, other)          __iand__(self, other)
__xor__(self, other)         __rxor__(self, other)          __ixor__(self, other)
__or__(self, other)          __ror__(self, other)           __ior__(self, other)

# Unary operators
__neg__(self)                # -x
__pos__(self)                # +x
__abs__(self)                # abs(x)
__invert__(self)             # ~x
```

### 7. Comparison operators
```
__lt__(self, other)          # <
__le__(self, other)          # <=
__eq__(self, other)          # ==
__ne__(self, other)          # !=
__gt__(self, other)          # >
__ge__(self, other)          # >=
```

### 8. Boolean / truth value testing
```
__bool__(self)               # bool(x)
__hash__(self)               # hash(x)
```

### 9. Context management (with‑statement)
```
__enter__(self)
__exit__(self, exc_type, exc_val, exc_tb)
```

### 10. Descriptor protocol
```
__get__(self, instance, owner)      # attribute access
__set__(self, instance, value)      # attribute assignment
__delete__(self, instance)          # attribute deletion
```

### 11. Class / subclass checks
```
__instancecheck__(self, instance)   # isinstance()
__subclasscheck__(self, subclass)   # issubclass()
```

### 12. Miscellaneous special methods
```
__dir__(self)               # dir()
__sizeof__(self)            # sys.getsizeof()
__format__(self, format_spec)
__await__(self)             # awaitable objects
__aiter__(self)             # async for iterator
__anext__(self)             # async iterator next()
__aenter__(self)            # async with entry
__aexit__(self, exc_type, exc, tb)   # async with exit
```

### 13. Pickling / serialization
```
__reduce__(self)
__reduce_ex__(self, protocol)
__getstate__(self)
__setstate__(self, state)
```

---

#### How to use the list
* Implement the method(s) that correspond to the operation you want your class to support.
* If you implement a binary operator, also consider the reflected version (`__radd__`, `__rsub__`, …) and the in‑place version (`__iadd__`, `__isub__`, …) if appropriate.
* For full compatibility with built‑ins, also provide the appropriate comparison and hash methods.

For the authoritative and up‑to‑date reference, see the Python documentation section **Data Model**:
https://docs.python.org/3/reference/datamodel.html

That page contains the complete table of special method names and the operations they control.
