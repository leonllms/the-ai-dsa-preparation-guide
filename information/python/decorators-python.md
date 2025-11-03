# Most important decorators to own in python

## The “Must‑Know” Python Decorators

Below is a curated list of the most useful built‑in (and a few standard‑library) decorators you’ll reach for in everyday Python code.  
I’ve grouped them by **what they’re used for**, given a short description, and included a minimal example you can copy‑paste and run.

---

### 1️⃣ Class‑related decorators

| Decorator | When to use it | Quick example |
|-----------|----------------|---------------|
| **`@property`** | Turn a method into a read‑only attribute (or add a setter / deleter). Great for computed values that should look like normal attributes. | ```python\nclass Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def radius(self):\n        return self._r\n    @radius.setter\n    def radius(self, value):\n        if value < 0:\n            raise ValueError('radius must be ≥ 0')\n        self._r = value\n    @property\n    def area(self):\n        import math\n        return math.pi * self._r ** 2\n\nc = Circle(2)\nprint(c.area)      # 12.566…\nc.radius = 3\nprint(c.area)      # 28.274…\n``` |
| **`@staticmethod`** | A method that does **not** need `self` or `cls`. Use it for utility functions that belong to the class namespace only. | ```python\nclass Math:\n    @staticmethod\n    def add(a, b):\n        return a + b\n\nprint(Math.add(1, 2))   # 3\n``` |
| **`@classmethod`** | Receives the class (`cls`) as first argument. Use it for alternative constructors or for methods that need to affect the class itself. | ```python\nclass Person:\n    def __init__(self, name):\n        self.name = name\n    @classmethod\n    def from_full_name(cls, full):\n        first, last = full.split()\n        return cls(f\"{first} {last}\")\n\np = Person.from_full_name('Ada Lovelace')\nprint(p.name)   # Ada Lovelace\n``` |

---

### 2️⃣ Function‑related decorators

| Decorator | When to use it | Quick example |
|-----------|----------------|---------------|
| **`@functools.wraps`** (used **inside** your own decorators) | Preserves the original function’s metadata (`__name__`, `__doc__`, etc.) when you write a wrapper. | ```python\nimport functools\n\ndef logger(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__}')\n        return func(*args, **kwargs)\n    return wrapper\n\n@logger\ndef greet(name):\n    \"\"\"Say hello\"\"\"\n    return f'Hi {name}'\n\nprint(greet('Bob'))\nprint(greet.__name__)   # greet (instead of wrapper)\n``` |
| **`@functools.lru_cache`** | Memoizes pure functions (no side‑effects). Great for expensive calculations that get called repeatedly with the same arguments. | ```python\nimport functools\n\n@functools.lru_cache(maxsize=128)\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(35))   # fast after first call\n``` |
| **`@functools.singledispatch`** | Turns a function into a generic function that dispatches on the type of the first argument. Useful for “type‑specific” implementations without a class hierarchy. | ```python\nfrom functools import singledispatch\n\n@singledispatch\ndef html_escape(value):\n    raise TypeError('Unsupported type')\n\n@html_escape.register(str)\ndef _(value):\n    return value.replace('&', '&amp;').replace('<', '&lt;')\n\n@html_escape.register(int)\ndef _(value):\n    return str(value)\n\nprint(html_escape('a<b&c'))   # a&lt;b&amp;c\nprint(html_escape(42))        # 42\n``` |
| **`@functools.total_ordering`** | If you define **one** rich comparison (`__lt__`, `__gt__`, `__le__`, `__ge__`) **and** `__eq__`, this decorator supplies the rest. | ```python\nfrom functools import total_ordering\n\n@total_ordering\nclass Version:\n    def __init__(self, major, minor):\n        self.major = major\n        self.minor = minor\n    def __eq__(self, other):\n        return (self.major, self.minor) == (other.major, other.minor)\n    def __lt__(self, other):\n        return (self.major, self.minor) < (other.major, other.minor)\n\nv1 = Version(1, 2)\nv2 = Version(2, 0)\nprint(v1 < v2)   # True\nprint(v1 >= v2)  # False (auto‑generated)\n``` |
| **`@contextmanager`** (from `contextlib`) | Turns a generator function into a context manager (`with … as …`). Handy for ad‑hoc resource handling without writing a full class. | ```python\nfrom contextlib import contextmanager\n\n@contextmanager\ndef open_temp(file, mode='w'):\n    f = open(file, mode)\n    try:\n        yield f\n    finally:\n        f.close()\n\nwith open_temp('tmp.txt') as f:\n    f.write('hello')\n``` |
| **`@dataclass`** (from `dataclasses`) | Generates `__init__`, `__repr__`, `__eq__`, and optional ordering for classes that mainly store data. | ```python\nfrom dataclasses import dataclass\n\n@dataclass(order=True)\nclass Point:\n    x: float\n    y: float\n\np1 = Point(1, 2)\np2 = Point(2, 3)\nprint(p1)          # Point(x=1, y=2)\nprint(p1 < p2)     # True (order=True gives comparison based on fields)\n``` |
| **`@typing.overload`** (type‑checking only) | Declares multiple signatures for a single implementation, letting static type checkers understand more complex call patterns. | ```python\nfrom typing import overload, Union\n\n@overload\ndef parse(s: str) -> int: ...\n@overload\ndef parse(s: bytes) -> int: ...\n\ndef parse(s: Union[str, bytes]) -> int:\n    return int(s)\n\nreveal_type(parse('5'))   # mypy: int\n``` |

---

### 3️⃣ “Write‑Your‑Own” Decorators (the patterns you’ll copy)

1. **Simple function decorator (no arguments)**  

   ```python
   import time
   import functools

   def timer(func):
       @functools.wraps(func)
       def wrapper(*args, **kwargs):
           start = time.perf_counter()
           result = func(*args, **kwargs)
           elapsed = time.perf_counter() - start
           print(f'{func.__name__} took {elapsed:.4f}s')
           return result
       return wrapper
   ```

2. **Decorator with arguments** (e.g., a configurable logger)  

   ```python
   def repeat(times: int):
       def decorator(func):
           @functools.wraps(func)
           def wrapper(*args, **kwargs):
               for _ in range(times):
                   func(*args, **kwargs)
           return wrapper
       return decorator

   @repeat(3)
   def hello():
       print('hi')
   ```

3. **Class‑based decorator** (useful when you need state that survives across calls)  

   ```python
   class CallCounter:
       def __init__(self, func):
           functools.update_wrapper(self, func)
           self.func = func
           self.calls = 0

       def __call__(self, *args, **kwargs):
           self.calls += 1
           print(f'Call #{self.calls}')
           return self.func(*args, **kwargs)

   @CallCounter
   def greet(name):
       print(f'Hello {name}')

   greet('Bob')
   greet('Alice')
   ```

---

## 🎯 When to Reach for Each

| Situation | Recommended decorator(s) |
|-----------|--------------------------|
| **Expose a read‑only/computed attribute** | `@property` (plus optional setter/deleter) |
| **Utility function that lives inside a class** | `@staticmethod` |
| **Alternative constructors or need to modify class state** | `@classmethod` |
| **Cache results of a pure function** | `@functools.lru_cache` |
| **Write a thin wrapper around an existing function** | Always use `@functools.wraps` inside the wrapper |
| **Implement a context manager quickly** | `@contextmanager` |
| **Define a data‑container class without boilerplate** | `@dataclass` |
| **Provide type‑specific behaviour without a class hierarchy** | `@functools.singledispatch` |
| **Need ordering but only want to implement `<` and `==`** | `@functools.total_ordering` |
| **Profiling / timing code** | Custom `@timer` (or `@profile` from external libs) |
| **Logging entry/exit of many functions** | Custom `@logger` decorator (often combined with `@wraps`) |
| **Complex call signatures for static type checking** | `@typing.overload` (type‑checker only) |

---

## 📚 Quick “cheat‑sheet” of import statements

```python
# core
import functools          # wraps, lru_cache, singledispatch, total_ordering
from contextlib import contextmanager
from dataclasses import dataclass

# typing (optional, for static analysis)
from typing import overload, Union
```

---

## TL;DR

- **`@property`**, **`@staticmethod`**, **`@classmethod`** – essential for class design.  
- **`@dataclass`** – eliminates boilerplate for data containers.  
- **`@functools.lru_cache`** – instant memoization for pure functions.  
- **`@functools.wraps`** – always use when you write a wrapper.  
- **`@contextmanager`** – quick, readable context managers.  
- **`@functools.singledispatch`** – clean way to do type‑based function overloading.  
- **`@functools.total_ordering`** – saves you from writing all comparison methods.  

Knowing these, plus the pattern for **custom decorators**, gives you a powerful toolbox for writing clean, reusable, and expressive Python code. Happy decorating! 🚀