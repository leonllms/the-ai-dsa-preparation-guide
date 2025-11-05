# Most important decorators to own in python

## The “Must‑Know” Python Decorators

Below is a curated list of the most useful built‑in (and a few standard‑library) decorators you’ll reach for in everyday Python code.  
I’ve grouped them by **what they’re used for**, given a short description, and included a minimal example you can copy‑paste and run.

---

### 1️⃣ Class‑related decorators

| Decorator | When to use it | Quick example |
|-----------|----------------|---------------|
| **`@property`** | Turn a method into a read‑only attribute (or add a setter / deleter). Great for computed values that should look like normal attributes. | <pre><code>class Circle:<br>    def __init__(self, r):<br>        self._r = r<br>    @property<br>    def radius(self):<br>        return self._r<br>    @radius.setter<br>    def radius(self, value):<br>        if value < 0:<br>            raise ValueError('radius must be ≥ 0')<br>        self._r = value<br>    @property<br>    def area(self):<br>        import math<br>        return math.pi * self._r ** 2<br><br>c = Circle(2)<br>print(c.area)      # 12.566…<br>c.radius = 3<br>print(c.area)      # 28.274…<br></code></pre> |
| **`@staticmethod`** | A method that does **not** need `self` or `cls`. Use it for utility functions that belong to the class namespace only. | <pre><code><br>class Math:<br>    @staticmethod<br>    def add(a, b):<br>        return a + b<br><br>print(Math.add(1, 2))   # 3<br></code></pre> |
| **`@classmethod`** | Receives the class (`cls`) as first argument. Use it for alternative constructors or for methods that need to affect the class itself. | <pre><code><br>class Person:<br>    def __init__(self, name):<br>        self.name = name<br>    @classmethod<br>    def from_full_name(cls, full):<br>        first, last = full.split()<br>        return cls(f\"{first} {last}\")<br><br>p = Person.from_full_name('Ada Lovelace')<br>print(p.name)   # Ada Lovelace<br></code></pre> |

---

### 2️⃣ Function‑related decorators

| Decorator | When to use it | Quick example |
|-----------|----------------|---------------|
| **`@functools.wraps`** (used **inside** your own decorators) | Preserves the original function’s metadata (`__name__`, `__doc__`, etc.) when you write a wrapper. | <pre><code><br>import functools<br><br>def logger(func):<br>    @functools.wraps(func)<br>    def wrapper(*args, **kwargs):<br>        print(f'Calling {func.__name__}')<br>        return func(*args, **kwargs)<br>    return wrapper<br><br>@logger<br>def greet(name):<br>    \"\"\"Say hello\"\"\"<br>    return f'Hi {name}'<br><br>print(greet('Bob'))<br>print(greet.__name__)   # greet (instead of wrapper)<br></code></pre> |
| **`@functools.lru_cache`** | Memoizes pure functions (no side‑effects). Great for expensive calculations that get called repeatedly with the same arguments. | <pre><code><br>import functools<br><br>@functools.lru_cache(maxsize=128)<br>def fib(n):<br>    if n < 2:<br>        return n<br>    return fib(n-1) + fib(n-2)<br><br>print(fib(35))   # fast after first call<br></code></pre> |
| **`@functools.singledispatch`** | Turns a function into a generic function that dispatches on the type of the first argument. Useful for “type‑specific” implementations without a class hierarchy. | <pre><code><br>from functools import singledispatch<br><br>@singledispatch<br>def html_escape(value):<br>    raise TypeError('Unsupported type')<br><br>@html_escape.register(str)<br>def _(value):<br>    return value.replace('&', '&amp;').replace('<', '&lt;')<br><br>@html_escape.register(int)<br>def _(value):<br>    return str(value)<br><br>print(html_escape('a<b&c'))   # a&lt;b&amp;c<br>print(html_escape(42))        # 42<br></code></pre> |
| **`@functools.total_ordering`** | If you define **one** rich comparison (`__lt__`, `__gt__`, `__le__`, `__ge__`) **and** `__eq__`, this decorator supplies the rest. | <pre><code><br>from functools import total_ordering<br><br>@total_ordering<br>class Version:<br>    def __init__(self, major, minor):<br>        self.major = major<br>        self.minor = minor<br>    def __eq__(self, other):<br>        return (self.major, self.minor) == (other.major, other.minor)<br>    def __lt__(self, other):<br>        return (self.major, self.minor) < (other.major, other.minor)<br><br>v1 = Version(1, 2)<br>v2 = Version(2, 0)<br>print(v1 < v2)   # True<br>print(v1 >= v2)  # False (auto‑generated)<br></code></pre> |
| **`@contextmanager`** (from `contextlib`) | Turns a generator function into a context manager (`with … as …`). Handy for ad‑hoc resource handling without writing a full class. | <pre><code><br>from contextlib import contextmanager<br><br>@contextmanager<br>def open_temp(file, mode='w'):<br>    f = open(file, mode)<br>    try:<br>        yield f<br>    finally:<br>        f.close()<br><br>with open_temp('tmp.txt') as f:<br>    f.write('hello')<br></code></pre> |
| **`@dataclass`** (from `dataclasses`) | Generates `__init__`, `__repr__`, `__eq__`, and optional ordering for classes that mainly store data. | <pre><code><br>from dataclasses import dataclass<br><br>@dataclass(order=True)<br>class Point:<br>    x: float<br>    y: float<br><br>p1 = Point(1, 2)<br>p2 = Point(2, 3)<br>print(p1)          # Point(x=1, y=2)<br>print(p1 < p2)     # True (order=True gives comparison based on fields)<br></code></pre> |
| **`@typing.overload`** (type‑checking only) | Declares multiple signatures for a single implementation, letting static type checkers understand more complex call patterns. | <pre><code><br>from typing import overload, Union<br><br>@overload<br>def parse(s: str) -> int: ...<br>@overload<br>def parse(s: bytes) -> int: ...<br><br>def parse(s: Union[str, bytes]) -> int:<br>    return int(s)<br><br>reveal_type(parse('5'))   # mypy: int<br></code></pre> |

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