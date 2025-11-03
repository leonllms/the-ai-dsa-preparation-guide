# Datastructures in python

**Python standard‑library “basic” containers and the data‑structure they are built on**

| Container (module)                | What it is logically                     | Underlying implementation (CPython) |
|-----------------------------------|------------------------------------------|--------------------------------------|
| `list`                            | Resizable array (dynamic array)          | C‑array that grows by a factor ≈ 1.125 |
| `tuple`                           | Immutable sequence (array)               | Same as `list` but read‑only        |
| `dict`                            | Mapping (hash table)                     | Open‑addressing hash table with perturbation |
| `set` (and `frozenset`)           | Unordered collection of unique items     | Same hash‑table code as `dict`      |
| `collections.deque`               | Double‑ended queue (O(1) pushes/pops)    | Block‑linked doubly‑linked list (each block holds 64 items) |
| `collections.OrderedDict`         | Mapping that remembers insertion order   | `dict` + doubly‑linked list of entries |
| `collections.defaultdict`        | Mapping with default‑value factory        | Subclass of `dict` (hash table)    |
| `collections.Counter`            | Multiset (mapping element → count)       | Subclass of `dict`                  |
| `collections.ChainMap`            | Logical view of several mappings         | Simple Python objects (no special C structure) |
| `array.array`                     | Compact homogeneous array of primitives  | C‑array of the specified type code  |
| `heapq` (functions)               | Binary heap (priority queue)             | Uses a plain `list` as the heap storage |
| `bisect` (functions)              | Binary search on a sorted sequence       | Works on any sequence (typically a `list`) |
| `queue.Queue`, `queue.LifoQueue`  | Thread‑safe FIFO/LIFO queues              | Internally a `collections.deque` protected by a lock |
| `multiprocessing.Queue`           | Process‑safe queue                        | Uses a pipe + a `deque` on the sending side |
| `weakref.WeakKeyDictionary` etc.  | Mapping that holds weak references       | Hash‑table‑based, similar to `dict` |

**What *is not* provided in the standard library**

| Missing container | Typical use‑case | Reason it isn’t in the std‑lib |
|-------------------|------------------|--------------------------------|
| Balanced binary tree (e.g. red‑black, AVL) | Ordered map / set with O(log n) ops | Not needed for most scripts; can be installed (`sortedcontainers`, `bintrees`) |
| Skip list, B‑tree, interval tree, segment tree, trie, suffix tree, etc. | Specialized algorithms / large‑scale indexing | Too niche for the core library |
| Graph data structures (adjacency list / matrix) | General graph algorithms | Provided by third‑party packages (`networkx`) |
| Bloom filter, Count‑Min sketch, other probabilistic structures | Approximate membership / counting | Not part of the language spec |
| Persistent/immutable data structures (e.g. HAMT) | Functional programming | Not a design goal of CPython |
| Thread‑safe lock‑free queues, lock‑free stacks | High‑performance concurrency | Hand‑rolled in C extensions when needed |
| Priority‑queue class (besides the `heapq` functions) | Object‑oriented heap | `heapq` supplies the algorithm; a thin wrapper can be built easily |

**Take‑away**

* The only container that is *actually* a doubly‑linked list in CPython is `collections.deque` (and the order‑preserving `OrderedDict` that builds on it).  
* All other “basic” containers are either dynamic arrays (`list`, `tuple`, `array.array`) or hash‑table based (`dict`, `set`, their subclasses).  
* More advanced structures (balanced trees, tries, graphs, probabilistic sketches, etc.) are deliberately left out of the standard library and are expected to be supplied by third‑party packages.

