# Data structures in C++

**C++ standard‑library “basic” containers and the data‑structure they are built on**  
*(CPython‑style table, but for the C++ standard library – all implementations are the ones used by the most common compiler libraries such as libstdc++ and libc++)*

| Container (header)                | Logical abstraction                              | Underlying implementation (typical) |
|-----------------------------------|------------------------------------------------|--------------------------------------|
| `std::vector<T>`                  | Resizable array (dynamic array)                | Contiguous C‑array that grows by a growth factor (≈ 1.5) |
| `std::array<T,N>`                 | Fixed‑size array                               | Plain C‑array of length *N* |
| `std::deque<T>`                   | Double‑ended queue, O(1) push/pop both ends    | Block‑linked list of fixed‑size buffers (usually 512 bytes) – each buffer is a small array; the blocks are doubly linked |
| `std::list<T>`                    | Doubly‑linked list                             | Classic doubly‑linked list (each node stores prev/next pointers) |
| `std::forward_list<T>`            | Singly‑linked list                             | Singly‑linked list (next pointer only) |
| `std::map<Key,T>`                 | Ordered associative container (log N)           | Red‑black tree (self‑balancing binary search tree) |
| `std::multimap<Key,T>`            | Ordered multimap                               | Same red‑black tree |
| `std::set<Key>`                   | Ordered set (log N)                            | Red‑black tree |
| `std::multiset<Key>`              | Ordered multiset                               | Red‑black tree |
| `std::unordered_map<Key,T>`       | Hash table (average O(1) lookup)               | Open‑addressing hash table with linear probing / robin‑hood style (implementation‑defined) |
| `std::unordered_set<Key>`         | Hash set                                       | Same as `unordered_map` |
| `std::unordered_multimap` / `unordered_multiset` | Hash‑table multimap/multiset | Same as above |
| `std::bitset<N>`                  | Fixed‑size bit vector                          | Array of machine words, bit‑wise ops |
| `std::stack<T>` (adapter)         | LIFO stack                                     | Uses an underlying container (`deque` by default) |
| `std::queue<T>` (adapter)         | FIFO queue                                     | Uses an underlying container (`deque` by default) |
| `std::priority_queue<T>`          | Max‑heap (or min‑heap)                          | Uses a `std::vector` as the heap storage (binary heap) |
| `std::span<T>` (C++20)            | Non‑owning view of a contiguous sequence       | Just a pointer + length, no storage |
| `std::string` / `std::basic_string`| Dynamic character array (like `vector<char>`) | Contiguous array with small‑string optimisation |
| `std::valarray<T>`                | Numeric array with element‑wise ops             | Contiguous array (often a `vector`‑like layout) |
| `std::array<T,N>` (already listed) | Fixed‑size array                               | Plain C‑array |
| `std::optional<T>`                | Wrapper that may hold a value                  | Union + bool flag – not a container per‑se |
| `std::variant<T…>`                | Tagged union                                   | Discriminated union, not a container |
| `std::pair`, `std::tuple`        | Fixed‑size heterogeneous aggregate             | Simple struct layout |
| `std::any`                       | Type‑erased holder                             | Pointer to heap‑allocated object |
| `std::weak_ptr<T>` / `std::shared_ptr<T>` | Reference‑counted smart pointers            | Control block + pointer, not a container |
| `std::stack`, `std::queue`, `std::priority_queue` are *adapters* that delegate to one of the containers above (normally `deque` or `vector`). |
| `std::unordered_map`/`set` families | Hash‑table based associative containers        | Same underlying hash‑table code as `unordered_map` |
| `std::unordered_multimap` / `unordered_multiset` | Hash‑table multimap/multiset | Same as above |

### What the C++ standard library **does not provide**

| Missing container | Typical use‑case | Reason for omission |
|-------------------|-------------------|---------------------|
| Balanced tree variants other than red‑black (e.g. AVL, B‑tree, treap) | Ordered map/set with different balancing guarantees | Red‑black tree already covers most needs; other variants are left to third‑party libraries |
| Skip list, interval tree, segment tree, suffix tree, trie, etc. | Specialized algorithmic structures | Too niche for a general‑purpose language library |
| Graph classes (adjacency list/matrix) | General graph algorithms | Provided by external libraries such as Boost.Graph, NetworkX (Python) |
| Bloom filter, Count‑Min sketch, HyperLogLog, etc. | Approximate set / multiset | Probabilistic structures are outside the core language spec |
| Persistent/immutable collections (HAMT, Clojure‑style) | Functional programming | Not a design goal of the STL |
| Lock‑free / wait‑free concurrent containers | High‑performance multithreaded code | Implemented in platform‑specific libraries (e.g. Intel TBB, folly) |
| Ordered map/set with O(log N) *and* hash‑table fallback (e.g. “linked‑hash map”) | Insertion‑order preserving map with fast lookup | `std::unordered_map` + `std::list` can be combined manually; not needed as a separate type |
| Memory‑mapped containers, external‑memory B‑tree, etc. | Very large data sets | Outside the scope of the core library |

### Quick “cheat‑sheet” of the **basic** containers and their data‑structure backbone

| Category | Containers (header) | Underlying structure |
|----------|---------------------|----------------------|
| **Contiguous array** | `std::vector`, `std::string`, `std::array`, `std::valarray` | Contiguous C‑array (grow‑by‑factor) |
| **Linked lists** | `std::list` (doubly), `std::forward_list` (singly) | Linked nodes |
| **Deque** | `std::deque` | Block‑linked list of fixed‑size buffers |
| **Hash tables** | `std::unordered_map`, `std::unordered_set`, `std::unordered_multimap`, `std::unordered_multiset` | Open‑addressing hash table (probing) |
| **Tree‑based ordered maps/sets** | `std::map`, `std::multimap`, `std::set`, `std::multiset` | Red‑black tree |
| **Adapters** | `std::stack`, `std::queue`, `std::priority_queue` | Use one of the above as the underlying container |
| **Other associative helpers** | `std::pair`, `std::tuple`, `std::optional`, `std::variant` | Simple struct‑like layout, not a container |
| **Numeric helpers** | `std::bitset<N>` | Fixed‑size bit vector |
| **Thread‑safe queues** | `std::atomic`, `std::mutex` + `std::queue` (user‑built) | No built‑in lock‑free queue; you build it on top of the above |

**Take‑away**

* The only *true* doubly‑linked list in the C++ standard library is `std::list`.  
* The only container that is a *deque* (block‑linked) is `std::deque`.  
* All ordered associative containers (`map`, `set`, …) are red‑black trees.  
* All hash‑based containers (`unordered_*`) are open‑addressing hash tables.  
* Everything else (`vector`, `string`, `array`) is a contiguous array.  
* More exotic structures (balanced trees other than red‑black, tries, graphs, probabilistic sketches, etc.) are deliberately omitted and must be obtained from third‑party libraries such as Boost, Abseil, or the range‑v3 ecosystem.

