# Most common cacheing schemes

Below is a concise yet fairly comprehensive tour of the most common **caching schemes** you’ll encounter, grouped by what they try to solve:

| **Category** | **Scheme / Algorithm** | **What it decides** | **How it works (high‑level)** | **Typical use‑cases / pros‑cons** |
|--------------|------------------------|---------------------|------------------------------|-----------------------------------|
| **Eviction / Replacement Policies** (which items to discard when the cache is full) |
| LRU – *Least‑Recently‑Used* | Discard the entry that has not been accessed for the longest time. | Maintains a recency order (often a doubly‑linked list or a “clock” approximation). | Simple, good for workloads with strong temporal locality. Can be expensive O(1) with extra metadata; suffers from *cache‑pollution* on sequential scans. |
| MRU – *Most‑Recently‑Used* | Discard the *most* recently accessed entry. | Opposite of LRU; useful when the most recent item is unlikely to be reused soon (e.g., certain database “hot‑spot” scans). |
| FIFO – *First‑In‑First‑Out* | Discard the oldest inserted entry. | Simple queue; O(1) insert & delete. Works well when items have roughly equal lifetime. |
| LFU – *Least‑Frequently‑Used* | Discard the entry with the lowest access count. | Keeps a frequency counter per entry (often with decay to avoid “immortal” items). Good for workloads with strong *frequency* locality, but can be heavy on bookkeeping and can retain stale “cold” items that were once popular. |
| **Approximate LFU** (e.g., **TinyLFU**, **LFU‑A**) | Same goal as LFU but with probabilistic counters (e.g., Count‑Min Sketch). | Very low memory overhead, works well in CDN / web‑cache layers. |
| **ARC – Adaptive Replacement Cache** | Dynamically balances between recency (LRU) and frequency (LFU). | Maintains two LRU lists (one for recent, one for frequent) and two “ghost” lists to learn the optimal split. Proven to be *K‑competitive* with the optimal offline algorithm. |
| **CAR – Clock with Adaptive Replacement** | ARC‑style adaptation but using the cheap Clock algorithm instead of true LRU lists. | Good trade‑off for OS page replacement where per‑entry overhead must be minimal. |
| **2Q** | Two‑queue: a small “filter” LRU queue for recently added items + a larger LRU queue for items that survived the filter. | Reduces pollution from one‑time accesses while keeping true LRU behavior for “good” items. |
| **CLOCK / Second‑Chance** | Approximation of LRU using a circular buffer and a single reference bit. | Very cheap O(1) per operation; used in many OS page‑replacement implementations. |
| **Random** | Pick a victim uniformly at random. | Extremely cheap; surprisingly effective when the cache size is large relative to working set (e.g., in some CPU caches). |
| **Seg‑LRU (SLRU)** | Multiple LRU segments of different ages (e.g., “new”, “mid”, “old”). | Gives finer control over promotion/demotion; used in Linux page cache (the “active/inactive” lists). |
| **b‑LRU (Bucket‑LRU)** | Partition the cache into buckets; each bucket runs its own LRU. | Helps avoid “cache‑thrashing” among unrelated groups of keys (e.g., multi‑tenant caches). |
| **Belady’s Optimal (MIN)** | Discard the entry that will be used farthest in the future. | Theoretical benchmark; impossible to implement without future knowledge. |
| **Write‑Policies** (how writes are propagated to the backing store) |
| Write‑Through | Every write updates both cache and backing store synchronously. | Simple consistency; higher write bandwidth. Common in CPU L1/L2 caches and many web‑proxy caches. |
| Write‑Back (Write‑Behind) | Writes are staged in the cache and flushed later (on eviction or periodically). | Reduces write traffic; needs dirty‑bit tracking & recovery mechanisms. Used in CPU L2/L3, SSD caches, database buffer pools. |
| Write‑Around (Write‑No‑Allocate) | Writes bypass the cache; cache is not allocated on a write miss. | Avoids polluting cache with write‑only data; good for write‑heavy streams where reads are rare. |
| Write‑Allocate (Fetch‑on‑Write‑Miss) | On a write miss, the block is first fetched into cache, then updated. | Works well with write‑back; ensures future reads hit. |
| **Coherence / Consistency Protocols** (multi‑core or distributed caches) |
| MSI, MESI, MOESI, MESIF | Define states (Modified, Exclusive, Shared, Invalid, etc.) for each cache line. | Keep multiple copies of the same line coherent across cores. MESI is the classic x86 protocol; MOESI adds an “Owned” state for write‑back sharing. |
| Directory‑Based Coherence | Central directory tracks which cores hold a line. | Scales better than broadcast snooping for large many‑core systems. |
| **Distributed / Hierarchical Caching** |
| CDN Edge Cache (e.g., Varnish, Nginx, CloudFront) | Uses LRU/LFU/TinyLFU variants; often combined with *cache‑key* hashing and *stale‑while‑revalidate*. | Optimized for HTTP objects, high read‑to‑write ratio. |
| In‑Memory Data Grid (e.g., Redis, Memcached) | Often simple LRU or LFU, sometimes combined with *eviction policies per‑key* (TTL, max‑memory‑policy). | Low‑latency key‑value store; eviction policy chosen based on workload (e.g., LFU for “hot” keys). |
| Multi‑Level CPU Cache (L1/L2/L3) | L1: usually LRU or pseudo‑LRU; L2/L3 may use a mixture of LRU, PLRU, or adaptive algorithms (ARC‑like). | Balances speed vs. area/power; L1 is tiny and fast, L2/L3 are larger and tolerate slightly higher miss latency. |
| **Specialized Schemes** |
| **Cache‑Stampede Mitigation** (e.g., *request coalescing*, *dog‑pile* protection) | Prevent many concurrent requests from all trying to recompute a missing entry. | Often implemented by “lock‑around‑populate” or “early‑expiration + stale‑while‑revalidate”. |
| **Probabilistic Caching** (e.g., *Cache‑Random*, *ProbCache*) | Insert an item with probability *p* (often based on item size or popularity). | Useful when the cache is heavily over‑subscribed (e.g., CDN edge with many small objects). |
| **Priority / Weighted Caching** | Assign a weight/priority to each key; eviction respects weight (e.g., *Weighted‑LRU*). | Allows business‑logic control (premium content stays longer). |
| **Hybrid / Multi‑Metric** | Combine recency, frequency, size, cost of recomputation, TTL, etc., into a single score (e.g., *Greedy-Dual‑Size* (GDS), *GDSF*). | Popular in web proxies and CDNs; balances “big but rarely accessed” vs. “small but hot”. |
| **Write‑Back Buffer / NVRAM Cache** | Uses a small non‑volatile buffer to guarantee durability of writes before they reach the main store. | Emerging in storage systems (e.g., Intel Optane as a write‑back cache for SSD/HDD). |

---

## How to Choose a Scheme

| **Workload characteristic** | **Good fit** | **Why** |
|------------------------------|--------------|----------|
| **High temporal locality, moderate size** | LRU, Clock, FIFO | Simple, cheap, recency captures most reuse. |
| **Strong frequency locality (few “hot” items)** | LFU / TinyLFU / GDSF | Keeps the truly popular objects regardless of when they were last accessed. |
| **Mixed recency & frequency, changing patterns** | ARC, CAR, 2Q, SLRU | Adaptively shifts between LRU‑like and LFU‑like behavior. |
| **Very large cache, low per‑item overhead required** | Random, Clock, Approximate LFU (Count‑Min) | O(1) metadata, minimal memory. |
| **Write‑intensive with occasional reads** | Write‑Around + Write‑Through, or Write‑Back with Write‑Allocate | Avoids polluting cache with write‑only data. |
| **Multi‑core CPU / shared‑memory system** | MESI / MOESI + pseudo‑LRU | Guarantees coherence while keeping replacement cheap. |
| **Web‑content distribution (CDN)** | TinyLFU, GDSF + TTL + Stale‑while‑revalidate | Handles massive catalog, differing object sizes, and freshness constraints. |
| **Key‑value store with many short‑lived objects** | Random / LRU with size‑aware eviction (e.g., LRU‑2Q) | Prevents “cache‑thrash” from bursts of unique keys. |
| **Need to protect against cache stampede** | Dog‑pile protection + probabilistic insertion | Guarantees only one recomputation per miss. |

---

## Quick Reference Cheat‑Sheet (Pseudo‑code)

Below are ultra‑light snippets that illustrate the core mechanics of a few popular policies. They are language‑agnostic; you can adapt them to C, Java, Go, Rust, etc.

### 1. Classic LRU (hash‑map + doubly linked list)

```text
struct Node { key, value, prev, next }
map = {}               // key -> Node*
head, tail = nil       // most‑recent <-> least‑recent

function get(k):
    if k not in map: return MISS
    node = map[k]
    move_to_head(node)          // O(1)
    return node.value

function put(k, v):
    if k in map:
        node = map[k]; node.value = v
        move_to_head(node)
    else:
        if size == CAP: evict(tail)   // remove LRU
        node = new Node(k, v)
        insert_at_head(node)
        map[k] = node
```

### 2. Clock (Second‑Chance) – cheap LRU approximation

```text
struct Slot { key, value, ref_bit }
clock = [Slot] * N          // circular array
hand = 0

function get(k):
    // linear search or auxiliary hash for O(1) locate
    if not found: return MISS
    slot.ref_bit = 1
    return slot.value

function put(k, v):
    while true:
        slot = clock[hand]
        if slot.key == null:                // empty slot → use it
            slot.key, slot.value = k, v
            slot.ref_bit = 1
            break
        if slot.ref_bit == 0:               // victim
            evict(slot.key)
            slot.key, slot.value = k, v
            slot.ref_bit = 1
            break
        // give a second chance
        slot.ref_bit = 0
        hand = (hand + 1) % N
```

### 3. TinyLFU (approximate LFU) – used in many CDNs

```text
// Two data structures:
//   - Count‑Min Sketch (CMS) for frequency estimation (tiny memory)
//   - Admission filter (also a CMS) to decide if a candidate should replace a victim
//   - Victim chosen by a simple LRU list (or CLOCK)

function access(k):
    cms.increment(k)               // cheap O(1) update
    if k in cache:
        // hit → move to MRU in LRU list
        lru.move_to_front(k)
    else:
        // miss → candidate for admission
        if cache.is_full():
            victim = lru.tail()     // LRU victim
            // compare estimated frequencies
            if cms.freq(k) > cms.freq(victim.key):
                evict(victim.key)
                cache.insert(k)
                lru.push_front(k)
        else:
            cache.insert(k)
            lru.push_front(k)
```

### 4. ARC (high‑level description)

```text
T1 = LRU list of recent items
T2 = LRU list of frequent items
B1, B2 = “ghost” lists (store only keys) for items evicted from T1/T2

p = target size for T1 (adaptively changes)

on hit in T1 or T2:
    move item to front of T2 (promote to frequent)

on miss:
    if key in B1:                     // was recently evicted from T1
        p = min(c, p + max(1, |B2|/|B1|))
        replace()
        insert key into T2
    else if key in B2:                // was evicted from T2
        p = max(0, p - max(1, |B1|/|B2|))
        replace()
        insert key into T2
    else:
        // cold miss
        if cache is full: replace()
        insert key into T1

function replace():
    if |T1| > 0 && (|T1| > p || (key in B2 && |T1| == p)):
        victim = T1.tail()
        move victim.key to B1
        evict(victim)
    else:
        victim = T2.tail()
        move victim.key to B2
        evict(victim)
```

---

## Further Reading & Resources

| **Topic** | **Key Papers / Articles** | **Where to Find Implementations** |
|-----------|---------------------------|-----------------------------------|
| LRU, FIFO, Clock | “A Study of Replacement Algorithms for Virtual Memory” – Denning (1968) | Linux kernel, `LRUCache` in many libraries |
| LFU & Approximate LFU | “TinyLFU: A Highly Efficient Cache Admission Policy” – Bădoiu et al., 2013 | `caffeine` (Java), `cachetools` (Python) |
| ARC / CAR | “ARC: A Self‑Tuning, Low Overhead Replacement Cache” – Megiddo & Modha, 2003 | `bcachefs`, `zfs` ARC implementation |
| 2Q | “2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm” – Zhou & Ross, 2002 | `Redis` uses a variant for its LRU mode |
| GDSF / Greedy‑Dual | “Greedy-Dual-Size: A Flexible, Low Overhead Web Cache Replacement Algorithm” – Cao & Irani, 1997 | Varnish, Squid (web cache) |
| MESI / MOESI | Classic CPU architecture textbooks, Intel® 64 and IA‑32 Architectures Software Developer’s Manual | CPU micro‑architectures, gem5 simulator |
| Distributed Cache Patterns | “Designing Data‑Intensive Applications” – Kleppmann (Ch. 5) | AWS ElastiCache, Azure Cache for Redis |
| Cache‑Stampede Mitigation | “Dog‑Pile Effect” – Varnish docs, “Cache‑Aside with Stale‑while‑revalidate” – RFC 5861 | Cloudflare Workers KV, CDN edge solutions |

---

### TL;DR Takeaway

- **Replacement**: LRU is the workhorse, but for mixed workloads consider **ARC / CAR / 2Q / TinyLFU**.  
- **Write handling**: Choose **write‑through** for strong consistency, **write‑back** for performance‑critical write‑heavy workloads, and **write‑around** when writes seldom become reads.  
- **Coherence**: Multi‑core CPUs rely on **MESI‑family** protocols; distributed systems need **consistent hashing + versioning**.  
- **Hierarchy**: CPU caches (L1/L2/L3) use fast approximations (pseudo‑LRU, CLOCK), while CDNs and in‑memory stores can afford richer policies (TinyLFU, GDSF, ARC).  

Pick the algorithm that matches **the dominant locality (recency vs. frequency), the cost of a miss, and the overhead you can afford**. Most modern libraries let you swap the policy with a single configuration flag—experiment with a few (LRU → TinyLFU → ARC) on a realistic workload and measure hit‑rate vs. CPU/memory overhead to find the sweet spot. Happy caching!