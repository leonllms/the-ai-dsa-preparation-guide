# 14. **Hash Table**

**Problem:**  
Design a hash table that stores **key‑value pairs** with **open addressing** (linear probing). Implement `insert`, `search`, and `delete` operations, handling **clustering** and **rehashing** when the load factor exceeds 0.7.


**Hash table with open addressing – linear probing**

| Component | What to do (in plain language) |
|-----------|--------------------------------|
| **Table layout** | Allocate an array `T[0 … m‑1]`. Each cell can be **empty**, **occupied** (stores a key‑value pair), or **deleted** (a *tombstone*). |
| **Hash function** | Compute an integer `h = H(key) mod m`. `H` should distribute keys uniformly (e.g., multiply‑shift or a built‑in 64‑bit hash). |
| **Insertion** | 1. Compute the start index `i = h`. <br>2. Scan forward (`i = (i+1) mod m`) until you find a cell that is **empty** or **deleted**. <br>3. Place the new pair there and increment the count of stored elements `n`. <br>4. If after the insertion `n / m > 0.7`, **rehash** (see below). |
| **Search** | 1. Compute `i = h`. <br>2. While the cell is **occupied** or **deleted**: <br> ‑ If the cell is **occupied** and its key equals the searched key → return the value. <br> ‑ Otherwise move to the next slot (`i = (i+1) mod m`). <br>3. If you hit an **empty** cell, the key is not present. |
| **Deletion** | 1. Locate the key with the same procedure as **search**. <br>2. If found, replace the entry with a **tombstone** (mark the slot “deleted”) and decrement `n`. <br>3. Do **not** stop the probe chain; tombstones let later searches skip over the removed slot. |
| **Clustering mitigation** | • Keep the load factor ≤ 0.7 (the rehash threshold). <br>• Use a good hash function that spreads keys uniformly, which reduces the chance that many keys start in the same region. <br>• When a tombstone is encountered during **insert**, you may remember the first tombstone seen and, after the probe finishes, place the new entry there (this “lazy‑cleaning” shortens future probe chains). |
| **Rehashing** (triggered when `n/m > 0.7`) | 1. Choose a new table size `m'` that is roughly twice the old size and preferably a prime number (or a power of two if the hash uses a power‑of‑two mask). <br>2. Allocate a fresh empty array `T'` of size `m'`. <br>3. Walk through the old table; for every **occupied** cell, recompute its hash modulo `m'` and insert it into `T'` using the normal linear‑probing insertion (tombstones are ignored). <br>4. Replace the old table with `T'` and reset the tombstone count. |
| **Complexity (average case)** | With load factor ≤ 0.7, the expected number of probes for any operation is ≤ ~2.5, i.e. **O(1)**. In the worst case (full table or pathological keys) linear probing can degrade to **O(m)**, but rehashing prevents that from happening in practice. |

**Summary of the design**

1. **Array + simple hash** → constant‑time index.  
2. **Linear probing** resolves collisions by walking forward until a free slot is found.  
3. **Tombstones** keep the probe sequence intact after deletions.  
4. **Load‑factor check (0.7)** triggers a **rehash** to a larger table, which both removes tombstones and restores low clustering.  
5. A good hash function and occasional clean‑up of tombstones keep clusters short, preserving the expected O(1) performance for `insert`, `search`, and `delete`.

## How it works

- Inserting start with computing the hash of the key, which will give the
  location ( address ) in the table where the value should be stored. If the
  slot in the table is occupied, then perform linear probing until a slot is
  found.
- Searching is the same process, by checking the stored key. 
- Removing the key is the same process, but we just mark-it deleted , 
    bc later slots might be occupied . Quick optimization, probe the next and
    if it is neither occupied , nor tombstoned , then the record can be cleared
    instead.

```python

class fastmap:

    _deathmark_ = (None, None)
    _max_no_cleanup_size_ = 10000000

    def __init__(self, size , hashfunction ):
        self._hashfun_ = hashfunction
        self._size_ = size
        self._inserted_ = 0
        self._occupied_ = 0
        self._dead_ = 0
        self._table_ = [None]*self._size_


    def _probe(self, key, 
        stop_onfound = True,
        stop_onoccupied = False,
        stop_ondead = False):

        size = self._size_
        address = 0
        def halt(next_keyhash):
            nonlocal address
            address = next_keyhash % size
            dst = self._table_[address]
            if dst is not None:
                slotkey, slotval = dst
                if dst == fastmap._deathmark_:
                    return stop_ondead
                if slotkey == key:
                    return stop_onfound
                if slotkey != key:
                    return stop_onoccupied
            return True

        keyhash = self._hashfun_(key)
        while not halt(keyhash):
            keyhash+=1

        return address

    def search(self, key):
        """
        skip forward until the key is found and then return the value

        skip rules: 

        stop_onfound = True , but the key was found
        stop_onfree = True , but the key was not found
        stop_ondead = False , skip
        stop_onoccupied = False, skip
        
        """
        address = self._probe(key, 
            stop_onfound = True,
            stop_ondead = False,
            stop_onoccupied = False)
        if self._table_[address] is not None:
            _, value_found = self._table_[address]
            return value_found
        return None


    def insert(self, key, value):
        """
        skip forward until an empty is found and the store the key and the value

        stop_onfound = True, but the key was found ( insert in place / overwrite ? )
        stop_onfree = True, and the key was not found ( add new value )
        stop_ondead = False, move forward
        stop_onoccupied = False , move forward

        """
        return self._place((key,value))

    def _place(self, record):
        """
        skip forward until an empty is found and the store the key and the value

        stop_onfound = True, but the key was found ( insert in place / overwrite ? )
        stop_onfree = True, and the key was not found ( add new value )
        stop_ondead = False, move forward
        stop_onoccupied = False , move forward

        """
        key, _ = record
        address = self._probe(key, 
            stop_onfound = True,
            stop_ondead = False,
            stop_onoccupied = False)

        self._table_[address] = record

        self._inserted_ += 1
        self._occupied_ += 1
        self._tidy_()
        return address

    def insert(self, key, value):
        """
        skip forward until an empty is found and the store the key and the value

        stop_onfound = True, but the key was found ( insert in place / overwrite ? )
        stop_ondead = False, move forward
        stop_onoccupied = False , move forward

        """

        address = self._probe(key, 
            stop_onfound = True,
            stop_ondead = False,
            stop_onoccupied = False)

        self._table_[address] = (key,value)

        self._inserted_ += 1
        self._occupied_ += 1
        self._tidy_()
        return address

    def delete(self, key):
        """
        skip forward until the key is found and then kill it by marking dead

        stop_onfound = True, the key can be deleted
        stop_ondead = False, move forward to find the key
        stop_onoccupied = False, move forward to find the key
        """
        address = self._probe(key, 
            stop_onfound = True,
            stop_ondead = False,
            stop_onoccupied = False)

        if self._table_[address] is not None:
            self._table_[address] = fastmap._deathmark_
            self._dead_ += 1
            self._inserted_ -= 1
        return address

    def _tidy_(self):
        """
        if we are past 1/4 of the population is dead and occupation_ratio > 3/4 
        cleanup otherwise grow the table
        """
        if (self._occupied_ / self._size_) > 3/4:
            if self._dead_ / self._occupied_ > 1/2:
                if self._size_ < fastmap._max_no_cleanup_size_:
                    self._grow_(self._size_)
                else:
                    self._cleanup_()
            else:
                return self._grow_()


    def _cleanup_(self):
        """

        Go over the table and for each dead key , probe until a free node is 
        found recording the occupied slots along the way. 

        Start replacing the chain with the keys found ressurecting the dead and
        until all keys are exhausted or no more keys to replace and resurrect the
        remaining dead.

        Need to implement for LARGE HMAP version
        """
        raise NotImplemented

    def _grow_(self, size = None):
        """
        Create a new table (double the size ) and copy over the keys, skiping the dead
        """

        if size is None:
            size=2*self._size_

        newmap = fastmap(size=size, hashfunction = self._hashfun_)

        for address in range(self._size_):
            slot = self._table_[address]
            if slot is not None and slot!= fastmap._deathmark_:
                newmap._place(slot)
                self._table_[address] = fastmap._deathmark_

        self._table_ = newmap._table_
        self._size_ = newmap._size_
        self._inserted_ = newmap._inserted_
        self._occupied_ = newmap._occupied_
        self._dead_ = 0


```

GPT-OSS-120B: 

Check if there are any errors , don't suggest improvements or judge the
efficiency of the algorithm , this is a naive solution, be very brief.


**Errors found**

1. **Duplicate method definitions** – `insert` is defined twice; the first version is never used and shadows the second one.

2. **`_probe` logic** – the `halt` function never checks for an empty slot (`dst is None`). As a result the loop may continue indefinitely when it reaches an empty cell.

3. **Unused parameters** – `stop_onoccupied` is passed but never examined inside `halt`; the comment mentions it but the code always returns `stop_onoccupied` (which is `False`) for a non‑matching occupied slot, causing the probe to continue even when it could stop.

4. **`_tidy_` call order** – `_tidy_` is invoked after every insertion, but it may call `_grow_` without first checking the load factor threshold (0.7). This is a logic error relative to the specification.

5. **`_grow_` size argument** – when `_tidy_` decides to grow because of high occupancy it calls `_grow_(self._size_)`, which creates a new table of the same size (no growth). It should pass a larger size (e.g., `2*self._size_`).

6. **`_cleanup_` placeholder** – the method raises `NotImplemented`; any call to it will crash.

7. **Counters mismatch** – `_occupied_` is incremented on every insertion even when a tombstone is overwritten; it should be incremented only when a previously empty slot is filled.

8. **`_probe` return value** – `address` is set only when `halt` returns `True`. If the loop exits because `halt` never returns `True` (e.g., table full), `address` remains 0, which is incorrect.

9. **`_deathmark_` comparison** – `dst == fastmap._deathmark_` works only because the tuple is immutable, but using `is` would be clearer and faster.

These are the concrete bugs that will cause incorrect behavior or runtime errors.