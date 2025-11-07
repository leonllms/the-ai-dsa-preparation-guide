# Matrix formats

## Executive Summary & High‑Level Comparison

| Format | Typical Storage | Core Idea | Best‑Fit Workloads | Strengths | Weaknesses | Natural Distributed‑Memory Sharding |
|--------|----------------|-----------|--------------------|-----------|------------|--------------------------------------|
| **COO (Coordinate)** | Three 1‑D arrays: `row[]`, `col[]`, `val[]` | Store every non‑zero as an explicit tuple | Construction, very irregular sparsity, graph algorithms, dynamic insert/delete | Simple to build, easy to modify, good for I/O | Redundant storage of indices, poor cache‑locality for mat‑vec, duplicate entries must be reduced | Row‑ or column‑wise partition of the tuple list; each rank holds a subset of (row,col,val) pairs. |
| **CSR (Compressed Sparse Row)** | `row_ptr[]` (size = n_rows+1), `col_idx[]`, `val[]` | Row offsets compress column/values per row | SpMV, triangular solves, row‑wise kernels, CPU & GPU | Very fast row‑wise access, compact, widely supported | Column access is expensive, poor for column‑oriented ops | **Row‑block** distribution: each process owns a contiguous block of rows → local `row_ptr` and corresponding `col_idx/val`. |
| **CSC (Compressed Sparse Column)** | `col_ptr[]`, `row_idx[]`, `val[]` | Column offsets compress row/values per column | Column‑wise kernels (e.g., back‑substitution), some GPU libraries | Fast column access, good for transpose‑heavy ops | Row access expensive, similar to CSR but column‑major | **Column‑block** distribution: each rank owns a set of columns → local `col_ptr` and associated data. |
| **DIA (Diagonal)** | `diag_offset[]`, `diag_val[][]` | Store only diagonals that contain non‑zeros | Structured banded matrices, finite‑difference stencils | Very memory‑efficient for banded matrices, contiguous memory → vectorised | Only works when number of distinct diagonals ≪ nnz; padding waste for irregular sparsity | Partition by rows (or columns) – each block keeps the same diagonal offsets; communication only for halo rows/cols crossing block boundaries. |
| **ELL (ELLPACK)** | `indices[n_rows][max_nnz_per_row]`, `values[n_rows][max_nnz_per_row]` | Pad each row to the same length | GPUs & SIMD where regular memory access matters | Excellent coalescing on GPUs, simple indexing | Padding can waste memory heavily if row lengths vary | **Row‑block** distribution; each block stores its own padded rows. Padding is local, no cross‑rank data needed. |
| **BSR (Block CSR)** | `row_ptr[]`, `col_idx[]`, `block_vals[]` (blocks stored dense) | Group non‑zeros into fixed‑size dense blocks (e.g., 4×4) | Matrices with block structure (e.g., FEM, CFD), cache‑friendly | Reduced index overhead, better cache reuse, SIMD friendly | Requires a regular block pattern; block size choice critical | **2‑D block** distribution (grid of processes) – each rank owns a sub‑matrix of blocks; communication follows a 2‑D SUMMA‑style pattern. |
| **HYB (Hybrid CSR+ELL)** | Separate ELL part + CSR remainder | Combine ELL for rows with similar length + CSR for outliers | GPUs where most rows are dense but a few are long | Retains ELL’s performance while avoiding extreme padding | Slightly more complex kernel; two kernels needed | Same sharding strategies as CSR/ELL – typically row‑block; each rank keeps its own ELL + CSR slices. |

**Take‑away:**  
* For **general‑purpose CPU code** CSR (or CSC if column‑wise operations dominate) is the default.  
* For **GPU / SIMD** workloads, ELL or HYB (or CSR with warp‑level optimisations) give the best memory‑coalescing.  
* For **structured banded or stencil** problems, DIA or BSR dramatically cut storage and improve cache behaviour.  
* **Distributed memory** (MPI, multi‑node) is easiest with CSR/ELL (row‑block) or CSC (column‑block). BSR naturally maps to a 2‑D block grid, while COO can be split arbitrarily but requires a reduction step to eliminate duplicates.

---

# 1. COO (Coordinate) Format

## 1.1 What it does
COO stores every non‑zero entry as a triple *(row, column, value)*. The three 1‑D arrays have equal length `nnz`. No compression – the format is essentially a list of points in the matrix.

## 1.2 Algorithmic view (Sparse‑Matrix‑Vector Multiply)

```
y = 0
for k = 0 … nnz-1
    y[row[k]] += val[k] * x[col[k]]
```

The algorithm is embarrassingly parallel – each non‑zero can be processed independently. However, concurrent writes to the same `y[i]` cause race conditions; a common solution is to use atomic adds or a two‑phase reduction (sort by row first).

## 1.3 Simple templated C++ implementation

```cpp
// coo.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

template<class Index = int, class Value = double>
class SparseCOO {
public:
    std::vector<Index> rows;   // size = nnz
    std::vector<Index> cols;   // size = nnz
    std::vector<Value> vals;   // size = nnz
    Index n_rows = 0, n_cols = 0;

    // ----- construction -------------------------------------------------
    SparseCOO(Index nr, Index nc) : n_rows(nr), n_cols(nc) {}

    void push_back(Index i, Index j, Value v) {
        rows.push_back(i);
        cols.push_back(j);
        vals.push_back(v);
    }

    // ----- optional: sort by row (helps reduction) --------------------
    void sort_by_row() {
        std::vector<Index> perm(rows.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(),
            [&](Index a, Index b){ return rows[a] < rows[b]; });

        apply_permutation(rows, perm);
        apply_permutation(cols, perm);
        apply_permutation(vals, perm);
    }

    // ----- matrix‑vector product (single‑threaded) --------------------
    // y = A * x
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_cols));
        std::vector<Value> y(n_rows, Value(0));
        for (size_t k = 0; k < vals.size(); ++k)
            y[ rows[k] ] += vals[k] * x[ cols[k] ];
        return y;
    }

private:
    // helper to permute a vector in place
    template<class T>
    static void apply_permutation(std::vector<T>& vec,
                                  const std::vector<Index>& perm) {
        std::vector<T> tmp(vec.size());
        for (size_t i = 0; i < perm.size(); ++i)
            tmp[i] = vec[ perm[i] ];
        vec.swap(tmp);
    }
};
```

*Only the essential operations are shown – real‑world code would add bounds checks, duplicate‑entry reduction, and parallel kernels (OpenMP, CUDA, etc.).*

## 1.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| Extremely easy to build, insert, delete entries (just push a tuple). | Stores two indices per non‑zero → 2× overhead compared with CSR/CSC. |
| Naturally suits graph‑oriented algorithms (edge list). | Poor data locality for SpMV → many random accesses to `y`. |
| Sorting by row or column is trivial, enabling easy conversion to CSR/CSC. | Duplicate entries must be summed manually; otherwise the matrix semantics are ambiguous. |

## 1.5 When to use

* During **matrix assembly** (e.g., finite‑element mesh generation) where the non‑zeros appear in arbitrary order.  
* When the sparsity pattern is **highly dynamic** (insertion/deletion each iteration).  
* For **graph algorithms** where edges are processed independently.

## 1.6 Sharding / Distributed‑Memory Strategies

* **Row‑wise tuple partition:** each MPI rank receives a subset of the `(i,j,v)` triples whose row index falls in its local range.  
* **Column‑wise tuple partition** is also possible (useful when the algorithm needs column access).  
* After partitioning, each rank can locally convert its subset to CSR/CSC for fast local SpMV.  
* **Duplicate handling across ranks:** if the same `(i,j)` appears on two ranks, a global reduction (e.g., `MPI_Reduce_scatter` on a hashed key) is required before the final multiplication.  

Because the format is just a list, it maps trivially onto **MapReduce‑style** pipelines: `map` emits `(i, (j,v))` pairs, `shuffle` groups by `i`, and `reduce` sums duplicates.

---

# 2. CSR (Compressed Sparse Row)

## 2.1 What it does

| Array | Meaning |
|-------|---------|
| `row_ptr` (size `n_rows+1`) | `row_ptr[i]` points to the start of row *i* in `col_idx`/`val`. |
| `col_idx` (size `nnz`) | Column index of each stored element. |
| `val` (size `nnz`) | Value of each stored element. |

Rows are stored consecutively; the number of non‑zeros per row is implicit via the differences of successive entries in `row_ptr`.

## 2.2 Algorithmic view (SpMV)

```
for i = 0 … n_rows-1
    sum = 0
    for k = row_ptr[i] … row_ptr[i+1]-1
        sum += val[k] * x[col_idx[k]]
    y[i] = sum
```

The outer loop is trivially parallel (each row independent). Memory accesses to `x` are indirect but usually cache‑friendly because rows tend to reuse nearby columns.

## 2.3 Simple templated C++ implementation

```cpp
// csr.hpp
#pragma once
#include <vector>
#include <cassert>
#include <algorithm>

template<class Index = int, class Value = double>
class SparseCSR {
public:
    Index n_rows = 0, n_cols = 0;
    std::vector<Index> row_ptr;   // size n_rows+1
    std::vector<Index> col_idx;   // size nnz
    std::vector<Value> val;       // size nnz

    // ----- construction -------------------------------------------------
    explicit SparseCSR(Index nr, Index nc) : n_rows(nr), n_cols(nc) {
        row_ptr.assign(n_rows + 1, 0);
    }

    // Build from COO (simple but O(nnz log nnz) if needed)
    template<class Index2, class Value2>
    void from_coo(const std::vector<Index2>& rows,
                  const std::vector<Index2>& cols,
                  const std::vector<Value2>& vals) {
        assert(rows.size() == cols.size() && cols.size() == vals.size());
        const size_t nnz = rows.size();
        col_idx.resize(nnz);
        val.resize(nnz);
        // 1) count per‑row entries
        for (size_t i = 0; i < nnz; ++i)
            ++row_ptr[ rows[i] + 1 ];
        // 2) prefix sum -> row_ptr
        for (Index i = 0; i < n_rows; ++i)
            row_ptr[i+1] += row_ptr[i];
        // 3) temporary copy of row_ptr to use as write pointer
        std::vector<Index> next = row_ptr;
        for (size_t i = 0; i < nnz; ++i) {
            Index r = rows[i];
            Index dest = next[r]++;
            col_idx[dest] = cols[i];
            val[dest]     = vals[i];
        }
        // (optional) sort columns inside each row
        for (Index r = 0; r < n_rows; ++r) {
            Index beg = row_ptr[r];
            Index end = row_ptr[r+1];
            // simple insertion sort because rows are usually short
            for (Index i = beg+1; i < end; ++i) {
                Index j = i;
                while (j > beg && col_idx[j-1] > col_idx[j]) {
                    std::swap(col_idx[j-1], col_idx[j]);
                    std::swap(val[j-1],     val[j]);
                    --j;
                }
            }
        }
    }

    // ----- matrix‑vector product (single‑threaded) --------------------
    // y = A * x
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_cols));
        std::vector<Value> y(n_rows, Value(0));
        for (Index i = 0; i < n_rows; ++i) {
            Value sum = 0;
            for (Index k = row_ptr[i]; k < row_ptr[i+1]; ++k)
                sum += val[k] * x[ col_idx[k] ];
            y[i] = sum;
        }
        return y;
    }

    // ----- row‑wise access ------------------------------------------------
    // returns a pair of iterators to (col, val) for row i
    auto row(Index i) const {
        return std::make_pair(
            col_idx.begin() + row_ptr[i],
            col_idx.begin() + row_ptr[i+1]);
    }
};
```

The implementation includes a simple `from_coo` builder, optional per‑row sorting, and a basic `multiply` routine.

## 2.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| **Very compact** – only one index per non‑zero. | Column‑wise operations (e.g., `Aᵀ * x`) are inefficient; need explicit transpose or CSC. |
| Fast **row‑wise** traversal → excellent for SpMV, triangular solves, ILU factorisation. | Random access to `x` can cause cache misses if column indices are scattered. |
| Widely supported (Intel MKL, cuSPARSE, Eigen, etc.). | Insertion/deletion of entries is expensive (requires rebuilding). |
| Simple **row‑block** domain decomposition for distributed memory. | Not well suited for matrices with many rows of highly variable length on GPUs (padding needed). |

## 2.5 When to use

* **CPU‑centric kernels** where the dominant operation is `y = A * x`.  
* **Iterative solvers** (CG, GMRES, BiCGSTAB) that repeatedly multiply by the matrix and its transpose (transpose handled by a separate CSC copy).  
* **Finite‑element assembly** after the matrix is assembled and fixed.  

## 2.6 Sharding / Distributed‑Memory Strategies

### 2.6.1 Row‑block distribution (the most common)

* **Partition** the global rows into contiguous chunks. Rank `p` stores rows `[r_start, r_end)`.  
* Local data: `row_ptr_local` (size = local_rows+1), `col_idx_local`, `val_local`.  
* **Communication pattern for SpMV**: each rank needs the entries of the input vector `x` whose column indices appear in its local rows.  
  * Build a **halo list** of needed remote columns (`col_idx` values that lie outside the rank’s owned columns).  
  * Exchange necessary parts of `x` via `MPI_Neighbor_alltoallv` or a custom `AllGather`.  
* After the halo exchange, each rank can perform its local SpMV independently.  

### 2.6.2 2‑D block (for BSR or when scaling beyond a few hundred ranks)

* Split rows into `P_r` groups and columns into `P_c` groups → a `P_r × P_c` process grid.  
* Each rank owns a sub‑matrix of CSR rows *and* a sub‑set of columns.  
* Communication becomes a **SUMMA**‑style broadcast of needed `x` segments across process rows, and a reduction of partial `y` contributions across process columns.  

### 2.6.3 Load‑balancing considerations

* If rows have highly variable nnz, simple contiguous partitioning may lead to imbalance.  
* **Weighted partitioning** (e.g., using METIS or a simple greedy algorithm on `row_ptr[i+1]-row_ptr[i]`) can produce more even work distribution.  

---

# 3. CSC (Compressed Sparse Column)

## 3.1 What it does

| Array | Meaning |
|-------|---------|
| `col_ptr` (size `n_cols+1`) | Start index of column *j* in `row_idx`/`val`. |
| `row_idx` (size `nnz`) | Row index of each stored element. |
| `val` (size `nnz`) | Value of each stored element. |

The transpose of CSR; ideal when column‑wise access is required.

## 3.2 Algorithmic view (Sparse‑Matrix‑Vector Multiply)

For `y = A * x` we traditionally use CSR, but CSC shines for `y = Aᵀ * x`:

```
y = 0
for j = 0 … n_cols-1
    for k = col_ptr[j] … col_ptr[j+1]-1
        y[ row_idx[k] ] += val[k] * x[j]
```

## 3.3 Simple templated C++ implementation

```cpp
// csc.hpp
#pragma once
#include <vector>
#include <cassert>
#include <algorithm>

template<class Index = int, class Value = double>
class SparseCSC {
public:
    Index n_rows = 0, n_cols = 0;
    std::vector<Index> col_ptr;   // size n_cols+1
    std::vector<Index> row_idx;   // size nnz
    std::vector<Value> val;       // size nnz

    explicit SparseCSC(Index nr, Index nc) : n_rows(nr), n_cols(nc) {
        col_ptr.assign(n_cols + 1, 0);
    }

    // Build from COO (similar to CSR version)
    template<class Index2, class Value2>
    void from_coo(const std::vector<Index2>& rows,
                  const std::vector<Index2>& cols,
                  const std::vector<Value2>& vals) {
        const size_t nnz = rows.size();
        row_idx.resize(nnz);
        val.resize(nnz);
        // 1) count per‑column entries
        for (size_t i = 0; i < nnz; ++i)
            ++col_ptr[ cols[i] + 1 ];
        // 2) prefix sum
        for (Index j = 0; j < n_cols; ++j)
            col_ptr[j+1] += col_ptr[j];
        // 3) fill
        std::vector<Index> next = col_ptr;
        for (size_t i = 0; i < nnz; ++i) {
            Index c = cols[i];
            Index dest = next[c]++;
            row_idx[dest] = rows[i];
            val[dest] = vals[i];
        }
        // optional per‑column sort by row index
        for (Index c = 0; c < n_cols; ++c) {
            Index beg = col_ptr[c];
            Index end = col_ptr[c+1];
            for (Index i = beg+1; i < end; ++i) {
                Index j = i;
                while (j > beg && row_idx[j-1] > row_idx[j]) {
                    std::swap(row_idx[j-1], row_idx[j]);
                    std::swap(val[j-1],     val[j]);
                    --j;
                }
            }
        }
    }

    // Multiply with the transpose: y = Aᵀ * x
    std::vector<Value> multiply_transpose(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_rows));
        std::vector<Value> y(n_cols, Value(0));
        for (Index c = 0; c < n_cols; ++c) {
            Value sum = 0;
            for (Index k = col_ptr[c]; k < col_ptr[c+1]; ++k)
                sum += val[k] * x[ row_idx[k] ];
            y[c] = sum;
        }
        return y;
    }
};
```

## 3.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| Fast **column** access → ideal for `Aᵀ * x`, column scaling, column‑wise preconditioners. | Row access (standard SpMV) is indirect and slower. |
| Same memory compactness as CSR (one index per non‑zero). | Requires a separate CSC copy if both row‑ and column‑wise operations are frequent, doubling storage. |
| Simple **column‑block** partitioning for distributed memory. | Insertion/deletion are expensive (needs rebuild). |

## 3.5 When to use

* Algorithms that repeatedly need the **transpose** of a matrix (e.g., BiCGSTAB, certain graph centrality measures).  
* Column‑wise scaling or normalization (e.g., stochastic matrices).  
* When the matrix naturally arises column‑by‑column (e.g., assembling from column‑oriented data).  

## 3.6 Sharding / Distributed‑Memory Strategies

### Column‑block distribution

* Each MPI rank owns a contiguous set of columns `[c_start, c_end)`.  
* Local data: `col_ptr_local`, `row_idx_local`, `val_local`.  
* For `y = A * x`, each rank needs the whole vector `x` (because rows can reference any column). However, **only the columns it owns** contribute to its part of `y`.  
* **Communication pattern:**  
  * **All‑gather** the input vector `x` (or a subset if you know a column’s rows are limited).  
  * Compute local contributions to the global result `y`.  
  * **Reduce‑scatter** the partial `y` contributions across the row dimension (if each rank only holds a subset of rows).  

### Hybrid row/column block (2‑D grid)

* Combine CSR row‑blocks with CSC column‑blocks to create a **2‑D checkerboard** layout.  
* Each process stores the sub‑matrix intersecting its row‑block and column‑block, typically in **COO** or **BSR** internally.  
* This layout reduces both the amount of communicated `x` and `y` data, at the cost of extra bookkeeping.  

---

# 4. DIA (Diagonal) Format

## 4.1 What it does

DIA stores only the **distinct diagonals** that contain non‑zeros.

| Array | Meaning |
|-------|---------|
| `offsets[d]` (size `ndiag`) | Integer offset `j - i` for diagonal *d* (positive → super‑diagonal). |
| `data[ndiag][max_len]` | Dense values for each diagonal; rows that exceed matrix bounds are padded with zeros. `max_len` = `n_rows` (or `n_cols`) – the longest diagonal length. |

If a matrix has *p* distinct diagonals, storage is `O(p * max(n_rows, n_cols))`.

## 4.2 Algorithmic view (SpMV)

```
y = 0
for d = 0 … ndiag-1
    off = offsets[d]
    for i = 0 … n_rows-1
        j = i + off
        if (0 <= j < n_cols)
            y[i] += data[d][i] * x[j];
```

Because each diagonal is stored contiguously, the inner loop can be vectorised.

## 4.3 Simple templated C++ implementation

```cpp
// dia.hpp
#pragma once
#include <vector>
#include <cassert>
#include <algorithm>

template<class Index = int, class Value = double>
class SparseDIA {
public:
    Index n_rows = 0, n_cols = 0;
    std::vector<Index> offsets;               // size = ndiag
    std::vector<std::vector<Value>> data;     // [ndiag][n_rows] (padded)

    explicit SparseDIA(Index nr, Index nc) : n_rows(nr), n_cols(nc) {}

    // Build from COO – keep only distinct diagonals
    template<class Index2, class Value2>
    void from_coo(const std::vector<Index2>& rows,
                  const std::vector<Index2>& cols,
                  const std::vector<Value2>& vals) {
        assert(rows.size() == cols.size() && cols.size() == vals.size());

        // 1) discover unique offsets
        std::vector<Index> tmp_offsets;
        tmp_offsets.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            tmp_offsets.push_back(static_cast<Index>(cols[i] - rows[i]));
        std::sort(tmp_offsets.begin(), tmp_offsets.end());
        auto it = std::unique(tmp_offsets.begin(), tmp_offsets.end());
        offsets.assign(tmp_offsets.begin(), it);
        const size_t ndiag = offsets.size();

        // 2) allocate dense storage (padded with zeros)
        data.assign(ndiag, std::vector<Value>(n_rows, Value(0)));

        // 3) scatter values into the appropriate diagonal
        for (size_t k = 0; k < rows.size(); ++k) {
            Index i = rows[k];
            Index j = cols[k];
            Index off = j - i;
            // locate diagonal index (binary search because offsets are sorted)
            Index d = static_cast<Index>(std::lower_bound(offsets.begin(),
                                                          offsets.end(),
                                                          off) - offsets.begin());
            assert(d < static_cast<Index>(ndiag) && offsets[d] == off);
            data[d][i] = static_cast<Value>(vals[k]);   // assumes at most one entry per (i,j)
        }
    }

    // Multiply: y = A * x
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_cols));
        std::vector<Value> y(n_rows, Value(0));
        for (size_t d = 0; d < offsets.size(); ++d) {
            Index off = offsets[d];
            const auto& diag = data[d];
            for (Index i = 0; i < n_rows; ++i) {
                Index j = i + off;
                if (j >= 0 && j < n_cols)
                    y[i] += diag[i] * x[j];
            }
        }
        return y;
    }
};
```

The implementation assumes **at most one value per diagonal position** – true for most banded matrices.

## 4.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| Extremely compact for **banded** or **stencil** matrices (storage ≈ `num_diagonals * n`). | Fails when the number of distinct diagonals grows with `nnz` (e.g., unstructured sparse matrices). |
| Diagonal data is stored **contiguously**, enabling SIMD/vectorisation. | Padding for short diagonals can waste memory if the matrix is highly irregular. |
| Simple to parallelise: each diagonal can be processed independently. | Random access to `x` (offsets) may still cause cache misses if the bandwidth is large. |
| Very easy to **partition** – each process can keep the same set of offsets. | No built‑in support for dynamic insertion of new diagonals; requires full rebuild. |

## 4.5 When to use

* Finite‑difference discretisations on regular grids (e.g., 1‑D/2‑D/3‑D Laplacian).  
* Tridiagonal, pentadiagonal, or generally **n‑band** matrices.  
* Problems where the bandwidth is known a priori and is small relative to matrix dimensions.

## 4.6 Sharding / Distributed‑Memory Strategies

### Row‑block (or column‑block) sharding

* Because every diagonal spans the whole matrix, each rank can store **all diagonals** but only a subset of rows.  
* For rank `p` owning rows `[r0, r1)`, allocate `data[d][r0…r1-1]` (a slice of each diagonal).  
* Communication: during SpMV each rank needs the portion of the input vector `x` that corresponds to columns `j = i + offset[d]` for its local rows. These columns may belong to neighboring ranks → exchange **halo regions** of `x`.  
* The amount of halo data is bounded by `num_diagonals * halo_width` (where `halo_width` is the maximal absolute offset). For a banded matrix with half‑bandwidth `b`, each rank needs at most `b` extra elements on each side.  

### 2‑D block for very wide bandwidth

* Split the matrix into a 2‑D grid of sub‑blocks; each block stores its local slice of each diagonal (many entries will be zero).  
* More complex but reduces the size of halo exchanges when the bandwidth is comparable to `n`.  

---

# 5. ELL (ELLPACK) / ELLPACK‑IT

## 5.1 What it does

* Determine `max_nnz_per_row = max_i (nnz in row i)`.  
* Allocate two dense 2‑D arrays of size `n_rows × max_nnz_per_row`:
  * `indices[i][k]` – column index of the *k*‑th entry in row *i* (or `-1`/sentinel for padded slots).  
  * `values[i][k]` – corresponding value (zero for padded slots).  

Rows are padded to the same length, enabling regular memory accesses.

## 5.2 Algorithmic view (SpMV)

```
for i = 0 … n_rows-1
    sum = 0
    for k = 0 … max_nnz_per_row-1
        j = indices[i][k]
        if (j >= 0)               // skip padding
            sum += values[i][k] * x[j]
    y[i] = sum
```

On GPUs each warp can load a whole row efficiently because memory accesses are coalesced.

## 5.3 Simple templated C++ implementation

```cpp
// ell.hpp
#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>

template<class Index = int, class Value = double>
class SparseELL {
public:
    Index n_rows = 0, n_cols = 0;
    Index max_nnz_per_row = 0;
    std::vector<Index> indices;   // flattened: size = n_rows * max_nnz_per_row
    std::vector<Value> values;    // same size

    explicit SparseELL(Index nr, Index nc)
        : n_rows(nr), n_cols(nc) {}

    // Build from COO – determines max_nnz_per_row automatically
    template<class Index2, class Value2>
    void from_coo(const std::vector<Index2>& rows,
                  const std::vector<Index2>& cols,
                  const std::vector<Value2>& vals) {
        const size_t nnz = rows.size();
        // 1) count per‑row entries
        std::vector<Index> row_counts(n_rows, 0);
        for (size_t i = 0; i < nnz; ++i)
            ++row_counts[ rows[i] ];
        max_nnz_per_row = *std::max_element(row_counts.begin(),
                                             row_counts.end());

        indices.assign(n_rows * max_nnz_per_row,
                       std::numeric_limits<Index>::max()); // sentinel
        values.assign(n_rows * max_nnz_per_row, Value(0));

        // 2) temporary write pointers per row
        std::vector<Index> next(n_rows, 0);
        for (size_t i = 0; i < nnz; ++i) {
            Index r = rows[i];
            Index pos = r * max_nnz_per_row + next[r]++;
            indices[pos] = cols[i];
            values[pos]  = vals[i];
        }
        // Remaining slots stay as sentinel/zero (padding)
    }

    // Multiply y = A * x
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_cols));
        std::vector<Value> y(n_rows, Value(0));
        const Index sentinel = std::numeric_limits<Index>::max();
        for (Index i = 0; i < n_rows; ++i) {
            Value sum = 0;
            Index base = i * max_nnz_per_row;
            for (Index k = 0; k < max_nnz_per_row; ++k) {
                Index col = indices[base + k];
                if (col != sentinel)        // skip padding
                    sum += values[base + k] * x[col];
            }
            y[i] = sum;
        }
        return y;
    }
};
```

The sentinel value `std::numeric_limits<Index>::max()` marks padded entries; on GPU implementations a special “invalid” flag (e.g., `-1`) is often used.

## 5.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| **Regular memory layout** → excellent cache‑line utilization on CPUs and coalesced accesses on GPUs. | Padding can cause **large storage blow‑up** if row lengths vary widely. |
| Simple to parallelise: each row can be processed by a separate thread/warp. | Insertion/deletion requires rebuilding the whole structure (max row length may change). |
| Easy to vectorise (inner loop has constant stride). | Not suitable for matrices with a few extremely long rows (e.g., power‑law graphs). |
| Deterministic work per thread → good load balancing on SIMD. | The sentinel check adds a branch; on some GPUs a “mask” approach is used instead. |

## 5.5 When to use

* **GPU kernels** where the matrix sparsity pattern is roughly uniform (e.g., discretised PDEs on regular grids).  
* **SIMD‑friendly CPU kernels** where you need predictable per‑row work.  
* When you can afford the extra memory (e.g., `max_nnz_per_row` ≤ 2× average nnz per row).  

## 5.6 Sharding / Distributed‑Memory Strategies

### Row‑block distribution (standard)

* Each rank stores a contiguous block of rows → its own slice of the dense `indices` and `values` arrays.  
* Because each row is self‑contained, **no cross‑rank column dependencies** beyond the usual halo exchange of the input vector `x`.  
* **Halo size**: each rank must receive the entries of `x` referenced by its local rows; the amount of data is bounded by `local_rows * max_nnz_per_row`.  

### Hybrid ELL/CSR (HYB) for irregular rows

* Split matrix into an **ELL part** (rows whose length ≤ `max_nnz_per_row`) and a **CSR remainder** for outlier rows.  
* Distribute both parts using row‑block partitioning; each rank deals with its own ELL slice (regular) and CSR slice (irregular).  
* Communication pattern remains the same as CSR: exchange needed `x` entries.  

---

# 6. BSR (Block Compressed Sparse Row)

## 6.1 What it does

BSR groups non‑zeros into **dense blocks** of fixed size `B_R × B_C` (commonly `2×2`, `4×4`, `8×8`). The matrix is viewed as a grid of blocks; only blocks that contain at least one non‑zero are stored.

| Array | Meaning |
|-------|---------|
| `block_ptr` (size `n_block_rows+1`) | Offsets into `col_idx`/`block_vals` for each block‑row. |
| `col_idx` (size `nnz_blocks`) | Column index of each stored block. |
| `block_vals` (size `nnz_blocks × B_R × B_C`) | Dense storage of the block’s values (row‑major). |

Effectively a CSR where the “value” of each entry is a small dense matrix.

## 6.2 Algorithmic view (SpMV)

```
y = 0
for br = 0 … n_block_rows-1
    for b = block_ptr[br] … block_ptr[br+1]-1
        bc = col_idx[b]               // block column
        for i = 0 … B_R-1
            sum = 0
            for j = 0 … B_C-1
                sum += block_vals[b][i][j] * x[ bc*B_C + j ];
            y[ br*B_R + i ] += sum;
```

The inner two loops are a **dense matrix‑vector multiply** on a micro‑block, which can be vectorised or executed with SIMD intrinsics.

## 6.3 Simple templated C++ implementation

```cpp
// bsr.hpp
#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstring>   // for std::memcpy

template<class Index = int, class Value = double>
class SparseBSR {
public:
    Index n_rows = 0, n_cols = 0;          // full matrix dimensions
    Index B_R = 0, B_C = 0;                // block dimensions
    Index n_block_rows = 0, n_block_cols = 0;

    std::vector<Index> block_ptr;   // size n_block_rows+1
    std::vector<Index> col_idx;     // size nnz_blocks
    std::vector<Value> block_vals;  // size nnz_blocks * B_R * B_C (row‑major)

    explicit SparseBSR(Index nr, Index nc,
                       Index br, Index bc)
        : n_rows(nr), n_cols(nc), B_R(br), B_C(bc) {
        assert(n_rows % B_R == 0 && n_cols % B_C == 0);
        n_block_rows = n_rows / B_R;
        n_block_cols = n_cols / B_C;
        block_ptr.assign(n_block_rows + 1, 0);
    }

    // Insert a dense block at block‑row br, block‑col bc.
    // `blk` must point to a contiguous B_R*B_C array (row‑major).
    void insert_block(Index br, Index bc, const Value* blk) {
        // naive insertion: push at the end of the row
        // (real implementations keep a map to avoid duplicates)
        Index row_start = block_ptr[br];
        Index row_end   = block_ptr[br+1];
        // Append at the end of the row
        col_idx.insert(col_idx.begin() + row_end, bc);
        block_vals.insert(block_vals.begin() + row_end * B_R * B_C,
                         blk, blk + B_R * B_C);
        // Update block_ptr for this and later rows
        for (Index r = br + 1; r <= n_block_rows; ++r)
            ++block_ptr[r];
    }

    // Multiply y = A * x (both dense vectors)
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        assert(x.size() == static_cast<size_t>(n_cols));
        std::vector<Value> y(n_rows, Value(0));

        for (Index br = 0; br < n_block_rows; ++br) {
            for (Index b = block_ptr[br]; b < block_ptr[br+1]; ++b) {
                Index bc = col_idx[b];
                const Value* blk = &block_vals[b * B_R * B_C];
                for (Index i = 0; i < B_R; ++i) {
                    Value sum = Value(0);
                    for (Index j = 0; j < B_C; ++j) {
                        sum += blk[i * B_C + j] *
                               x[ bc * B_C + j ];
                    }
                    y[ br * B_R + i ] += sum;
                }
            }
        }
        return y;
    }
};
```

The implementation is deliberately simple (no duplicate‑block checking, no CSR‑style sorting). Production code would sort `col_idx` per block‑row and possibly store blocks in a small dense layout for SIMD.

## 6.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| **Cache‑friendly**: each block is a small dense matrix → good temporal locality. | Requires **regular block pattern**; if non‑zeros are scattered, many blocks become mostly zero → storage waste. |
| Reduces index overhead (one index per block instead of per element). | Block size selection is critical; too small → no benefit, too large → padding waste. |
| Enables **SIMD / vectorised micro‑kernels** (e.g., AVX, CUDA warp‑level GEMV). | Insertion/deletion is more complex (must maintain block‑grid). |
| Naturally maps to **2‑D process grids** (each rank owns a sub‑matrix of blocks). | Not ideal for matrices with highly irregular row/column lengths (e.g., social networks). |

## 6.5 When to use

* Finite‑element or finite‑volume discretisations where the stencil naturally forms a small dense block (e.g., 3‑D vector fields → 3×3 blocks).  
* **GPU kernels** that profit from small dense GEMV inside each thread block.  
* Situations where the matrix has an **inherent block sparsity pattern** (e.g., multi‑physics coupling).  

## 6.6 Sharding / Distributed‑Memory Strategies

### 2‑D block decomposition (grid of processes)

* Partition the **block grid** into `P_r × P_c` tiles. Rank `(p_r, p_c)` owns block rows `[br0, br1)` and block columns `[bc0, bc1)`.  
* Local data: `block_ptr_local` (for its block rows), `col_idx_local`, `block_vals_local`.  
* **Communication** for SpMV: each rank needs the parts of the input vector `x` that correspond to its remote block columns.  
  * Perform a **row‑wise broadcast** of `x` segments across process columns (MPI `Bcast` within each row of the process grid).  
  * After local multiplication, each rank holds partial contributions to the output vector `y` for its block rows; a **column‑wise reduction** (MPI `Reduce_scatter`) combines these contributions.  
* This pattern mirrors the classic **SUMMA** algorithm used for dense matrix multiplication and scales well to thousands of nodes.

### Row‑block (fallback)

* If the block grid is highly rectangular, a simple **row‑block** distribution (each rank gets a contiguous set of block rows) also works; communication reduces to a single all‑gather of needed `x` segments per iteration.

---

# 7. HYB (Hybrid CSR + ELL)

## 7.1 What it does

Hybrid format combines the **regular part** of the matrix (rows whose length ≤ `k`) stored in **ELL**, with the **irregular tail** stored in **CSR**. The parameter `k` is typically chosen as the average or 2‑times average nnz per row.

* **ELL part:** fast, vectorised, ideal for GPU kernels.  
* **CSR part:** handles the few rows that would cause excessive padding in pure ELL.

## 7.2 Algorithmic view (SpMV)

```
y = 0
// 1) ELL phase (vectorised)
for i = 0 … n_rows-1
    for k = 0 … K-1
        j = ell_indices[i][k];
        if (j != sentinel) y[i] += ell_values[i][k] * x[j];

// 2) CSR phase (irregular rows)
for i = 0 … n_rows-1
    for p = csr_row_ptr[i] … csr_row_ptr[i+1]-1
        y[i] += csr_val[p] * x[ csr_col_idx[p] ];
```

On GPU the two phases can be launched as separate kernels or fused with conditional logic.

## 7.3 Simple templated C++ implementation

```cpp
// hyb.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include "ell.hpp"
#include "csr.hpp"

template<class Index = int, class Value = double>
class SparseHYB {
public:
    SparseELL<Index, Value> ell;
    SparseCSR<Index, Value> csr;

    explicit SparseHYB(Index nr, Index nc, Index ell_k)
        : ell(nr, nc), csr(nr, nc) {
        // ell.k is set later by building from COO
        (void)ell_k; // placeholder
    }

    // Build from COO, choosing K = ell_k (max entries kept in ELL)
    template<class Index2, class Value2>
    void from_coo(const std::vector<Index2>& rows,
                  const std::vector<Index2>& cols,
                  const std::vector<Value2>& vals,
                  Index ell_k) {
        // 1) count per row
        std::vector<Index> cnt(rows.size());
        std::vector<Index> row_nnz(ell.n_rows, 0);
        for (size_t i = 0; i < rows.size(); ++i)
            ++row_nnz[ rows[i] ];
        // 2) decide which rows go to ELL (those with <= ell_k nnz)
        std::vector<char> to_ell(ell.n_rows, 0);
        for (Index r = 0; r < ell.n_rows; ++r)
            if (row_nnz[r] <= ell_k) to_ell[r] = 1;

        // 3) split data
        std::vector<Index> ell_rows, ell_cols;
        std::vector<Value> ell_vals;
        std::vector<Index> csr_rows, csr_cols;
        std::vector<Value> csr_vals;

        for (size_t i = 0; i < rows.size(); ++i) {
            Index r = rows[i];
            if (to_ell[r]) {
                ell_rows.push_back(r);
                ell_cols.push_back(cols[i]);
                ell_vals.push_back(vals[i]);
            } else {
                csr_rows.push_back(r);
                csr_cols.push_back(cols[i]);
                csr_vals.push_back(vals[i]);
            }
        }
        // 4) build each sub‑format
        ell.from_coo(ell_rows, ell_cols, ell_vals);
        csr.from_coo(csr_rows, csr_cols, csr_vals);
    }

    // Multiply y = A * x
    std::vector<Value> multiply(const std::vector<Value>& x) const {
        std::vector<Value> y = ell.multiply(x);
        std::vector<Value> y_csr = csr.multiply(x);
        for (size_t i = 0; i < y.size(); ++i) y[i] += y_csr[i];
        return y;
    }
};
```

The implementation re‑uses the `SparseELL` and `SparseCSR` classes defined earlier.

## 7.4 Benefits & Weaknesses

| Benefit | Weakness |
|---------|----------|
| Captures the **speed of ELL** for the bulk of rows while **avoiding massive padding** for outliers. | Slightly more complex data management (two formats, two kernels). |
| Works well on **GPUs** where a small irregular tail does not dominate performance. | The choice of `K` (threshold) is heuristic; a poor choice can degrade either memory usage or speed. |
| Retains CSR’s ability to insert/delete rows in the irregular part if needed. | Requires two passes over the matrix when building or converting. |

## 7.5 When to use

* Matrices with **moderately skewed row length distribution** (e.g., graph adjacency matrices where most vertices have degree ≈ 10 but a few have degree > 100).  
* **GPU‑accelerated solvers** where pure CSR kernels are slower due to irregular memory access.  
* Situations where you already have a CSR matrix and want to accelerate it without a full format conversion.

## 7.6 Sharding / Distributed‑Memory Strategies

* **Row‑block partition** of the entire HYB matrix: each rank owns a contiguous range of rows, thus owning both the ELL slice and the CSR slice for those rows.  
* Communication pattern identical to CSR: exchange needed pieces of `x` that correspond to columns referenced by local rows (both ELL and CSR parts).  
* Because the ELL part is regular, the halo exchange can be **pre‑computed** and reused across iterations, reducing overhead.  

---

# 8. Detailed Comparison

Below is a consolidated view that juxtaposes all formats on the most relevant axes for both **single‑node** and **distributed‑memory** contexts.

| Feature | COO | CSR | CSC | DIA | ELL | BSR | HYB |
|---------|-----|-----|-----|-----|-----|-----|-----|
| **Memory per non‑zero** | 2 indices + value | 1 index + value | 1 index + value | 1 diagonal offset per *distinct* diagonal + padded values | `max_nnz_per_row` indices + values (padded) | 1 block index + dense block (B_R·B_C) | Combination of CSR + ELL (see below) |
| **Construction cost** | O(nnz) (push) | O(nnz log nnz) (if sorting) or O(nnz) (if rows known) | Same as CSR | O(nnz) + O(num_diagonals) | O(nnz) + O(max_nnz_per_row) | O(nnz) (need to map each entry to a block) | Two‑phase: split then build CSR+ELL |
| **Row access** | O(nnz) linear scan | O(nnz_row) contiguous | O(nnz_row) via indirect column search | O(num_diagonals) per row (fast if few diags) | O(max_nnz_per_row) (includes padding) | O(nnz_block_row) (blocks) | Fast for ELL rows, CSR rows for outliers |
| **Column access** | O(nnz) linear scan | O(nnz) via indirect search (slow) | O(nnz_col) contiguous | O(num_diagonals) per column | O(max_nnz_per_row) (needs scan) | O(nnz_block_col) (blocks) | Same as row access (split) |
| **Best hardware** | CPU (flexible), graph‑processing | CPU (SpMV), many libraries | CPU for transpose‑heavy ops | CPU & GPU (structured PDE) | GPUs, SIMD CPUs | GPUs (block‑level GEMV), CPUs with vector units | GPUs (most common) |
| **Parallelism (shared‑mem)** | Easy (each nnz independent) | Row‑parallel (good load‑balance) | Column‑parallel (rare) | Diagonal parallel (SIMD) | Row‑parallel, constant work per thread | Block‑parallel (dense micro‑kernel) | Row‑parallel (ELL) + irregular CSR |
| **Distributed sharding** | Arbitrary tuple split (COO) | Row‑block (most common) | Column‑block | Row‑block (halo width = bandwidth) | Row‑block (same as CSR) | 2‑D block grid (process grid) | Row‑block (both parts) |
| **Typical use‑case** | Assembly, dynamic graphs | General sparse linear algebra | Transpose‑heavy algorithms | Banded/stencil PDEs | Uniform‑degree graphs, GPU kernels | Block‑structured PDEs, multi‑physics | Mixed‑degree graphs on GPUs |
| **Weaknesses** | Index overhead, poor cache for SpMV | Bad column access, rebuild needed to insert | Duplicate storage (need CSR for rows) | Only works for few diagonals | Memory blow‑up if rows vary | Requires regular block pattern | Needs tuning of `K` |
| **Strengths** | Simplicity, easy insertion | Compact, fast row traversal | Fast column traversal | Minimal storage for banded matrices | Regular memory layout, SIMD‑friendly | Index reduction, dense micro‑kernel performance | Balances speed & memory usage |

### Choosing a format – decision flow

1. **Is the sparsity pattern regular (banded, block‑structured, or uniform degree)?**  
   *Yes →* Prefer **DIA** (banded) or **BSR** (block) or **ELL** (uniform).  
2. **Do you need fast column access or frequent transposes?**  
   *Yes →* Use **CSC** (or keep a CSR + CSC pair).  
3. **Are you targeting GPUs / SIMD and the row lengths are roughly equal?**  
   *Yes →* **ELL** or **HYB** (if a few outliers exist).  
4. **Is the matrix assembled dynamically or does it change shape often?**  
   *Yes →* **COO** (or a hybrid where COO is used for assembly then converted).  
5. **Do you plan to run on many nodes with a 2‑D process grid?**  
   *Yes →* **BSR** (or CSR/CSC with 2‑D block distribution).  

---

# 9. Closing Remarks

Sparse matrix storage is not a one‑size‑fits‑all problem. The **right format** depends on a combination of:

* **Structural properties** (bandwidth, blockiness, row‑length variance).  
* **Target hardware** (CPU cache hierarchy, GPU warp size, SIMD width).  
* **Algorithmic demands** (row‑ vs column‑centric kernels, need for transpose).  
* **Parallelism model** (shared‑memory threads vs MPI‑based distributed memory).  

Understanding each format’s memory layout, access patterns, and sharding characteristics lets you design high‑performance linear‑algebra kernels that scale from a single GPU up to large clusters. The code snippets above provide a **starting point** for prototyping; production libraries (MKL, cuSPARSE, Eigen, KokkosKernels, PETSc) implement many of these ideas with aggressive optimisations, but the core concepts remain the same. 

Feel free to adapt the templated structures to your own data‑type requirements (e.g., `float`, `complex<double>`) or to embed them in higher‑level abstractions such as **matrix‑free operators** or **preconditioner objects**. Happy coding!