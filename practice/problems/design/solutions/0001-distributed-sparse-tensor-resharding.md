# 1. Distributed Sparse‑Tensor Sharding & Re‑sharding Engine

## Problem definition

**Problem Sketch**  
You are given a *very* large sparse matrix `M` (think billions of rows, millions of columns) that is stored **sharded** across a cluster of machines. Each shard is a `scipy.sparse.csr_matrix` saved as a separate file on a distributed file system (e.g., GCS, S3).  

The team needs a **generic “re‑sharding” service** that can:

1. **Re‑partition** the matrix from an *old* sharding scheme (e.g., row‑wise blocks of size `R_old`) to a *new* scheme (e.g., column‑wise blocks of size `C_new`).  
2. **Guarantee exactly‑once semantics** – no row/column is lost or duplicated.  
3. **Minimise network traffic** – only move the minimum amount of data required.  
4. **Support streaming** – the service must be able to start re‑sharding while other jobs are reading the *old* shards (read‑while‑write).  
5. **Expose a Python API** that can be called from a training pipeline:  

```python
def reshuffle(
    src_pattern: str,          # e.g. "gs://bucket/matrix/shard_{i}.npz"
    dst_pattern: str,          # e.g. "gs://bucket/matrix_v2/shard_{i}.npz"
    old_scheme: ShardingSpec,
    new_scheme: ShardingSpec,
    *,
    max_concurrency: int = 32,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> None: ...
```

`ShardingSpec` contains the block dimensions, the number of workers, and a deterministic hash function for row/column assignment.

## Solution

**Design Document – Distributed Sparse‑Tensor Sharding & Re‑sharding Engine**

---

### 1. Title & Metadata
- **Title:** Design of Sparse Tensor Re‑sharding Service  
- **Author:** (Your Name)  
- **Date:** 2025‑11‑07  
- **Version:** 1.0  
- **Audience:** Data engineers, ML platform engineers, reliability engineers  

---

### 2. Problem Statement
- **Business need:** Large sparse matrices are stored shard‑wise on object storage. Training pipelines require the matrix to be reorganised (e.g., row blocks → column blocks) without downtime and without data loss.  
- **Success criteria:**  
  - Exactly‑once semantics for every non‑zero element.  
  - Network traffic ≤ 20 % of total matrix size (move only elements that change shard).  
  - Ability to start re‑sharding while existing jobs read old shards.  
  - API callable from Python with configurable concurrency.  
  - End‑to‑end latency ≤ 2 hours for a 10 TB matrix on a 64‑node cluster.  

---

### 3. Scope & Assumptions
| In‑scope | Out‑of‑scope |
|----------|--------------|
| Re‑partitioning of existing shards from one deterministic scheme to another. | Generation of new matrix data, online updates during re‑sharding. |
| Streaming read‑while‑write support. | Real‑time query service on the re‑sharded matrix. |
| Exactly‑once guarantees using idempotent writes. | Migration of metadata stores not owned by this service. |

**Assumptions**  
- Object storage provides atomic put/delete for whole files.  
- Each shard fits in memory of a single worker (typical shard size ≤ 2 GB).  
- Row/column indices are 64‑bit integers.  
- Deterministic hash functions are pure and side‑effect free.  

---

### 4. High‑Level Architecture
```
Client (Python) 
      |
      v
Job Scheduler (e.g. Airflow / custom runner)
      |
      v
Controller Service
      |
      +--- Worker Pool (max_concurrency)
      |        |
      |        v
      |   Shard Processor
      |        |
      |        +--- Read old shard from src_pattern
      |        +--- Partition rows/cols according to new_scheme
      |        +--- Write partial shards to dst_pattern (temp files)
      |
      v
Metadata Store (e.g. Cloud Firestore)
      |
      v
Object Storage (GCS / S3)
```

- **Controller** decides which source shards to process, tracks progress, and records completion status in the metadata store.  
- **Workers** run independently, read a source shard, compute the mapping of each non‑zero entry to its destination shard, and write to temporary destination files.  
- **Commit phase** renames temporary files to final names once all contributing workers have finished a destination shard (ensures exactly‑once).  

---

### 5. Component Details

#### 5.1 Controller Service
- **Purpose:** Orchestrate the re‑sharding job, enforce concurrency limits, expose progress callback.
- **Responsibilities:**  
  - Enumerate source shards from `src_pattern`.  
  - Compute destination shard layout from `new_scheme`. - Issue work units to the worker pool.  
  - Update progress in metadata store.  
  - Perform two‑phase commit for each destination shard.
- **Interfaces:**  
  - `POST /start` – starts a job, returns job_id.  
  - `GET /status/{job_id}` – returns percent complete.  
- **Dependencies:** Metadata store, object storage client, worker executor (ThreadPool/ProcessPool).

#### 5.2 Worker (Shard Processor)
- **Purpose:** Transform a single source shard into one or more destination shard fragments.
- **Responsibilities:**  
  - Load source `.npz` (CSR) into memory.  
  - For each non‑zero `(row, col, value)` compute destination shard id using `new_scheme.hash(row, col)`.  
  - Append entry to an in‑memory buffer per destination shard.  
  - When buffer reaches threshold (e.g., 64 MB) flush to a temporary file `dst_pattern.tmp_{worker_id}_{shard_id}`.  
  - After processing all rows, signal completion to controller.
- **Interfaces:** Called via internal RPC or direct function call from controller.
- **Data storage:** Writes temporary files to the same bucket as final destination (different prefix).  

#### 5.3 Metadata Store
- **Purpose:** Track job state, per‑shard write progress, and commit status.
- **Schema (simplified):**  
  - `jobs/{job_id}`: status, start_time, end_time, total_shards, completed_shards.  
  - `shards/{dst_shard_id}`: list of writer ids that have produced a fragment, commit_flag.  

#### 5.4 Object Storage Client
- Provides streaming download/upload, supports multipart upload for large fragments.  

---

### 6. Data Model
- **Source shard file:** CSR stored as SciPy `.npz` (arrays `data`, `indices`, `indptr`, plus shape).  
- **Destination shard file:** Same format.  
- **Mapping function:** `new_shard_id = new_scheme.hash(row, col) % new_scheme.num_shards`.  
- **Indexes:** No secondary indexes needed; each worker writes directly to the correct shard file.  

**Write Path:**  
1. Worker reads source CSR.  
2. For each entry, compute `new_shard_id`.  
3. Append to in‑memory buffer for that shard.  
4. Flush buffer to temporary file.  

**Read Path (post‑commit):**  
- Consumers read final destination shards; temporary files are invisible.  

---

### 7. Scalability & Performance

| Metric | Estimate | Strategy |
|--------|----------|----------|
| Source shards | up to 10 000 | Parallel processing, max_concurrency controls load |
| Non‑zero entries per shard | up to 200 M | Buffer flushing to avoid OOM |
| Network traffic | only entries that move to a different shard (≈ 30 % for row→column reshuffle) | Workers write directly to destination bucket, no intermediate shuffle service |
| Horizontal scaling | Add more workers up to cluster limit | Stateless workers, no shared memory |

- **Caching:** Not required; each worker streams directly from object storage.  
- **Back‑pressure:** Controller limits number of in‑flight workers; workers pause if upload bandwidth saturates.  

---

### 8. Reliability & Fault Tolerance

- **Idempotent writes:** Temporary files include worker id; re‑running a failed worker overwrites its own temp files only.  
- **Two‑phase commit:** After all workers have written fragments for a destination shard, controller merges fragments (concatenates CSR parts) and atomically renames the merged file to final name.  
- **Retries:** Worker failures are retried up to 3 times with exponential back‑off.  
- **Redundancy:** Object storage provides multi‑AZ replication; controller state stored in highly available metadata store.  
- **Monitoring:** Emit metrics `shards_processed`, `bytes_written`, `retry_count`. Health checks on controller endpoint.  

---

### 9. Security

- **Authentication:** Service accounts with IAM roles granting read on `src_pattern` bucket and write on `dst_pattern` bucket.  
- **Authorization:** Controller validates that caller’s token matches allowed project.  
- **Encryption:** Data at rest encrypted by storage service; TLS used for all API calls.  
- **Input validation:** Verify that `src_pattern` and `dst_pattern` resolve to the same storage provider; reject malformed `ShardingSpec`.  

---

### 10. Trade‑offs & Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **In‑place rewrite (single pass, no temp files)** | Minimal storage overhead | Hard to guarantee exactly‑once if worker crashes; cannot support read‑while‑write |
| **Map‑Reduce style shuffle (intermediate key‑value store)** | Well‑known pattern, easy to scale | Requires additional cluster (e.g., Spark), higher network traffic, more complex ops |
| **Current design (temp files + two‑phase commit)** | Simple, leverages existing object storage, strong exactly‑once, low extra traffic | Requires extra storage for temporary fragments, merge step adds CPU overhead |

Chosen design balances simplicity, reliability, and network efficiency for the expected workload.

---

### 11. Deployment & Operations

- **CI/CD:** GitHub Actions builds Docker image, runs unit tests, pushes to Artifact Registry.  
- **Infrastructure as Code:** Terraform module creates Cloud Run service for controller, Pub/Sub topic for worker tasks, Firestore database, IAM bindings.  
- **Rolling upgrade:** Deploy new container version with traffic split; controller reads version from env, ensuring backward compatibility.  
- **Canary:** Start new job with a small subset of shards; monitor metrics before full rollout.  

---

### 12. Open Questions & Risks

| Question | Risk | Mitigation |
|----------|------|------------|
| What is the maximum size of a single destination shard? | Could exceed worker memory if too large. | Enforce shard size limit in `new_scheme`; fallback to sub‑sharding. |
| How to handle concurrent writes to the same destination shard from multiple jobs? | Data corruption. | Use job‑scoped destination prefixes; enforce exclusive lock in metadata store. |
| Impact of object‑storage eventual consistency on rename operation? | Consumers might see partially merged file. | Use generation‑matched rename (if supported) or write to a new prefix and switch consumer pointer only after commit. |

---

### 13. References

- SciPy CSR format documentation.  
- Google Cloud Storage consistency model.  
- “Designing Data‑Intensive Applications” – chapter on two‑phase commit.  
- Existing internal sharding spec definition (`ShardingSpec.proto`).  

--- 

**End of Document**