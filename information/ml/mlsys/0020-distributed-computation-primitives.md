# Essential Distributed Communication Primitives ( Focus: Deep Learning Training )

Distributed deep learning training relies on a small set of collective communication primitives that synchronize data across multiple GPUs or nodes.  Operations such as All‑Reduce, Broadcast, Reduce, All‑Gather, Gather, Scatter, Reduce‑Scatter, and point‑to‑point Send/Recv provide the core mechanisms for aggregating gradients, distributing model parameters, collecting metrics, and enabling model‑parallel pipelines.  These primitives are implemented in libraries like PyTorch Distributed, TensorFlow Distribute, NCCL, and MPI, and their efficient use is essential for scaling training workloads to large clusters.

In distributed deep learning training, several **communication primitives** are used to coordinate computation across multiple devices (e.g., GPUs or nodes). These primitives ensure that model parameters or gradients are synchronized efficiently. 

Here are the **basic distributed communication primitives** commonly used:


---

### 1. **All-Reduce**
- **Purpose**: Aggregate data (e.g., gradients) from all processes and distribute the result back to all.
- **Operation**:
  - Each process contributes a tensor (e.g., local gradients).
  - All processes receive the **sum (or average, max, etc.)** of all tensors.
- **Use Case**: Synchronizing gradients in data-parallel training.
- **Example**: After computing local gradients on each GPU, `all-reduce` sums them so every GPU has the global gradient.
- **Common Algorithms**: Ring AllReduce, Recursive Halving, or Tree-based.

```python
# PyTorch example
dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
```

---

### 2. **Broadcast (Bcast)**
- **Purpose**: Send data from one process (the root) to all others.
- **Operation**:
  - One process sends a tensor.
  - All other processes receive a copy.
- **Use Case**: Initializing model weights, distributing hyperparameters, or syncing model state at startup.

```python
dist.broadcast(model_state, src=0)
```

---

### 3. **Reduce**
- **Purpose**: Aggregate data from all processes to a single destination.
- **Operation**:
  - All processes send their data.
  - Only the root process gets the reduced result (e.g., sum).
- **Use Case**: Logging metrics (e.g., average loss across all GPUs) on a single node.

```python
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
```

---

### 4. **All-Gather**
- **Purpose**: Gather tensors from all processes and concatenate them.
- **Operation**:
  - Each process contributes a tensor.
  - All processes receive a list (or concatenated tensor) of all inputs.
- **Use Case**:
  - Gathering outputs for evaluation.
  - Synchronizing batch norms across GPUs (`SyncBatchNorm`).
  - Sharded training where partial results need to be collected.

```python
dist.all_gather(tensor_list, local_tensor)
```

---

### 5. **Gather**
- **Purpose**: Collect tensors from all processes to one root process.
- **Operation**:
  - Only the root process receives all tensors.
  - Other processes send their data but don't receive anything.
- **Use Case**: Centralized logging or debugging.

```python
dist.gather(tensor, gather_list, dst=0)
```

---

### 6. **Scatter**
- **Purpose**: Distribute a list of tensors from one root process to all others.
- **Operation**:
  - Root process splits a list and sends one piece to each process.
  - Each process receives one tensor.
- **Use Case**: Distributing data or parameters in model-parallel settings.

```python
dist.scatter(recv_tensor, scatter_list, src=0)
```

---

### 7. **Reduce-Scatter**
- **Purpose**: Combine `reduce` and `scatter`: reduce across processes and then scatter parts of the result.
- **Operation**:
  - Each process contributes a tensor.
  - The global reduced result is split, and each process gets one chunk.
- **Use Case**:
  - Optimized `all-reduce` implementations.
  - Sharded optimizer states (e.g., ZeRO in DeepSpeed).

```python
# Each process gets one part of the reduced sum
dist.reduce_scatter(output_tensor, input_tensor_list, op=dist.ReduceOp.SUM)
```

---

### 8. **Point-to-Point Communication**
- **Send / Receive**:
  - One process sends a tensor to another.
- **Use Case**: Custom pipelines, asynchronous updates, or model parallelism.

```python
dist.send(tensor, dst=1)
dist.recv(tensor, src=0)
```

---

### Summary Table

| Primitive        | Sends From | Sends To     | Aggregation? | Common Use Case |
|------------------|-----------|--------------|--------------|------------------|
| **All-Reduce**   | All       | All          | Yes          | Gradient sync    |
| **Broadcast**    | One (root)| All          | No           | Init weights     |
| **Reduce**       | All       | One (root)   | Yes          | Logging metrics  |
| **All-Gather**   | All       | All          | No (concat)  | Batch norm sync  |
| **Gather**       | All       | One (root)   | No (concat)  | Central logging  |
| **Scatter**      | One       | All          | No           | Data distribution|
| **Reduce-Scatter**| All     | All (chunks) | Yes          | Sharded optimizers |
| **Send/Recv**    | One       | One          | No           | Custom comms     |

---

### Key Libraries & Frameworks
- **PyTorch**: `torch.distributed`
- **TensorFlow**: `tf.distribute`
- **NCCL** (NVIDIA): Optimized backend for GPU collectives.
- **MPI** (Message Passing Interface): Traditional HPC standard.

---

These primitives form the foundation of scalable distributed training and are optimized heavily in frameworks like PyTorch Distributed, Horovod, and DeepSpeed.

