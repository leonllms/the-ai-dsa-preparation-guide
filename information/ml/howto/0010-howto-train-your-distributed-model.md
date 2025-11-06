# LLM Distributed Training

Below is a **complete, production‑ready recipe** for training a very large model on **4 separate servers (nodes)**, each equipped with **8 GPUs** (so 32 GPUs in total).  
The approach uses **PyTorch Distributed Data‑Parallel (DDP)** as the core primitive, and I also show how you can layer on **FSDP / DeepSpeed** (ZeRO) or **pipeline parallelism** if the model is too big to fit on a single GPU.

---

## 1️⃣ High‑level Architecture

| Layer | What it does | Typical tool |
|------|---------------|--------------|
| **Node‑level communication** | Sends gradients / parameters across machines | `torch.distributed` with **NCCL** backend |
| **GPU‑level parallelism** | Replicates the model on every GPU, each GPU processes its own mini‑batch | **DDP** (or **FSDP** for sharding) |
| **Model‑parallelism (optional)** | Splits a single model across GPUs when it cannot fit on one GPU | **torch.distributed.pipeline** or **DeepSpeed ZeRO‑3** |
| **Elastic / fault‑tolerant launch** | Handles pre‑emptions, node failures, dynamic scaling | `torchrun` / `torch.distributed.elastic` |

The simplest “out‑of‑the‑box” solution is **DDP + torchrun**. All the extra pieces (FSDP, ZeRO, pipeline) are just plug‑ins on top of the same launch command.

---

## 2️⃣ Prerequisites (Software & Hardware)

| Item | Minimal version / notes |
|------|------------------------|
| **PyTorch** | `torch >= 2.2` (has built‑in `torchrun` and FSDP improvements) |
| **CUDA** | Same major version as your GPUs (e.g., CUDA 12.4) |
| **NCCL** | Comes bundled with the PyTorch wheels, but ensure the driver is recent. |
| **Python** | 3.9+ (3.11 recommended) |
| **Network** | 10 GbE works, but for best NCCL performance use **InfiniBand** or **RoCE**. |
| **OS** | Linux (Ubuntu 22.04, CentOS 8, etc.) |
| **SSH** | Passwordless SSH from every node to every other node (for `torchrun` to spawn remote processes). |
| **Environment** | Same Python environment (conda/venv) on every node (same PyTorch, same CUDA). |
| **Storage** | Shared filesystem (NFS, Lustre, GPFS) for dataset and checkpoints, *or* each node can have its own copy. |

---

## 3️⃣ Network & SSH Setup

1. **Pick a master node** (e.g., `node0`). All other nodes will connect to it.
2. On each node edit `~/.ssh/config` (or use key‑based auth) so that you can `ssh nodeX` without a password from any other node.
3. Verify connectivity:

```bash
# From each node
ssh node0 hostname   # should print node0's hostname
ssh node1 hostname   # should print node1's hostname
...
```

4. Reserve a free TCP port on the master (e.g., `29500`). Make sure firewalls allow inbound traffic on this port from the other nodes.

---

## 4️⃣ Directory Layout (recommended)

```
/shared/project/
│
├─ train.py                # entry point (DDP / FSDP / DeepSpeed)
├─ model.py                # model definition
├─ dataset.py              # dataset / dataloader
├─ utils/
│   └─ dist_utils.py       # helper functions for init, barrier, etc.
└─ scripts/
    └─ launch.sh           # wrapper that calls torchrun
```

All nodes mount `/shared/project` at the same path (e.g., via NFS). If you cannot share a FS, you can copy the repo to each node and use the same absolute paths.

---

## 5️⃣ Minimal DDP Training Script

Below is a **self‑contained** `train.py`. It works for any model you plug in; the only required arguments are the usual training hyper‑parameters plus the distributed settings.

```python
# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import MyHugeModel          # <-- your model definition
from dataset import MyDataset          # <-- your dataset definition
from utils.dist_utils import init_distributed, cleanup

def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------- Distributed args -----------------
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes (machines)")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="Rank of this node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                        help="IP address of the master node")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Port on the master node")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="GPUs per node")
    # ----------------- Training args -----------------
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./ckpts")
    # ----------------------------------------------------
    return parser.parse_args()

def main():
    args = parse_args()

    # ----------------------------------------------------
    # 1️⃣ Initialize torch.distributed
    # ----------------------------------------------------
    dist.init_process_group(
        backend="nccl",
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.nnodes * args.nproc_per_node,
        rank=args.node_rank * args.nproc_per_node + int(os.environ["LOCAL_RANK"])
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    # ----------------------------------------------------
    # 2️⃣ Seed for reproducibility (only on rank 0 is enough)
    # ----------------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ----------------------------------------------------
    # 3️⃣ Create model, optimizer, loss
    # ----------------------------------------------------
    model = MyHugeModel()
    model = model.to(device)

    # If the model does NOT fit on a single GPU you can switch
    # to FullyShardedDataParallel (FSDP) or DeepSpeed later.
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # ----------------------------------------------------
    # 4️⃣ Dataset & DistributedSampler
    # ----------------------------------------------------
    dataset = MyDataset(split="train")
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank(), shuffle=True, seed=args.seed)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True)

    # ----------------------------------------------------
    # 5️⃣ Training loop
    # ----------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)      # important for shuffling
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0 and dist.get_rank() == 0:
                print(f"[Epoch {epoch}/{args.epochs}] "
                      f"Iter {batch_idx}/{len(loader)} "
                      f"Loss: {loss.item():.4f}")

        # ------------------------------------------------
        # 6️⃣ Validation (optional, same pattern as train)
        # ------------------------------------------------
        # (skip for brevity)

        # ------------------------------------------------
        # 7️⃣ Checkpointing – only rank 0 writes to disk
        # ------------------------------------------------
        if dist.get_rank() == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir,
                                     f"epoch_{epoch}_rank0.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss / len(loader),
                "args": vars(args)
            }, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

    # ----------------------------------------------------
    # 8️⃣ Clean up
    # ----------------------------------------------------
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### What the script does

| Step | Why it matters |
|------|----------------|
| **`init_process_group`** | Creates a NCCL communicator across *all* processes (total `world_size = nnodes * nproc_per_node`). |
| **`torch.cuda.set_device`** | Guarantees each process talks to its own GPU (`LOCAL_RANK` is set automatically by `torchrun`). |
| **`DistributedSampler`** | Guarantees each GPU sees a *different* slice of the dataset, avoiding duplication. |
| **`DDP(model)`** | Wraps the model so that after each `loss.backward()` gradients are automatically reduced (averaged) across the whole world. |
| **Checkpoint only on rank 0** | Prevents NFS‑contention and duplicate files. |
| **`sampler.set_epoch(epoch)`** | Makes the RNG seed for shuffling deterministic across epochs. |

---

## 6️⃣ Launching on 4 Nodes × 8 GPUs

### 6.1 One‑liner with `torchrun`

On **each node** run exactly the same command (you can put it in a `launch.sh` script and `ssh` it from a scheduler, e.g., SLURM, PBS, or a simple `bash` loop).

```bash
#!/usr/bin/env bash
# scripts/launch.sh
# Arguments:
#   $1 = node_rank (0,1,2,3)
#   $2 = master_addr (IP of node0)

NODE_RANK=$1
MASTER_ADDR=$2
MASTER_PORT=29500          # any free port, same on all nodes
NNODES=4
GPUS_PER_NODE=8

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --epochs=20 \
    --batch_size=64 \
    --lr=5e-5 \
    --output_dir=/shared/project/ckpts
```

**How to start:**

```bash
# On node0 (master)
bash scripts/launch.sh 0 $(hostname -I | awk '{print $1}')

# On node1
bash scripts/launch.sh 1 <MASTER_IP>

# On node2
bash scripts/launch.sh 2 <MASTER_IP>

# On node3
bash scripts/launch.sh 3 <MASTER_IP>
```

If you have a **job scheduler (SLURM)** you can replace the manual SSH with a single `srun` command that automatically sets `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, etc. Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=huge_model
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=compute

module load cuda/12.4
module load anaconda3
source activate my_pytorch_env

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  train.py \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --epochs=25 \
    --batch_size=128 \
    --lr=2e-5 \
    --output_dir=/shared/project/ckpts
```

### 6.2 Using `torch.distributed.elastic` (elastic fault tolerance)

If you want the training job to survive a node crash or pre‑emptible instance termination, replace `torchrun` with `torch.distributed.run` (the same binary) **plus** the `--standalone` flag and a simple `elastic` launch script:

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:29500 \
  --max_restarts=3 \
  --monitor_interval=5 \
  train.py ...
```

`elastic` will automatically restart the whole job if any rank fails, up to `max_restarts` times. This is handy on cloud spot‑instances.

---

## 7️⃣ Scaling Beyond Simple DDP

If **your model cannot fit in the memory of a single GPU**, you need **model‑parallelism** or **parameter sharding**. Below are three proven ways that integrate nicely with the same launch script.

### 7.1 Fully‑Sharded Data Parallel (FSDP)

*Introduced in PyTorch 1.12, dramatically reduces memory by sharding parameters, gradients, and optimizer states.*

```python
# In train.py – replace the DDP block
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Wrap model
model = MyHugeModel().to(device)

# You can enable mixed precision inside FSDP:
fsdp_wrap_policy = lambda m: isinstance(m, torch.nn.Module)  # or a custom policy
model = FSDP(
    model,
    auto_wrap_policy=fsdp_wrap_policy,
    cpu_offload=False,                # set True if you want offload to CPU
    mixed_precision=True,            # FP16 training (requires torch.cuda.amp)
    device_id=int(os.environ["LOCAL_RANK"]),
)

# Later, when saving a checkpoint:
if dist.get_rank() == 0:
    # Convert sharded state dict to a regular one
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        full_state = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": full_state,
        "optimizer_state_dict": optimizer.state_dict(),
        ...
    }, ckpt_path)
```

**Pros:**  
- Works with *any* model, no code changes aside from the wrapper.  
- Gives up to 4‑10× memory reduction.

**Cons:**  
- Slightly higher communication overhead (still NCCL‑based).  
- Checkpointing needs the `state_dict_type` context.

### 7.2 DeepSpeed ZeRO (Stage‑2/3)

DeepSpeed adds sophisticated optimizer‑state sharding (ZeRO‑1/2) and parameter sharding (ZeRO‑3). If you already use DeepSpeed for other projects, it’s the most memory‑efficient.

1. **Install**: `pip install deepspeed==0.14.0` (or newer).  
2. **Create a `deepspeed_config.json`**:

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu"
    },
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

3. **Modify `train.py`**:

```python
import deepspeed

# after parsing args
deepspeed_config = "ds_config.json"
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config
)

# Replace `model` → `model_engine`
outputs = model_engine(inputs)
loss = criterion(outputs, targets)
model_engine.backward(loss)
model_engine.step()
```

**Launch** – exactly the same `torchrun` command. DeepSpeed will automatically handle the world‑size and process‑group initialization.

### 7.3 Pipeline Parallelism (torch.distributed.pipeline)

If your model has a natural depth (e.g., a transformer with many encoder layers), you can split it into **stages** and run each stage on a different GPU. This reduces per‑GPU memory even further, but introduces pipeline bubbles.

```python
# In model.py
from torch import nn
from torch.distributed.pipeline.sync import Pipe

class MyHugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: 12 transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(12)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

# In train.py – after Distributed init
model = MyHugeModel()
# Split into N stages (e.g., 4 stages → 4 GPUs per node)
chunks = 8   # micro‑batching factor; typical 8–16
model = Pipe(model,
             balance=[3,3,3,3],   # number of layers per stage
             devices=[f'cuda:{i}' for i in range(args.nproc_per_node)],
             chunks=chunks,
             checkpoint='always')   # activation checkpointing
```

**Pros:**  
- Very low per‑GPU memory (only a subset of layers lives on each GPU).  
- Works well when you have *many* layers.

**Cons:**  
- More complex to debug.  
- Requires careful batch‑size tuning (`chunks * world_size` must be divisible by your global batch size).

---

## 8️⃣ Mixed‑Precision (AMP) – Boost Throughput

All three scaling methods (DDP, FSDP, DeepSpeed) support **automatic mixed‑precision** (FP16/BF16). The cheapest way is to wrap the forward‑backward pass with `torch.cuda.amp.autocast`:

```python
scaler = torch.cuda.amp.GradScaler()   # for DDP/FSDP
for inputs, targets in loader:
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

If you use **DeepSpeed** you can enable AMP via the JSON (`"fp16": {"enabled": true}`) – it will automatically insert the autocast and scaler.

---

## 9️⃣ Checkpointing & Resuming (Robust Production)

### 9.1 What to Save

| Item | Where to store |
|------|----------------|
| `epoch` | integer |
| `model_state_dict` | full state (DDP → `model.module.state_dict()`, FSDP → `full_state` via context, DeepSpeed → handled by `model_engine.save_checkpoint`) |
| `optimizer_state_dict` | same as model |
| `scaler_state_dict` (if AMP) | `scaler.state_dict()` |
| `rng_state` | `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` |
| `args` | original hyper‑params (helpful for reproducibility) |

### 9.2 Saving (rank 0)

```python
if dist.get_rank() == 0:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state,   # depends on wrapper
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "args": vars(args)
    }
    torch.save(ckpt, f"{args.output_dir}/ckpt_epoch_{epoch}.pt")
```

### 9.3 Loading (any rank)

```python
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
if scaler and ckpt["scaler_state_dict"]:
    scaler.load_state_dict(ckpt["scaler_state_dict"])
torch.set_rng_state(ckpt["torch_rng_state"])
torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
start_epoch = ckpt["epoch"] + 1
```

**DeepSpeed**: `model_engine.load_checkpoint(ckpt_dir)` automatically restores everything (including scaler).

---

## 10️⃣ Performance‑Tuning Tips

| Area | Recommended Settings |
|------|----------------------|
| **NCCL** | `export NCCL_DEBUG=INFO` (first run), `export NCCL_IB_DISABLE=0` (if using InfiniBand), `export NCCL_SOCKET_IFNAME=^lo,docker0` |
| **CUDA kernels** | Use `torch.backends.cudnn.benchmark = True` if input size is constant. |
| **Batch size** | Scale linearly with GPUs: `global_batch = per_gpu_batch * world_size`. Keep `per_gpu_batch` as large as memory permits; otherwise use gradient accumulation. |
| **Gradient accumulation** | If the desired global batch does not fit, set `--gradient_accumulation_steps N`. |
| **Profiler** | `torch.profiler.profile` with `schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)` to spot bottlenecks. |
| **CPU‑pinning** | Set `export OMP_NUM_THREADS=4` (or `num_workers` in DataLoader) to avoid oversubscription. |
| **Data loading** | Use `prefetch_factor=2` and `persistent_workers=True`. When using many nodes, consider **torchdata** or **WebDataset** for sharded data. |
| **Mixed‑precision** | BF16 is often faster on Ampere+ GPUs (`torch.backends.cuda.matmul.allow_tf32 = True`). |
| **Checkpoint frequency** | Every 1–2 epochs or based on wall‑clock (e.g., every 4 h) to avoid long recovery times. |
| **Fault tolerance** | Use `torchrun --max_restarts=5` or DeepSpeed’s `deepspeed --deepspeed_config ds_config.json` which already supports checkpoint‑restart. |

---

## 11️⃣ Full End‑to‑End Example (Putting It All Together)

Below is a **single script** you can copy to `train_full.py`. It automatically selects the best parallelism based on a command‑line flag:

```python
# train_full.py
import argparse, os, torch, torch.nn as nn, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import deepspeed   # optional, safe to import even if not used
from model import MyHugeModel
from dataset import MyDataset

def parse_args():
    p = argparse.ArgumentParser()
    # Distributed
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--node_rank", type=int, default=0)
    p.add_argument("--master_addr", type=str, default="127.0.0.1")
    p.add_argument("--master_port", type=str, default="29500")
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./ckpts")
    # Parallelism mode
    p.add_argument("--mode", choices=["ddp","fsdp","deepspeed"], default="ddp")
    # DeepSpeed config path (if mode=deepspeed)
    p.add_argument("--ds_config", type=str, default="ds_config.json")
    # Mixed precision
    p.add_argument("--amp", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------
    # 1️⃣ Distributed init (same for all modes)
    # -------------------------------------------------
    dist.init_process_group(
        backend="nccl",
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.nnodes * args.nproc_per_node,
        rank=args.node_rank * args.nproc_per_node + int(os.environ["LOCAL_RANK"])
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    # -------------------------------------------------
    # 2️⃣ Seed
    # -------------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------
    # 3️⃣ Model & optimizer
    # -------------------------------------------------
    model = MyHugeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------------------------------------
    # 4️⃣ Choose parallelism wrapper
    # -------------------------------------------------
    if args.mode == "ddp":
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        scaler = torch.cuda.amp.GradScaler() if args.amp else None

    elif args.mode == "fsdp":
        # Simple auto‑wrap policy – you can make it more selective.
        fsdp_wrap_policy = lambda m: isinstance(m, torch.nn.Module)
        model = FSDP(
            model,
            auto_wrap_policy=fsdp_wrap_policy,
            mixed_precision=args.amp,
            device_id=int(os.environ["LOCAL_RANK"]),
        )
        scaler = torch.cuda.amp.GradScaler() if args.amp else None

    elif args.mode == "deepspeed":
        # DeepSpeed will replace optimizer & scaler for us.
        ds_config = args.ds_config
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        model = model_engine
        scaler = None  # DeepSpeed handles it internally
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # -------------------------------------------------
    # 5️⃣ Data
    # -------------------------------------------------
    dataset = MyDataset(split="train")
    sampler = DistributedSampler(dataset,
                                 num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank(),
                                 shuffle=True,
                                 seed=args.seed)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True,
                        prefetch_factor=2,
                        persistent_workers=True)

    criterion = nn.CrossEntropyLoss().to(device)

    # -------------------------------------------------
    # 6️⃣ Training loop
    # -------------------------------------------------
    start_epoch = 0
    # If you want to resume:
    # ckpt_path = os.path.join(args.output_dir, "latest.pt")
    # if os.path.isfile(ckpt_path):
    #     ckpt = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #     if scaler and "scaler_state_dict" in ckpt:
    #         scaler.load_state_dict(ckpt["scaler_state_dict"])
    #     start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            if args.amp:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0 and dist.get_rank() == 0:
                print(f"[Epoch {epoch}/{args.epochs}] "
                      f"Iter {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.4f}")

        # -------------------------------------------------
        # 7️⃣ Checkpoint (rank 0 only)
        # -------------------------------------------------
        if dist.get_rank() == 0:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            if args.mode == "fsdp":
                # Convert sharded state dict to a full one
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    full_state = model.state_dict()
                model_state = full_state
            elif args.mode == "deepspeed":
                # DeepSpeed writes its own checkpoint files; we just save optimizer state for safety
                model_engine.save_checkpoint(args.output_dir, tag=f"epoch_{epoch}")
                model_state = None  # already saved by DeepSpeed
            else:   # ddp
                model_state = model.module.state_dict()

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "args": vars(args)
            }
            torch.save(ckpt, ckpt_path)
            print(f"✅ Saved checkpoint {ckpt_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**How to run it**

```bash
# 4 nodes, 8 GPUs each → 32 total
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=$SLURM_NODEID \   # or set manually if you use ssh
  --master_addr=<MASTER_IP> \
  --master_port=29500 \
  train_full.py \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    --epochs=30 \
    --batch_size=64 \
    --lr=2e-5 \
    --mode=fsdp \                 # choose ddp / fsdp / deepspeed
    --amp                         # optional mixed‑precision
```

> **Tip:** Keep the `--mode` flag the same across all nodes. The script will automatically pick the right wrapper.

---

## 12️⃣ Common Pitfalls & How to Fix Them

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RuntimeError: NCCL error 2: unhandled system error` | Mismatched NCCL versions, firewall, or missing InfiniBand drivers. | Verify `nccl-tests` works between nodes (`nccl-tests/build/all_reduce_perf -b 8 -e 64M -f 2 -g 8`). Open the port, ensure same CUDA driver version. |
| `torch.cuda.OutOfMemoryError` even with small batch | Model still too big for one GPU. | Switch to **FSDP** or **DeepSpeed ZeRO‑3**. Enable activation checkpointing (`torch.utils.checkpoint`). |
| `RuntimeError: Expected all tensors to be on the same device` | Accidentally moving tensors to CPU on some ranks. | Ensure every tensor you create (`inputs`, `targets`, model parameters) is `.to(device)` **after** `torch.cuda.set_device`. |
| `FileNotFoundError` on checkpoint load (different rank) | Only rank 0 saved checkpoint, other ranks tried to read it. | All ranks should `torch.load` the same file (it’s a normal `.pt`). Use `torch.distributed.barrier()` after loading to sync. |
| Slow scaling, < 80% of ideal speed‑up | Network bottleneck (e.g., Ethernet instead of IB), small batch size, or excessive gradient sync overhead. | Increase `batch_size` / `gradient_accumulation_steps`, enable `torch.backends.cudnn.benchmark=True`, tune NCCL env vars (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`). |
| “Process exited with code 255” on node launch | SSH key missing or `torchrun` cannot start remote processes. | Test `ssh nodeX hostname` manually; add `-vvv` to `torchrun` for deeper logs. |

---

## 13️⃣ TL;DR – One‑Line Checklist

1. **Same conda/venv** on all 4 nodes.  
2. **Password‑less SSH** between nodes.  
3. **Set `MASTER_ADDR`** to the IP of node 0; open port 29500.  
4. **Install PyTorch ≥2.2** (with NCCL).  
5. **Pick a parallelism mode** (`ddp`, `fsdp`, or `deepspeed`).  
6. **Write/modify training script** (see `train_full.py`).  
7. **Launch with `torchrun`** (or SLURM `srun`).  
8. **Monitor**: `nvidia-smi`, `torch.distributed.elastic`, NCCL logs.  
9. **Checkpoint** regularly, only rank 0 writes.  
10. **Resume** by loading checkpoint on all ranks.

---

### 🎉 You’re ready!

With the steps, scripts, and tuning tips above you can now train a **single gigantic model** across **32 GPUs spread over 4 machines**. Feel free to swap in your own model, dataset, or optimizer, and experiment with the three parallelism strategies to find the best memory‑throughput trade‑off for your workload.

If you hit a specific error or want deeper guidance on, say, **activation checkpointing**, **tensor‑parallelism**, or **hyper‑parameter scaling**, just let me know! Happy training! 🚀