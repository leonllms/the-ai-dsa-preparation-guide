# 🎓 Llama‑3 70B – From Architecture to GPU Kernels
*A step‑by‑step tutorial that ties together model dimensions, attention mechanics, MLP design, and the low‑level kernels that make inference fast.*

---

## 1. Model‑level Overview

| Component | Value (Llama‑3 70B) | Meaning |
|-----------|--------------------|---------|
| **Model dimension** `d_model` | 8192 | Width of every token representation |
| **Number of layers** | 80 | Stacked transformer blocks |
| **Query heads** `h_q` | 64 | Heads that produce queries |
| **KV heads** `h_kv` | 8 | Heads that produce keys & values (Grouped‑Query Attention) |
| **Head size** `d_head` | 128 | `d_model / h_q = 8192 / 64` |
| **KV hidden size** `d_kv` | 8 × 128 = **1024** |
| **MLP hidden size** `d_ff` | 28672 (≈ 3.5 × d_model) |
| **Norm** | RMSNorm (8192‑dim) | Applied before attention and before MLP |

> **Grouped‑Query Attention (GQA)** – 64 query heads share only 8 KV heads, reducing the KV cache size by a factor of 8 while keeping the same expressive power.

---

## 2. Linear Projections (the “big” weight matrices)

### 2.1 Shapes

| Projection | Input dim | Output dim | Weight shape | # parameters |
|------------|-----------|------------|--------------|--------------|
| `W_Q` (query) | 8192 | 8192 (=64 × 128) | (8192, 8192) | 67 108 864 |
| `W_K` (key)   | 8192 | 1024 (=8 × 128) | (8192, 1024) | 8 388 608 |
| `W_V` (value) | 8192 | 1024 (=8 × 128) | (8192, 1024) | 8 388 608 |
| `W_up_g` (gate) | 8192 | 28672 | (8192, 28672) | 235 929 600 |
| `W_up` (proj)   | 8192 | 28672 | (8192, 28672) | 235 929 600 |
| `W_down` (proj) | 28672 | 8192 | (28672, 8192) | 235 929 600 |

All matrices are **dense** (full‑size). They are *not* diagonal or “big vectors”.

### 2.2 Per‑head view

- One **query head**: `8192 × 128` weights.  
- One **KV head**: `8192 × 128` weights → 8 such heads give the 1024‑dim KV projection.  

---

## 3. Forward Pass of a Single Transformer Block

Below is the mathematical pipeline (layer‑norm omitted for brevity).

1. **Q, K, V projection**  

\[
\begin{aligned}
Q &= X\,W_Q \quad\in\mathbb{R}^{L\times d_{\text{model}}} \\
K &= X\,W_K \quad\in\mathbb{R}^{L\times d_{kv}} \\
V &= X\,W_V \quad\in\mathbb{R}^{L\times d_{kv}}
\end{aligned}
\]

2. **Reshape to heads**  

\[
\begin{aligned}
Q &\rightarrow \text{reshape}(L, h_q, d_{\text{head}}) \\
K &\rightarrow \text{reshape}(L, h_{kv}, d_{\text{head}}) \\
V &\rightarrow \text{reshape}(L, h_{kv}, d_{\text{head}})
\end{aligned}
\]

3. **Scaled dot‑product attention**  

\[
S_{i,j}^{(h)} = \frac{Q_i^{(h)}\,(K_j^{(h)})^\top}{\sqrt{d_{\text{head}}}} + \text{mask}_{i,j}
\]

4. **Row‑wise soft‑max**  

\[
A_{i,j}^{(h)} = \frac{\exp\!\bigl(S_{i,j}^{(h)}\bigr)}{\sum_{k=1}^{L}\exp\!\bigl(S_{i,k}^{(h)}\bigr)}
\]

5. **Attention output**  

\[
O_i^{(h)} = \sum_{j=1}^{L} A_{i,j}^{(h)} V_j^{(h)}
\]

6. **Concat heads → linear**  

\[
O_i = \text{concat}_{h=1}^{h_q}\! O_i^{(h)} \; W_O \quad (W_O\text{ is usually a fused projection)}
\]

7. **MLP (SwiGLU)**  

\[
\begin{aligned}
g &= \sigma\!\bigl(XW_{\text{up\_g}}\bigr) \quad (\sigma = \text{SiLU})\\
u &= XW_{\text{up}}\\
\text{SwiGLU}(X) &= (g \odot u)W_{\text{down}}
\end{aligned}
\]

8. **Residual connections**  

\[
X_{\text{next}} = X + O + \text{SwiGLU}(X)
\]

---

## 4. GPU‑level View – Why GEMM Matters

All linear layers above are **dense matrix‑multiply (GEMM)** operations:

\[
Y = X\,W \quad\Longleftrightarrow\quad \text{GEMM}(X,W)
\]

- **GEMM** = General Matrix Multiply, the workhorse of every transformer.
- On GPUs a GEMM is executed by a **kernel** that tiles the matrices into shared‑memory blocks, uses warp‑level registers, and performs many fused multiply‑add (FMA) instructions.

### 4.1 FLOP count (per token, per layer)

| Operation | FLOPs (multiply‑add) |
|-----------|----------------------|
| Q·Kᵀ (dense) | \(2\,L\,d_{\text{model}}\,d_{\text{head}}\) ≈ 2 × L × 8192 × 128 |
| V‑multiply (after softmax) | \(2\,L\,d_{\text{head}}\,d_{\text{model}}\) (same order) |
| MLP up‑proj + gate | \(2\,d_{\text{model}}\,d_{\text{ff}}\) ≈ 2 × 8192 × 28672 |
| MLP down‑proj | \(2\,d_{\text{ff}}\,d_{\text{model}}\) (same) |

For a typical context length `L = 2048` the attention GEMMs dominate memory traffic, while the MLP dominates FLOPs.

---

## 5. Causal Mask – Where does it live?

The mask is **added** to the raw scores **after** the Q·Kᵀ GEMM, before soft‑max:

```python
# pseudo‑code
scores = torch.matmul(Q, K.transpose(-2, -1))   # (L, h_q, L)
scores = scores / math.sqrt(d_head)
scores = scores + causal_mask   # -inf for j > i
```

- The mask is **not** a diagonal weight matrix.
- It forces the soft‑max to become lower‑triangular (future tokens get zero weight).

---

## 6. Soft‑max – Row‑wise Reduction

The soft‑max is applied **independently per row** (per query token, per head).  

Mathematically:

\[
\text{softmax}_i(\mathbf{s}) = \frac{\exp(\mathbf{s}_i)}{\sum_j \exp(\mathbf{s}_j)}
\]

Implementation steps (numerically stable):

```python
def rowwise_softmax(scores):
    # scores: (L, h, L)
    max_val, _ = scores.max(dim=-1, keepdim=True)          # (L, h, 1)
    scores = scores - max_val                               # subtract max for stability
    exp_scores = torch.exp(scores)                          # could be table‑lookup approx
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)          # reduction
    return exp_scores / sum_exp
```

**Key point:** Even if `exp(x)` is approximated by a tiny lookup table, the **sum‑reduction** over the whole row is still required and dominates memory traffic.

---

## 7. Kernel Fusion – From Three Kernels to One

### 7.1 Naïve sequence (3 kernels)

1. `GEMM_QK = Q·Kᵀ` → write scores to global memory.  
2. `softmax_kernel(scores)` → read, mask, soft‑max, write attention weights.  
3. `GEMM_AV = A·V` → read weights, multiply with V, write output.

Each step incurs a full global‑memory write/read of an `L×L` matrix → **huge bandwidth waste**.

### 7.2 Fused approach (FlashAttention style)

```
for each block of rows (tile):
    load Q_tile, K_tile, V_tile into shared memory
    compute partial scores = Q_tile @ K_tile.T
    add causal mask (‑inf) inside the tile
    apply scaling (1/√d_head)
    compute max, subtract, exponentiate (tiny table possible)
    accumulate row‑wise sum of exp (in registers)
    after the whole K dimension is processed:
        normalize: exp / sum
        multiply normalized weights with V_tile
        write final O_tile to global memory
```

- **Only one write** of the final attention output.  
- Intermediate `scores` never touch global memory.  
- The reduction (sum of exponentials) stays in registers/shared memory → **orders‑of‑magnitude speedup**.

### 7.3 Public implementations

| Library | Main feature |
|---------|--------------|
| **FlashAttention** (CUDA) | Tile‑wise GEMM + soft‑max + V‑multiply in a single kernel. |
| **xFormers** | Modular fused attention kernels, supports GQA. |
| **torch.nn.functional.scaled_dot_product_attention** (PyTorch 2.0+) | Calls cuDNN/FlashAttention under the hood. |
| **Triton** examples | Write custom kernels with explicit tiling. |
| **flash‑attention‑hip** | AMD‑GPU counterpart. |

---

## 8. KV Cache – Why GQA Saves Memory

During autoregressive generation we keep **keys** and **values** for every past token:

\[
\text{Cache}_K \in \mathbb{R}^{\text{seq\_len} \times d_{kv}},\qquad
\text{Cache}_V \in \mathbb{R}^{\text{seq\_len} \times d_{kv}}
\]

- With `d_kv = 1024` the cache per layer is `2 × seq_len × 1024 × 4 bytes` (FP32) ≈ **8 MiB** for `seq_len = 2048`.  
- If we used full `d_model = 8192` the cache would be **8× larger** (≈ 64 MiB), quickly exhausting GPU memory.

GQA (8 KV heads) therefore **reduces the KV cache by a factor of 8** while preserving the same number of query heads (64) for expressive attention.

---

## 9. Approximate Soft‑max via Lookup Tables – Why It’s Not a Full Replacement

| Idea | What can be table‑looked‑up? | What still needs work? |
|------|-----------------------------|------------------------|
| **Quantised `exp(x)`** | Map a bounded `x ∈ [-5,5]` to `exp(x)` using a 256‑entry table (linear interpolation). | Row‑wise sum of those values and division by the sum. |
| **Clipping + pre‑computed soft‑max** | If you knew the *entire* row distribution you could store the final soft‑max, but rows are data‑dependent → impossible. | The denominator depends on *all* elements in the row, which changes every token. |
| **Top‑k sparsification** | Keep only the largest `k` scores, compute soft‑max on that subset. | Still need a reduction over `k` values and a fallback for the rest. |

**Conclusion:** A lookup table can accelerate the *exp* operation, but the **reduction** (sum) and **normalisation** remain mandatory. In practice the extra work is cheaper than the memory traffic saved by a fused kernel, which is why libraries keep the full soft‑max (often with a tiny exp‑approx table) inside the same kernel.

---

## 10. End‑to‑End PyTorch Example (FlashAttention‑style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Model hyper‑parameters (Llama‑3 70B)
# -------------------------------------------------
d_model = 8192
n_layers = 80
n_q_head = 64
n_kv_head = 8
d_head = d_model // n_q_head          # 128
d_kv = n_kv_head * d_head             # 1024
d_ff = 28672

# -------------------------------------------------
# 2. Simple block (no RMSNorm for brevity)
# -------------------------------------------------
class Llama3Block(nn.Module):
    def __init__(self):
        super().__init__()
        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, n_q_head * d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_kv, bias=False)
        self.W_V = nn.Linear(d_model, d_kv, bias=False)

        # Output projection (concatenated heads → model dim)
        self.W_O = nn.Linear(n_q_head * d_head, d_model, bias=False)

        # SwiGLU MLP
        self.W_up_g = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, cache_k=None, cache_v=None):
        """
        x : (B, L, d_model)
        cache_k, cache_v : (B, total_len, d_kv)  or None during first step
        """
        B, L, _ = x.shape

        # ---------- Q, K, V ----------
        Q = self.W_Q(x)                         # (B, L, n_q_head*d_head)
        K = self.W_K(x)                         # (B, L, d_kv)
        V = self.W_V(x)                         # (B, L, d_kv)

        # reshape Q to heads
        Q = Q.view(B, L, n_q_head, d_head).transpose(1, 2)   # (B, n_q_head, L, d_head)

        # KV are shared across heads (GQA)
        K = K.view(B, L, n_kv_head, d_head).transpose(1, 2) # (B, n_kv_head, L, d_head)
        V = V.view(B, L, n_kv_head, d_head).transpose(1, 2) # (B, n_kv_head, L, d_head)

        # ---------- KV cache handling ----------
        if cache_k is not None:
            # concatenate new keys/values to the cache
            K = torch.cat([cache_k, K], dim=2)   # (B, n_kv_head, total_len, d_head)
            V = torch.cat([cache_v, V], dim=2)

        # ---------- Scaled dot‑product attention (fused) ----------
        # PyTorch 2.0+ provides a fused kernel under the hood:
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,          # causal mask is applied internally for decoder mode
            dropout_p=0.0,
            is_causal=True)          # <-- tells the kernel to apply the lower‑triangular mask

        # attn_out: (B, n_q_head, L, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)  # (B, L, n_q_head*d_head)

        # ---------- Output projection ----------
        attn_out = self.W_O(attn_out)

        # ---------- SwiGLU MLP ----------
        gate = F.silu(self.W_up_g(x))          # (B, L, d_ff)
        up   = self.W_up(x)                    # (B, L, d_ff)
        mlp  = (gate * up) @ self.W_down.weight.t()   # (B, L, d_model)

        # ---------- Residual ----------
        return x + attn_out + mlp, K, V
```

**What this code shows**

- The **attention** call `scaled_dot_product_attention` fuses:
  1. Q·Kᵀ GEMM,
  2. causal mask addition,
  3. scaling by `1/√d_head`,
  4. row‑wise soft‑max,
  5. multiplication with V.
- The **KV cache** is simply concatenated along the sequence dimension; during inference only the newest token’s Q is multiplied with the cached K/V, giving O(`seq_len`) work per step.
- The **MLP** uses two parallel up‑projections (`gate` and `up`) followed by a down‑projection, exactly matching the SwiGLU formula.

---

## 11. Putting It All Together – Inference Flow

```
for t in generation_steps:
    # 1️⃣ embed new token → x_t (B, 1, d_model)
    # 2️⃣ run through all 80 blocks, passing along KV caches
    # 3️⃣ each block:
    #    • fused attention (Q·Kᵀ + mask + softmax + V) → O_attn
    #    • SwiGLU MLP → O_mlp
    #    • residual add
    #    • update KV cache (append K_t, V_t)
    # 4️⃣ final linear → logits
    # 5️⃣ sample next token
```

Because the **attention kernel is fused**, the only per‑step memory traffic is:

- Write the new KV vectors (`8 × 128 floats` each) → ~4 KB per layer.  
- Read the cached KV (size grows linearly with generated length).  

All heavy matrix multiplications stay on‑chip, making Llama‑3 70B feasible on a few high‑end GPUs with tensor‑parallelism.

---

## 12. Recap – Key Take‑aways

| Topic | Core Insight |
|-------|--------------|
| **Model dimensions** | 8192‑wide, 80 layers, 64 Q‑heads, 8 KV‑heads, head size 128. |
| **Projection matrices** | Dense `8192×8192` (Q) and `8192×1024` (K,V). Not diagonal. |
| **GQA** | Reduces KV cache size 8× while keeping 64 Q‑heads. |
| **MLP (SwiGLU)** | Three dense matrices (≈ 0.71 B params per layer). |
| **GEMM** | The fundamental kernel; executed as tiled matrix‑multiply on GPU. |
| **Soft‑max** | Row‑wise reduction; cannot be replaced by a static lookup table. |
| **Kernel fusion** | FlashAttention‑style kernels combine QKᵀ, mask, soft‑max, and V‑multiply → massive bandwidth savings. |
| **KV cache** | Stores `seq_len × 1024` per layer; GQA makes it tractable. |
| **Approximate tricks** | Quantised `exp` tables can speed up the exponent, but reduction remains. |
| **Practical code** | `torch.nn.functional.scaled_dot_product_attention` already gives you the fused kernel on modern PyTorch. |

With this understanding you can now read the source code of Llama‑3, reason about memory footprints, and even write your own custom kernels (e.g., in Triton) that respect the same dimensions and constraints. Happy modeling! 🚀