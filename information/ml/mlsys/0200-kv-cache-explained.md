### The Core Idea: Why Do We Need a Cache?

In autoregressive generation (like a language model writing a story), we generate one word (or "token") at a time. The core mechanism of a Transformer is the **self-attention** layer, which allows a token to "look at" all the other tokens that came before it to understand the context.

**The Problem:** Without a cache, every time we generate a *new* token, we would have to re-calculate the attention information for *all* the previous tokens in the sequence. This is incredibly redundant and computationally expensive.

**The Solution: The K-V Cache.** The cache is a simple and brilliant optimization. It acts as the model's "short-term memory." Instead of re-computing everything, we just save the **Key (K)** and **Value (V)** vectors for each token as it's processed. When we generate a new token, we only need to compute its *own* Q, K, and V vectors and then combine them with the K and V vectors we saved from all the previous steps.

---

### Setting the Stage: Our Simplified Model

*   **Single-Head Attention:** We only have one attention mechanism.
*   **Projection Matrices:** $W_Q, W_K, W_V$ are all of size **8x8**.
*   **Embedding Dimension:** To make the math work, our token embeddings must also have a dimension of 8.

So, any input token embedding `x` will be a vector of size `(1, 8)`.

---

### Computation Walkthrough: Generating a Sentence

Let's generate the sentence "The cat sat". We'll process it token by token.

#### Step 1: Processing "The"

1.  **Input:** The first token is "The". Its embedding is `x_1`, a `(1, 8)` vector.
2.  **Compute Q, K, V:** We compute the Query, Key, and Value vectors for this token.
    *   $Q_1 = x_1 \cdot W_Q$  ->  `(1, 8) @ (8, 8) = (1, 8)` vector
    *   $K_1 = x_1 \cdot W_K$  ->  `(1, 8) @ (8, 8) = (1, 8)` vector
    *   $V_1 = x_1 \cdot W_V$  ->  `(1, 8) @ (8, 8) = (1, 8)` vector

3.  **Initialize the Cache:** Since this is the first token, our cache is empty. We now store the Key and Value we just computed.

    *   `K_cache` = $[K_1]$  (Shape: `(1, 8)`)
    *   `V_cache` = $[V_1]$  (Shape: `(1, 8)`)

4.  **Calculate Attention:** The token "The" attends to itself.
    *   `scores = Q_1 @ K_cache.T` -> `(1, 8) @ (8, 1) = (1, 1)` (a single score)
    *   `attention_weights = softmax(scores)` -> `(1, 1)`
    *   `output_1 = attention_weights @ V_cache` -> `(1, 1) @ (1, 8) = (1, 8)`

This `output_1` vector is then passed to the next layer of the model, which eventually predicts the next token, "cat".

#### Step 2: Processing "cat"

1.  **Input:** The new token is "cat". Its embedding is `x_2`, a `(1, 8)` vector.
2.  **Compute Q, K, V (for the new token only):**
    *   $Q_2 = x_2 \cdot W_Q$ -> `(1, 8)` vector
    *   $K_2 = x_2 \cdot W_K$ -> `(1, 8)` vector
    *   $V_2 = x_2 \cdot W_V$ -> `(1, 8)` vector

3.  **Use and Update the Cache:** This is the key step!
    *   **Retrieve** the old keys and values from the cache: `K_cache` is $[K_1]$ and `V_cache` is $[V_1]$.
    *   **Concatenate** the new K and V with the cached ones.
        *   `All_K = concat(K_cache, K_2)` -> $[K_1, K_2]$ (Shape: `(2, 8)`)
        *   `All_V = concat(V_cache, V_2)` -> $[V_1, V_2]$ (Shape: `(2, 8)`)
    *   **Update** the cache for the next step. The new cache now holds the K and V for both "The" and "cat".
        *   `K_cache` is now $[K_1, K_2]$
        *   `V_cache` is now $[V_1, V_2]$

4.  **Calculate Attention:** The token "cat" ($Q_2$) now attends to both "The" and "cat" (`All_K`).
    *   `scores = Q_2 @ All_K.T` -> `(1, 8) @ (8, 2) = (1, 2)` (Two scores: one for how much "cat" attends to "The", one for how much it attends to itself).
    *   `attention_weights = softmax(scores)` -> `(1, 2)`
    *   `output_2 = attention_weights @ All_V` -> `(1, 2) @ (2, 8) = (1, 8)`

This `output_2` vector (which contains context from both "The" and "cat") is used to predict the next token, "sat".

#### Step 3: Processing "sat"

1.  **Input:** The new token is "sat". Its embedding is `x_3`, a `(1, 8)` vector.
2.  **Compute Q, K, V (for "sat" only):**
    *   $Q_3 = x_3 \cdot W_Q$ -> `(1, 8)` vector
    *   $K_3 = x_3 \cdot W_K$ -> `(1, 8)` vector
    *   $V_3 = x_3 \cdot W_V$ -> `(1, 8)` vector

3.  **Use and Update the Cache:**
    *   **Retrieve** `K_cache` ($[K_1, K_2]$) and `V_cache` ($[V_1, V_2]$).
    *   **Concatenate** the new K and V.
        *   `All_K = concat(K_cache, K_3)` -> $[K_1, K_2, K_3]$ (Shape: `(3, 8)`)
        *   `All_V = concat(V_cache, V_3)` -> $[V_1, V_2, V_3]$ (Shape: `(3, 8)`)
    *   **Update** the cache for the future: `K_cache` is now $[K_1, K_2, K_3]$ and `V_cache` is now $[V_1, V_2, V_3]$.

4.  **Calculate Attention:** The token "sat" ($Q_3$) attends to "The", "cat", and "sat" (`All_K`).
    *   `scores = Q_3 @ All_K.T` -> `(1, 8) @ (8, 3) = (1, 3)` (Three scores).
    *   `attention_weights = softmax(scores)` -> `(1, 3)`
    *   `output_3 = attention_weights @ All_V` -> `(1, 3) @ (3, 8) = (1, 8)`

This process continues for every new token generated.

### Summary of Benefits

*   **Computational Efficiency:** For the N-th token, we perform calculations relative to N past tokens, not N². We only compute Q, K, and V for the single new token, which is a fixed cost. The matrix multiplication `Q @ K.T` grows linearly with the sequence length, but we avoid re-computing all the previous K and V vectors, which would be a quadratic cost.
*   **Enables Fast Inference:** The K-V cache is the fundamental reason why generating text from large language models is feasible in real-time. Without it, generating each new word would get progressively and prohibitively slower.
*   **Memory Footprint:** The cache only needs to store the K and V vectors. For multi-head attention, you'd have a separate K and V cache for each head. The size of the cache is `(num_layers, 2, batch_size, num_heads, sequence_length, head_dim)`. It grows only with the length of the sequence being generated.
