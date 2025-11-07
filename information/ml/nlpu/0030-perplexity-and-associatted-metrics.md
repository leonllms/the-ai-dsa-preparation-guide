# Perplexity

Perplexity is a measure of how well a language model predicts a sequence of tokens.  
For a test set with N tokens and probability P assigned to each token by the model,  

```
perplexity = exp( - (1/N) * Σ log P(token_i) )
```

A lower perplexity means the model (or the tokenizer‑model pair) assigns higher probability to the observed tokens, i.e., it is less “surprised”.

When evaluating a tokenizer, perplexity is computed by training a language model on text tokenized with that tokenizer and then measuring the model’s perplexity on held‑out data. The tokenizer that yields lower perplexity is considered better because it produces token sequences that are easier for the model to predict.

**Other tokenizer‑evaluation metrics**

| Metric | What it measures | Typical use |
|--------|------------------|-------------|
| **Vocabulary size vs. coverage** | Fraction of characters/words in a corpus that can be represented without OOV (out‑of‑vocabulary) tokens. | Checks trade‑off between compact vocab and coverage. |
| **Token length (average tokens per word or per sentence)** | How many tokens are produced for a given amount of text. | Fewer tokens → faster training/inference, but may lose granularity. |
| **BLEU / ROUGE on detokenized reconstruction** | Compare original text with text reconstructed after tokenization → detokenization. | Measures loss of information or distortion caused by tokenization. |
| **Character‑level reconstruction error** | Percentage of characters that differ between original and detokenized text. | Fine‑grained check for exact recovery. |
| **Subword regularity (e.g., Zipf‑like distribution)** | How well token frequencies follow expected linguistic distributions. | Indicates naturalness of token splits. |
| **Speed / memory footprint** | Time to tokenize a fixed corpus and memory needed for vocab lookup. | Practical performance metric. |
| **Downstream task performance** | Accuracy, F1, BLEU, etc., of a model trained on data tokenized with the tokenizer. | Ultimate test: does the tokenizer help the target task? |
| **Morphological consistency** | Proportion of tokens that align with true morpheme boundaries (often measured against a gold morphological segmentation). | Useful for languages with rich morphology. |

**How to choose**

1. Start with coverage and average token length to ensure the tokenizer can represent the data efficiently.  
2. Verify that detokenization loss is negligible (BLEU ≈ 1, low character error).  
3. Train a small language model and record perplexity – lower is better.  
4. Finally, confirm that the tokenizer improves the metric of your downstream task (e.g., classification accuracy).  

These metrics together give a comprehensive picture of tokenizer quality beyond perplexity alone.

For reference provide details for each of the metrics listed in the table. 
Specifically, the mathematical type, explanation, and a schoolbook implementation
in python.

Below is a **compact “cheat‑sheet”** for every metric that appears in the table.  
For each metric I give

| Item | What you see |
|------|--------------|
| **Mathematical type** – the kind of quantity (real‑valued, ratio, distribution, …) |
| **Formal definition** – a short equation or statistical description |
| **Intuition / explanation** – why the metric matters and how to read it |
| **Reference implementation** – a minimal, self‑contained Python function that you can drop into a notebook or script (no heavy‑weight framework required).  The code uses only the Python standard library and a few ubiquitous packages (`numpy`, `torch`, `nltk`, `sklearn`).  If you do not have a package, the import line can be removed or replaced with a simple stub.  

Feel free to copy‑paste each block into a single file; the functions are deliberately **stateless** (they take the data they need as arguments and return a scalar or a small dictionary).

---

## 1. Vocabulary size vs. coverage

| **Mathematical type** | Real number in **[0, 1]** (coverage) and integer (vocab size) |
|---|---|
| **Definition** |  

\[
\text{coverage}(C, V)=\frac{1}{|C|}\sum_{w\in C}\mathbf{1}\bigl[w\in V\bigr]
\]

where  

* \(C\) = set (or multiset) of tokens/words in the corpus,  
* \(V\) = vocabulary of the tokenizer,  
* \(\mathbf{1}[\cdot]\) = indicator function (1 if true, 0 otherwise).  

| **Explanation** |  
* **Vocabulary size** tells you how many distinct symbols the tokenizer knows.  
* **Coverage** tells you the fraction of tokens in a target corpus that can be expressed *without* falling back to an “unknown” (OOV) token.  
* High coverage with a *small* vocab is the sweet spot – it means the tokenizer is compact yet expressive. |
| **Python implementation** |  

```python
from collections import Counter
from typing import Iterable, Set, Tuple

def vocab_coverage(
    corpus_tokens: Iterable[str],
    vocab: Set[str],
    oov_token: str = "<unk>"
) -> Tuple[int, float]:
    """
    Returns (vocab_size, coverage) for a given token list and a vocab set.
    
    Parameters
    ----------
    corpus_tokens : iterable of str
        Tokens (or words) that appear in the evaluation corpus.
    vocab : set of str
        Tokenizer vocabulary.
    oov_token : str, optional
        Symbol used for out‑of‑vocabulary items (ignored for coverage).

    Returns
    -------
    vocab_size : int
        Number of unique entries in `vocab` (excluding the OOV token if present).
    coverage : float
        Fraction of tokens in `corpus_tokens` that are present in `vocab`.
    """
    # Remove the OOV placeholder from the vocab count (optional)
    vocab_size = len(vocab - {oov_token})
    total = 0
    covered = 0
    for t in corpus_tokens:
        total += 1
        if t in vocab:
            covered += 1
    coverage = covered / total if total > 0 else 0.0
    return vocab_size, coverage


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # pretend we have a tiny corpus
    corpus = "the quick brown fox jumps over the lazy dog".split()
    # a BPE‑style vocab (just for illustration)
    vocab = {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "##s"}
    print(vocab_coverage(corpus, vocab))
```

---

## 2. Token length (average tokens per word or per sentence)

| **Mathematical type** | Real number (average tokens) |
|---|---|
| **Definition** |  

\[
\text{avg\_tokens\_per\_unit}= \frac{1}{N}\sum_{i=1}^{N} \bigl|\,\text{tokens}(u_i)\,\bigr|
\]

where  

* \(u_i\) = a *unit* (either a word, a sentence, or any chunk you decide),  
* \(\text{tokens}(u_i)\) = token list produced for that unit,  
* \(N\) = number of units. |

| **Explanation** |  
* A **smaller** average means the tokenizer is *coarser* (fewer pieces per word), which speeds up training and inference because the model works on fewer steps.  
* A **larger** average indicates a finer granularity (often better for rare words or morphologically rich languages) but increases compute cost.  
* You can compute the metric at two levels:  
  * **Tokens per word** – useful for assessing how much a word is broken up.  
  * **Tokens per sentence** – useful for estimating sequence length for language‑model training. |
| **Python implementation** |  

```python
from typing import List, Callable

def avg_tokens_per_unit(
    units: List[str],
    tokenizer: Callable[[str], List[str]],
) -> float:
    """
    Compute the average number of tokens produced for a list of textual units.
    
    Parameters
    ----------
    units : list of str
        Textual units (words, sentences, paragraphs …) to be tokenized.
    tokenizer : callable
        Function that maps a string -> list of token strings.
        Example: lambda s: tokenizer.encode(s, add_special_tokens=False)
    
    Returns
    -------
    avg_len : float
        Average token count per unit.
    """
    if not units:
        return 0.0
    total_tokens = sum(len(tokenizer(u)) for u in units)
    return total_tokens / len(units)


# ----------------------------------------------------------------------
# Example with the HuggingFace tokenizer (requires `transformers`)
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # tokens per word
    words = "the quick brown fox".split()
    print("Tokens per word:", avg_tokens_per_unit(words, tokenizer))
    
    # tokens per sentence
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love natural language processing!"
    ]
    print("Tokens per sentence:", avg_tokens_per_unit(sentences, tokenizer))
```

---

## 3. BLEU / ROUGE on detokenized reconstruction

Both metrics compare **original text** with **detokenized‑then‑re‑tokenized text** (i.e., the round‑trip).  
The goal is to see whether the tokenizer + detokenizer pair loses information.

### 3.1 BLEU (sentence‑level, 4‑gram, brevity penalty)

| **Mathematical type** | Real number in **[0, 1]** (often multiplied by 100) |
|---|---|
| **Definition** |  

\[
\text{BLEU}= \text{BP}\,\exp\Bigl(\sum_{n=1}^{4} w_n \log p_n\Bigr)
\]

with  

* \(p_n =\frac{\sum_{\text{ngram}\in C} \text{Count}_{\text{clip}}(\text{ngram})}{\sum_{\text{ngram}\in C} \text{Count}(\text{ngram})}\) – clipped n‑gram precision,  
* \(w_n = \frac{1}{4}\) (uniform weights),  
* \(\text{BP}= \begin{cases}
1 & \text{if } |C| > |R|\\
\exp\bigl(1-\frac{|R|}{|C|}\bigr) & \text{otherwise}
\end{cases}\) – brevity penalty,  
* \(C\) = candidate (detokenized) sentence, \(R\) = reference (original) sentence. |

| **Explanation** |  
* BLEU ≈ 1 (or 100 %) means the round‑trip reproduces the reference almost word‑for‑word.  
* BLEU is *n‑gram* based, so it tolerates small re‑ordering but penalises missing or extra tokens. |
| **Python implementation** |  

```python
import math
from collections import Counter
from typing import List

def _ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n‑grams from a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def sentence_bleu(reference: List[str], candidate: List[str], max_n: int = 4) -> float:
    """
    Compute sentence‑level BLEU (uniform weights, no smoothing).
    Returns a score in the range [0, 1].
    """
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / cand_len)

    # Geometric mean of clipped precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _ngrams(reference, n)
        cand_counts = _ngrams(candidate, n)

        # Clip candidate counts by reference counts
        overlap = {ng: min(count, ref_counts.get(ng, 0))
                   for ng, count in cand_counts.items()}
        clipped = sum(overlap.values())
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(clipped / total)

    # If any precision is zero, BLEU becomes zero (no smoothing)
    if min(precisions) == 0:
        return 0.0

    # Uniform weights
    log_prec = sum((1.0 / max_n) * math.log(p) for p in precisions)
    bleu = bp * math.exp(log_prec)
    return bleu


# ----------------------------------------------------------------------
# Example round‑trip test
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    original = "I love natural‑language processing!  🤖"
    # tokenization → detokenization
    ids = tokenizer.encode(original, add_special_tokens=False)
    roundtrip = tokenizer.decode(ids, skip_special_tokens=True)
    
    # Tokenize both strings for BLEU (simple whitespace split works for English)
    ref_tokens = original.split()
    cand_tokens = roundtrip.split()
    print("BLEU:", sentence_bleu(ref_tokens, cand_tokens))
```

### 3.2 ROUGE‑L (Longest Common Subsequence)

| **Mathematical type** | Real number in **[0, 1]** (recall, precision, F‑measure) |
|---|---|
| **Definition** |  

\[
\text{ROUGE-L}_{\text{F}} = \frac{(1+\beta^2) \, \text{R} \, \text{P}}
                                 {\beta^2 \, \text{R} + \text{P}}
\]

where  

* \(\text{R} = \frac{LCS(\text{ref},\text{cand})}{|\text{ref}|}\) – recall,  
* \(\text{P} = \frac{LCS(\text{ref},\text{cand})}{|\text{cand}|}\) – precision,  
* \(LCS\) = length of the longest common subsequence (order‑preserving, not necessarily contiguous),  
* \(\beta = 1\) (default) gives the harmonic mean. |

| **Explanation** |  
* ROUGE‑L measures **sequence‑level overlap** (the longest ordered subsequence) and is less sensitive to exact n‑gram matching than BLEU.  
* It is commonly used for summarisation but works well for a detokenization check because it rewards preserving word order. |
| **Python implementation** |  

```python
def _lcs_length(x: List[str], y: List[str]) -> int:
    """Dynamic‑programming LCS length (O(|x|·|y|))."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]

def rouge_l(reference: List[str], candidate: List[str], beta: float = 1.0) -> float:
    """
    Compute ROUGE‑L F‑measure (beta‑weighted harmonic mean of precision/recall).
    Returns a score in [0, 1].
    """
    lcs = _lcs_length(reference, candidate)
    if lcs == 0:
        return 0.0
    recall = lcs / len(reference)
    precision = lcs / len(candidate)
    beta_sq = beta ** 2
    f_score = ((1 + beta_sq) * recall * precision) / (beta_sq * recall + precision)
    return f_score


# ----------------------------------------------------------------------
# Example usage (same round‑trip as before)
if __name__ == "__main__":
    original = "I love natural‑language processing!  🤖"
    ids = tokenizer.encode(original, add_special_tokens=False)
    roundtrip = tokenizer.decode(ids, skip_special_tokens=True)
    ref = original.split()
    cand = roundtrip.split()
    print("ROUGE‑L:", rouge_l(ref, cand))
```

---

## 4. Character‑level reconstruction error

| **Mathematical type** | Percentage (0 % – 100 %) |
|---|---|
| **Definition** |  

\[
\text{CharError\%} = 100 \times \frac{1}{|S|}\sum_{i=1}^{|S|}\mathbf{1}\bigl[\,c_i^{\text{orig}} \neq c_i^{\text{detok}}\bigr]
\]

where  

* \(S\) = set of characters in the *reference* string,  
* \(c_i^{\text{orig}}\) = i‑th character of the original,  
* \(c_i^{\text{detok}}\) = i‑th character after the round‑trip (padded/truncated to the same length).  

If the lengths differ, we first **align** by padding the shorter string with a special sentinel (e.g., `\0`) so the indicator works element‑wise. |

| **Explanation** |  
* This metric is a **strict** sanity check: any mismatch—space, punctuation, Unicode normalisation—counts as an error.  
* In practice, a well‑behaved tokenizer‑detokenizer pair should have **< 0.1 %** error on clean English data. |
| **Python implementation** |  

```python
def char_error_rate(original: str, roundtrip: str) -> float:
    """
    Return character‑level error rate (percentage) between two strings.
    """
    # Normalise to the same length by padding the shorter string
    max_len = max(len(original), len(roundtrip))
    o = original.ljust(max_len, "\0")
    r = roundtrip.ljust(max_len, "\0")
    errors = sum(1 for a, b in zip(o, r) if a != b)
    return (errors / max_len) * 100.0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Char error %:", char_error_rate(original, roundtrip))
```

---

## 5. Subword regularity (Zipf‑like distribution)

| **Mathematical type** | Real number (exponent of a power‑law fit) + goodness‑of‑fit (R²) |
|---|---|
| **Definition** | For a token frequency list \(\{f_i\}_{i=1}^{V}\) sorted descending, Zipf’s law predicts  

\[
f_i \approx C \, i^{-\alpha}
\]

where \(\alpha\) is the **Zipf exponent**.  We estimate \(\alpha\) by linear regression on the log‑log plot:

\[
\log f_i = \log C - \alpha \log i + \varepsilon_i
\]

The coefficient of determination \(R^{2}\) measures how well the power‑law fits. |

| **Explanation** |  
* A *good* subword vocabulary exhibits a **smooth, approximately linear** relationship on a log‑log plot, i.e., high \(R^{2}\) (≥ 0.9).  
* The exponent \(\alpha\) is typically around **1.0 – 1.2** for natural language; values far outside this range suggest an unnatural token split (too many rare fragments or too many overly common tokens). |
| **Python implementation** |  

```python
import numpy as np
from collections import Counter
from typing import List, Tuple
from sklearn.linear_model import LinearRegression

def zipf_fit(token_list: List[str]) -> Tuple[float, float]:
    """
    Fit a Zipf (power‑law) to token frequencies.
    
    Returns
    -------
    alpha : float
        Estimated Zipf exponent (slope with sign flipped).
    r2   : float
        Coefficient of determination (goodness‑of‑fit).
    """
    # Frequency of each token
    freq = np.array(sorted(Counter(token_list).values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freq) + 1, dtype=float)

    # Log‑log transform
    log_r = np.log(ranks).reshape(-1, 1)   # predictor
    log_f = np.log(freq)                    # response

    # Linear regression (no intercept -> intercept = log C)
    model = LinearRegression()
    model.fit(log_r, log_f)
    slope = model.coef_[0]          # this is -alpha
    r2 = model.score(log_r, log_f)  # R^2

    alpha = -slope
    return alpha, r2


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: tokenizing a small corpus with a word‑piece tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "I love natural language processing",
        "subword tokenization is fun"
    ]
    tokens = [t for line in corpus for t in tokenizer.tokenize(line)]
    alpha, r2 = zipf_fit(tokens)
    print(f"Zipf exponent α = {alpha:.3f}, R² = {r2:.3f}")
```

---

## 6. Speed / memory footprint

Speed and memory are **operational** metrics.  
We usually report:

* **Throughput** – tokens per second (or characters per second).  
* **Peak RAM** – maximum resident set size while tokenizing a fixed‑size batch.  

Both can be measured with the standard library (`time`, `resource` on Unix) or with third‑party helpers (`psutil`).

| **Mathematical type** | Real number (tokens · s⁻¹) and real number (MiB) |
|---|---|
| **Definition** |  

\[
\text{throughput} = \frac{N_{\text{tokens}}}{t_{\text{elapsed}}}
\qquad
\text{peak\_mem} = \max_{0\le\tau\le t_{\text{elapsed}}} \text{RSS}(\tau)
\]

where \(N_{\text{tokens}}\) is the total token count processed, \(t_{\text{elapsed}}\) is wall‑clock time, and RSS = Resident Set Size (memory in bytes). |

| **Explanation** |  
* **Higher throughput** → faster tokenization, which matters for large corpora or on‑device inference.  
* **Lower peak memory** → can run on constrained devices (mobile, edge).  
* Reporting both lets you see the classic speed‑vs‑memory trade‑off (e.g., a trie‑based tokenizer may be fast but memory‑hungry). |
| **Python implementation** |  

```python
import time
import psutil
import os
from typing import Iterable, Callable

def measure_tokenizer_speed(
    texts: Iterable[str],
    tokenizer: Callable[[str], List[str]],
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Measure throughput (tokens/s) and peak RAM (MiB) while tokenizing `texts`.
    
    Parameters
    ----------
    texts : iterable of str
        Input strings to be tokenized.
    tokenizer : callable
        Function that maps a string -> list of token strings.
    batch_size : int, optional
        Number of strings processed per batch (helps amortise Python overhead).
    
    Returns
    -------
    throughput : float
        Tokens per second.
    peak_mem_mib : float
        Peak resident memory usage in MiB.
    """
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss  # bytes
    peak_mem = start_mem
    total_tokens = 0

    start_time = time.time()
    batch = []
    for i, txt in enumerate(texts, 1):
        batch.append(txt)
        if len(batch) == batch_size:
            # tokenise the whole batch
            for s in batch:
                total_tokens += len(tokenizer(s))
            batch.clear()
            # record memory after each batch
            peak_mem = max(peak_mem, process.memory_info().rss)

    # leftover batch
    for s in batch:
        total_tokens += len(tokenizer(s))
    peak_mem = max(peak_mem, process.memory_info().rss)

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    peak_mem_mib = peak_mem / (1024 * 1024)

    return throughput, peak_mem_mib


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple whitespace tokenizer for demo
    whitespace_tokenizer = lambda s: s.split()
    # Fake a large list of sentences
    dummy_corpus = ["This is a sample sentence."] * 10_000
    tp, mem = measure_tokenizer_speed(dummy_corpus, whitespace_tokenizer, batch_size=256)
    print(f"Throughput: {tp:,.0f} tokens/s, Peak RAM: {mem:.1f} MiB")
```

*If you prefer the built‑in `resource` module (Unix only), replace the `psutil` calls with `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (the latter returns KiB on Linux, bytes on macOS).*

---

## 7. Downstream task performance

| **Mathematical type** | Depends on the task – typical scalars: accuracy (0–1), F1 (0–1), BLEU (0–100), etc. |
|---|---|
| **Definition** | Let \(\mathcal{M}_V\) be a model trained on data tokenized with vocabulary \(V\).  For a downstream task \(T\) with a standard evaluation metric \(\mathcal{E}\) (e.g., classification accuracy, QA exact‑match), the **downstream performance** is  

\[
\text{Perf}_T(V) = \mathcal{E}\bigl(\mathcal{M}_V, \text{test}_T\bigr)
\]

where \(\text{test}_T\) is the held‑out test split for task \(T\). |

| **Explanation** |  
* This is the **ultimate** yardstick: a tokenizer that yields low perplexity but hurts the end‑task is not useful.  
* When comparing multiple tokenizers you typically **control** everything else (model architecture, hyper‑parameters, random seed) and only vary the tokenization step. |
| **Python implementation** | Below is a **template** that works for any scikit‑learn‑compatible task (e.g., text classification).  Replace the model/training code with your own (PyTorch, TensorFlow, HuggingFace, …) – the evaluation logic stays the same. |

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple, Callable

def downstream_classification(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer: Callable[[str], List[str]],
    metric: str = "accuracy",
) -> float:
    """
    Train a simple linear classifier on tokenized text and report a metric.
    
    Parameters
    ----------
    train_texts, test_texts : list of str
        Raw sentences.
    train_labels, test_labels : list of int (or str)
        Ground‑truth class IDs.
    tokenizer : callable
        Function that maps a string -> list of token strings.
    metric : {"accuracy", "f1"}
        Which evaluation metric to return.
    
    Returns
    -------
    score : float
        Metric value on the test split.
    """
    # Convert token lists back to a whitespace‑joined string for CountVectorizer.
    # (CountVectorizer can also accept a custom analyzer, but this keeps it simple.)
    train_tok = [" ".join(tokenizer(t)) for t in train_texts]
    test_tok  = [" ".join(tokenizer(t)) for t in test_texts]

    vectorizer = CountVectorizer(min_df=1, token_pattern=r"(?u)\S+")
    X_train = vectorizer.fit_transform(train_tok)
    X_test  = vectorizer.transform(test_tok)

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    if metric == "accuracy":
        return accuracy_score(test_labels, preds)
    elif metric == "f1":
        return f1_score(test_labels, preds, average="macro")
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny dummy dataset
    train_txt = ["I love cats", "Dogs are great", "I hate rain", "The sky is blue"]
    train_lbl = [0, 0, 1, 1]          # 0 = positive, 1 = negative (just for demo)
    test_txt  = ["Cats are cute", "Rain is terrible"]
    test_lbl  = [0, 1]

    # Use the same whitespace tokenizer as before
    acc = downstream_classification(train_txt, train_lbl, test_txt, test_lbl,
                                    tokenizer=lambda s: s.split(),
                                    metric="accuracy")
    print(f"Downstream classification accuracy: {acc:.2%}")
```

*When you have a **pre‑trained language model** (e.g., BERT) the function would instead:

1. Tokenize with `tokenizer.encode_plus` (using the tokenizer you are evaluating).  
2. Feed the IDs into the model, fine‑tune on the training set, and compute the same metric on the test set.  

The **scalar** you obtain (`accuracy`, `F1`, `BLEU`, etc.) is the downstream performance for that tokenizer.*

---

## 8. Morphological consistency

| **Mathematical type** | Ratio (0 – 1) – proportion of tokens that respect morpheme boundaries |
|---|---|
| **Definition** |  

\[
\text{MorphCons} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\bigl[\text{token}_i \in \mathcal{M}\bigr]
\]

where  

* \(N\) = total number of subword tokens produced on a morphologically annotated test set,  
* \(\mathcal{M}\) = set of **gold‑standard morphemes** (e.g., from a morphological analyzer or a manually curated list).  
A token is counted as *consistent* if it **exactly matches** a morpheme boundary (or is a concatenation of whole morphemes without splitting inside a morpheme). |

| **Explanation** |  
* For languages with rich morphology (Turkish, Finnish, Arabic, etc.) a tokenizer that frequently splits inside a morpheme loses linguistic information.  
* **Higher** MorphCons (≈ 0.9) indicates that most subwords align with real morphemes, which often improves downstream tasks that benefit from morphological cues (e.g., POS tagging, NER). |
| **Python implementation** | The function expects a **gold segmentation** in the form of a list of lists, where each inner list contains the morphemes for a word.  

```python
from typing import List, Tuple, Set

def morphological_consistency(
    tokenized_words: List[List[str]],
    gold_morphemes: List[List[str]],
) -> float:
    """
    Compute the proportion of subword tokens that respect gold morpheme boundaries.
    
    Parameters
    ----------
    tokenized_words : list of list of str
        Tokenizer output per word, e.g. [["un", "##break", "##able"], ["cat"]].
    gold_morphemes : list of list of str
        Gold morpheme segmentation per word, e.g. [["un", "break", "able"], ["cat"]].
    
    Returns
    -------
    consistency : float
        Ratio in [0, 1]; 1.0 means *every* token aligns with a morpheme.
    """
    assert len(tokenized_words) == len(gold_morphemes), "Mismatched lengths"

    total_tokens = 0
    aligned_tokens = 0

    for toks, gold in zip(tokenized_words, gold_morphemes):
        # Build the set of *allowed* token strings from gold morphemes.
        # For simplicity we treat each morpheme as a possible token.
        allowed: Set[str] = set(gold)

        # Count tokens that are exactly a morpheme.
        for t in toks:
            total_tokens += 1
            if t in allowed:
                aligned_tokens += 1

    return aligned_tokens / total_tokens if total_tokens > 0 else 0.0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example with an English BPE tokenizer (just a toy illustration)
    tokenizer_output = [["un", "##break", "##able"], ["cat"]]
    gold_morphemes   = [["un", "break", "able"], ["cat"]]
    print("Morphological consistency:", morphological_consistency(tokenizer_output,
                                                                 gold_morphemes))
```

*In practice you would obtain `gold_morphemes` from a morphological analyzer such as **Morfessor**, **UDPipe**, or a manually annotated corpus.  For languages with complex concatenative morphology you may also want to allow concatenations of consecutive morphemes (e.g., token `"##breakable"` equals `"break"+"able"`).  Extending the function to handle that is straightforward: build all possible concatenations of adjacent gold morphemes and add them to `allowed`. *

---

# Putting It All Together

Below is a **single driver** that runs the whole suite on a small demo corpus.  It shows how you can combine the individual functions to produce a *report card* for a tokenizer.

```python
def tokenizer_report(
    corpus: List[str],
    tokenizer,
    gold_morph: List[List[str]] = None,
    downstream_data: Tuple[List[str], List[int], List[str], List[int]] = None,
):
    """
    Run every metric from the cheat‑sheet and print a concise summary.
    
    Parameters
    ----------
    corpus : list of raw sentences (str)
    tokenizer : any object with .tokenize(str) → List[str] and .decode(List[int]) → str
    gold_morph : optional gold morphological segmentation (same length as tokenized words)
    downstream_data : optional (train_texts, train_labels, test_texts, test_labels)
    """
    # 1. Vocabulary size & coverage
    vocab = set(tokenizer.get_vocab().keys()) if hasattr(tokenizer, "get_vocab") else set()
    tokens = [t for line in corpus for t in tokenizer.tokenize(line)]
    vocab_sz, cov = vocab_coverage(tokens, vocab)

    # 2. Token length
    avg_per_word = avg_tokens_per_unit(
        [w for line in corpus for w in line.split()], tokenizer.tokenize
    )
    avg_per_sent = avg_tokens_per_unit(corpus, tokenizer.tokenize)

    # 3. BLEU / ROUGE on detokenization
    roundtrip = [tokenizer.decode(tokenizer.encode(l, add_special_tokens=False))
                 for l in corpus]
    bleu_scores = [sentence_bleu(ref.split(), cand.split())
                   for ref, cand in zip(corpus, roundtrip)]
    rouge_scores = [rouge_l(ref.split(), cand.split())
                    for ref, cand in zip(corpus, roundtrip)]

    # 4. Char‑error
    char_err = np.mean([char_error_rate(r, c) for r, c in zip(corpus, roundtrip)])

    # 5. Zipf regularity
    alpha, r2 = zipf_fit(tokens)

    # 6. Speed / memory
    tp, mem = measure_tokenizer_speed(corpus, tokenizer.tokenize, batch_size=128)

    # 7. Downstream task (if supplied)
    downstream_perf = None
    if downstream_data is not None:
        tr_txt, tr_lbl, te_txt, te_lbl = downstream_data
        downstream_perf = downstream_classification(
            tr_txt, tr_lbl, te_txt, te_lbl,
            tokenizer=lambda s: tokenizer.tokenize(s),
            metric="accuracy",
        )

    # 8. Morphological consistency (if gold provided)
    morph_cons = None
    if gold_morph is not None:
        # Align tokenized words with gold morphemes (simple word‑wise split)
        tokenized_words = [tokenizer.tokenize(w) for line in corpus for w in line.split()]
        morph_cons = morphological_consistency(tokenized_words, gold_morph)

    # ---------- Print ----------
    print("\n--- Tokenizer Report ------------------------------------------------")
    print(f"Vocab size                : {vocab_sz}")
    print(f"Coverage (tokens)         : {cov:.2%}")
    print(f"Avg tokens / word         : {avg_per_word:.2f}")
    print(f"Avg tokens / sentence     : {avg_per_sent:.2f}")
    print(f"BLEU (round‑trip) mean    : {np.mean(bleu_scores):.3f}")
    print(f"ROUGE‑L (round‑trip) mean: {np.mean(rouge_scores):.3f}")
    print(f"Char‑error %              : {char_err:.4f}")
    print(f"Zipf exponent α          : {alpha:.3f} (R²={r2:.3f})")
    print(f"Throughput                : {tp:,.0f} tokens/s")
    print(f"Peak RAM                  : {mem:.1f} MiB")
    if downstream_perf is not None:
        print(f"Downstream accuracy       : {downstream_perf:.2%}")
    if morph_cons is not None:
        print(f"Morphological consistency: {morph_cons:.2%}")
    print("---------------------------------------------------------------------\n")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Demo with the HuggingFace GPT‑2 tokenizer (byte‑level BPE)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("gpt2")
    demo_corpus = [
        "I love natural language processing.",
        "The quick brown fox jumps over the lazy dog!",
        "¡Hola! ¿Cómo estás?",
        "今日は天気がいいです。"
    ]
    tokenizer_report(demo_corpus, tk)
```

The **report** produced by the script gives you a one‑page snapshot of everything discussed:

| Metric | Typical “good” range |
|--------|---------------------|
| Vocabulary coverage | ≥ 0.95 (for in‑domain data) |
| Tokens / word | 1.1 – 1.5 (English) |
| BLEU (round‑trip) | > 0.98 |
| Char‑error % | < 0.1 % |
| Zipf α | 1.0 – 1.2, R² ≥ 0.9 |
| Throughput | > 10 k tokens/s on a modern CPU |
| Peak RAM | < 200 MiB for a BPE vocab ≤ 30 k |
| Downstream accuracy | Should be on par with a baseline tokenizer |
| MorphCons | ≥ 0.9 for morphologically rich languages |

You can now **swap in any tokenizer** (SentencePiece, WordPiece, Unigram, custom rule‑based) and instantly see how it fares across the full spectrum of evaluation criteria.  

Happy tokenizing! 🚀