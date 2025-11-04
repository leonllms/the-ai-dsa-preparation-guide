# Tokenizer

## 1️⃣  What is a **tokenizer**?

In natural‑language processing (NLP) a *tokenizer* is the component that turns raw text (a string of characters) into a sequence of **tokens** – the atomic units a model consumes.  
Tokens can be:

| Token type | Example (sentence: “I love NLP!”) |
|------------|-----------------------------------|
| **Word‑level** | `["I", "love", "NLP", "!"]` |
| **Character‑level** | `["I", " ", "l", "o", "v", "e", …]` |
| **Sub‑word** (e.g. BPE, WordPiece, Unigram) | `["I", "love", "NL", "##P", "!"]` |
| **Byte‑level** (e.g. GPT‑2) | `["ĠI", "Ġlove", "ĠN", "L", "P", "!"]` |

The tokenizer does **more than just split on spaces**:

1. **Normalisation** – lower‑casing, Unicode NFKC/NFKD, stripping accents, etc.  
2. **Pre‑processing** – adding special markers (e.g. `Ġ` for a leading space in GPT‑2) or handling punctuation.  
3. **Vocabulary lookup** – mapping each token to an integer ID (the *vocab*).  
4. **Handling unknowns** – using a fallback token like `<unk>` or breaking a word into smaller sub‑words.  
5. **Post‑processing** – adding start‑/end‑of‑sentence tokens, padding, truncation.

A good tokenizer is **fast**, **deterministic**, and **robust** to the many quirks of human language (emoji, diacritics, mixed scripts, etc.).

---

## 2️⃣  How does **Byte‑Pair Encoding (BPE)** work?

BPE is a **sub‑word tokenisation algorithm** originally invented for data compression (Gage, 1994) and later adapted for NLP (Sennrich et al., 2015).  
The idea is simple:

1. **Start** with a vocabulary that contains every *character* (or byte) that appears in the training corpus.  
2. **Count** all adjacent symbol pairs (e.g. `("a","b")`, `("b","c")`, …) across the whole corpus.  
3. **Pick** the most frequent pair and **merge** it into a new symbol (e.g. `ab`).  
4. **Add** the new symbol to the vocabulary and repeat steps 2‑3 **N** times (or until a target vocab size is reached).

After training, any word can be **greedily** segmented into the longest possible symbols from the learned vocabulary. This yields a compact, language‑agnostic set of sub‑words that can represent rare or out‑of‑vocab words by breaking them into known pieces.

### Visual example

| Iteration | Most frequent pair | Merge → New Symbol | Example word “lowest” after merge |
|-----------|-------------------|-------------------|-----------------------------------|
| 0 (init)  | –                 | –                 | `l o w e s t`                     |
| 1         | (`l`,`o`)         | `lo`              | `lo w e s t`                      |
| 2         | (`e`,`s`)         | `es`              | `lo w es t`                       |
| 3         | (`w`,`es`)        | `wes`             | `lo wes t`                        |
| 4         | (`lo`,`wes`)      | `lowes`           | `lowes t`                         |
| 5         | (`lowes`,`t`)     | `lowest`          | `lowest` (now a single token)    |

The final vocab might contain `["l","o","w","e","s","t","lo","es","wes","lowes","lowest", …]`.

---

## 3️⃣  School‑book (educational) implementation

Below is a **minimal, pure‑Python** implementation that:

* builds a BPE vocab from a list of training sentences,
* merges the most frequent pair `num_merges` times,
* provides `encode` (segment a word) and `decode` (re‑assemble tokens).

> **Note** – This code is intentionally simple for learning purposes.  
> It does **not** handle Unicode normalisation, special tokens, or large corpora efficiently.

```python
# --------------------------------------------------------------
# 1️⃣  Helper utilities
# --------------------------------------------------------------
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

def get_initial_vocab(corpus: List[str]) -> Tuple[Dict[Tuple[str, ...], int], List[List[Tuple[str, ...]]]]:
    """
    Turn each word into a tuple of characters + a special end‑of‑word marker.
    Returns:
        vocab   – mapping from symbol tuple -> frequency (initially char counts)
        tokenized_corpus – list of words represented as list of symbol tuples
    """
    tokenized = []
    vocab = Counter()
    for line in corpus:
        for word in line.strip().split():
            # Append </w> to mark word boundary (standard BPE practice)
            symbols = tuple(word) + ("</w>",)
            tokenized.append(list(symbols))
            vocab.update([symbols])
    return vocab, tokenized

def get_pair_frequencies(tokenized: List[List[Tuple[str, ...]]]) -> Counter:
    """Count all adjacent symbol pairs in the current tokenisation."""
    pairs = Counter()
    for word in tokenized:
        # word is a list of symbols (each symbol is a tuple of chars)
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs

def merge_pair(pair: Tuple[Tuple[str, ...], Tuple[str, ...]],
               tokenized: List[List[Tuple[str, ...]]]) -> List[List[Tuple[str, ...]]]:
    """Replace all occurrences of `pair` with its merged symbol."""
    merged = []
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # not needed for list version, kept for clarity

    for word in tokenized:
        i = 0
        new_word = []
        while i < len(word):
            # If the next two symbols match the pair, merge them
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])  # tuple concatenation
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged.append(new_word)
    return merged

# --------------------------------------------------------------
# 2️⃣  BPE trainer
# --------------------------------------------------------------
def train_bpe(corpus: List[str], num_merges: int = 1000) -> Tuple[Dict[str, int], List[Tuple[str, ...]]]:
    """
    Train a BPE tokenizer.
    Returns:
        vocab_dict – mapping token string → id (int)
        merges     – list of merge operations in order (for decoding)
    """
    # 1️⃣ Initialise tokenised corpus (list of symbol tuples)
    _, tokenized = get_initial_vocab(corpus)

    merges = []                     # keep the order of merges for later use
    for i in range(num_merges):
        pair_freqs = get_pair_frequencies(tokenized)
        if not pair_freqs:
            break
        most_frequent = pair_freqs.most_common(1)[0][0]   # tuple of two symbols
        merges.append(most_frequent)

        # 2️⃣ Merge the most frequent pair everywhere
        tokenized = merge_pair(most_frequent, tokenized)

    # Build final vocab (string representation of each symbol)
    vocab = {}
    idx = 0
    for word in tokenized:
        for sym in word:
            token = ''.join(sym)          # e.g. ('l','o') → "lo"
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    # Add the special unknown token
    vocab["<unk>"] = idx
    return vocab, merges

# --------------------------------------------------------------
# 3️⃣  Encoding / Decoding utilities
# --------------------------------------------------------------
def encode_word(word: str, merges: List[Tuple[Tuple[str, ...], Tuple[str, ...]]], vocab: Dict[str, int]) -> List[int]:
    """
    Greedy BPE segmentation of a single word using the learned merges.
    """
    # start with characters + </w>
    symbols = [tuple(c) for c in word] + [("</w>",)]

    # Apply merges in the same order as training
    for merge in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == merge:
                symbols[i] = symbols[i] + symbols[i + 1]   # merge
                del symbols[i + 1]
            else:
                i += 1

    # Convert to string tokens and look up IDs
    ids = []
    for sym in symbols:
        token = ''.join(sym)
        ids.append(vocab.get(token, vocab["<unk>"]))
    return ids

def decode_ids(ids: List[int], vocab: Dict[str, int]) -> str:
    """Simple reverse lookup (ignores </w> marker)."""
    inv_vocab = {i: t for t, i in vocab.items()}
    tokens = [inv_vocab[i] for i in ids if i in inv_vocab]
    # Remove the end‑of‑word marker and concatenate
    text = ''.join(t.replace('</w>', ' ') for t in tokens).strip()
    return text

# --------------------------------------------------------------
# 4️⃣  Quick demo
# --------------------------------------------------------------
if __name__ == "__main__":
    training_corpus = [
        "low lowest lower lowly",
        "new newer newest",
        "wide wider widest",
        "quick quickly",
        "hello world"
    ]

    vocab, merges = train_bpe(training_corpus, num_merges=50)
    print("=== Vocabulary (sample) ===")
    for token, idx in list(vocab.items())[:20]:
        print(idx, "→", token)

    # Encode a new word
    word = "lowest"
    ids = encode_word(word, merges, vocab)
    print("\nEncoded:", word, "→", ids)

    # Decode back
    print("Decoded:", decode_ids(ids, vocab))
```

### What this *school‑book* code teaches you

| Step | What you learn |
|------|----------------|
| **Initialisation** | Represent each word as a list of character tuples plus a special end‑of‑word marker (`</w>`). |
| **Pair counting** | How to count adjacent symbol pairs across the whole corpus. |
| **Merging** | In‑place replacement of the most frequent pair, building longer sub‑words. |
| **Vocabulary construction** | Turning merged tuples into string tokens and assigning integer IDs. |
| **Greedy segmentation** | Re‑applying the same merge order to unseen words. |
| **Decoding** | Simple reverse lookup (useful for debugging). |

> **Caveats** – This implementation is *O(N · M)* where *N* is the number of merges and *M* the corpus size, and it stores the whole corpus in memory. Real‑world tokenizers need a more efficient data structure (e.g. a trie or a hash‑map of pair frequencies) and careful Unicode handling.

---

## 4️⃣  Production‑ready BPE tokenizer

When you move from a teaching demo to a **production system**, you typically want:

| Requirement | Why it matters |
|-------------|----------------|
| **Speed** – tokenisation must be sub‑millisecond per sentence (often millions of tokens per second). |
| **Memory efficiency** – the vocab (often 30‑50 k entries) should be stored compactly. |
| **Robust Unicode handling** – NFC/NFKC normalisation, emoji, CJK characters, etc. |
| **Deterministic & reproducible** – same input → same token IDs across machines. |
| **Integration with ML frameworks** – easy export to ONNX, TensorFlow, PyTorch, etc. |
| **Support for special tokens** – `<pad>`, `<bos>`, `<eos>`, `<unk>`, etc. |
| **Thread‑safety** – safe to call from many workers (e.g. in a web service). |

The **de‑facto** production solution in the Python ecosystem is the **🤗 Hugging Face `tokenizers` library** (Rust core, Python bindings). It provides a fast BPE implementation, a clean API, and can be exported to a binary file (`vocab.json` + `merges.txt`) that can be loaded by any framework.

Below is a step‑by‑step guide to build a production‑ready BPE tokenizer using that library.

### 4.1  Install the library

```bash
pip install tokenizers   # includes the fast Rust implementation
```

### 4.2  Train a BPE tokenizer on your own data

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# 1️⃣  Choose the model (BPE)
bpe_model = models.BPE()

# 2️⃣  Define normalisation & pre‑tokenisation
normalizer = normalizers.Sequence([NFKC(), Lowercase(), Strip()])
pre_tokenizer = Whitespace()                     # split on whitespace, keep punctuation as separate tokens

# 3️⃣  Build the tokenizer object
tokenizer = Tokenizer(bpe_model)
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

# 4️⃣  Trainer – you can control vocab size, min frequency, special tokens, etc.
trainer = trainers.BpeTrainer(
    vocab_size=30_000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    show_progress=True
)

# 5️⃣  Train on a list of files (or a list of strings)
files = ["data/train.txt"]          # each line = a raw sentence
tokenizer.train(files, trainer)

# 6️⃣  Post‑processing – add BOS/EOS automatically
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# 7️⃣  Save for later reuse (fast loading)
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer saved – vocab size:", tokenizer.get_vocab_size())
```

**What makes this production‑ready?**

| Feature | Implementation |
|---------|----------------|
| **Fast Rust core** | All tokenisation steps (normalisation, pre‑tokenisation, BPE merges) run in compiled Rust → ~10‑30× faster than pure Python. |
| **Deterministic** | The same `vocab.json` + `merges.txt` (or the single `.json` above) always yields identical IDs. |
| **Unicode‑aware** | `NFKC` normalisation handles composed/decomposed characters; the library works on UTF‑8 byte strings directly. |
| **Special tokens** | `<pad>`, `<s>`, `</s>`, `<unk>`, `<mask>` are baked into the vocab and can be referenced by ID. |
| **Thread‑safe** | The tokenizer object can be shared across processes (or re‑loaded per worker). |
| **Exportable** | The JSON file can be loaded by 🤗 Transformers, ONNX Runtime, or even the Rust `tokenizers` crate in other languages (C++, Java, etc.). |

### 4.3  Using the tokenizer in inference pipelines

```python
from tokenizers import Tokenizer

# Load the saved tokenizer (fast, < 10 ms even for large vocab)
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Encode a single sentence
sentence = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer.encode(sentence)

print("Token IDs :", encoding.ids)
print("Tokens    :", encoding.tokens)

# Decode back (useful for debugging)
print("Decoded   :", tokenizer.decode(encoding.ids))
```

### 4.4  Integration with 🤗 Transformers (PyTorch / TensorFlow)

If you are already using the `transformers` library, you can directly load the same tokenizer:

```python
from transformers import PreTrainedTokenizerFast

# The file saved by `tokenizer.save` is compatible with this class
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json",
                                    unk_token="<unk>",
                                    pad_token="<pad>",
                                    bos_token="<s>",
                                    eos_token="</s>")

# Example: tokenising a batch
batch = ["Hello world!", "Byte‑pair encoding is cool."]
enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
print(enc["input_ids"])
print(enc["attention_mask"])
```

Now you can feed `enc["input_ids"]` straight into any `transformers` model (e.g., `GPT2Model`, `BertModel`, etc.) without extra preprocessing.

### 4.5  Production deployment tips

| Tip | Reason |
|-----|--------|
| **Cache the tokenizer** – Load it once at service start‑up, keep the instance in memory. |
| **Avoid per‑request file I/O** – The `.json` file is read only once; subsequent calls are pure in‑memory operations. |
| **Batch tokenisation** – If your API receives many sentences at once, call `tokenizer.batch_encode_plus` (or `tokenizer.encode_batch`) to amortise overhead. |
| **GPU‑friendly tensors** – Return `torch.LongTensor` directly (`return_tensors="pt"`). |
| **Version pinning** – Store the tokenizer file alongside your model checkpoint; this guarantees reproducibility. |
| **Monitoring** – Log the average latency of `tokenizer.encode` in production; if it spikes, check for Unicode edge‑cases or unusually long inputs. |
| **Security** – The tokenizer does not execute code, but always validate input size (e.g., max characters) to avoid denial‑of‑service attacks. |

---

## 5️⃣  TL;DR Summary

| Concept | Key points |
|---------|------------|
| **Tokenizer** | Turns raw text → token IDs. Handles normalisation, special tokens, unknowns, and can work at word, sub‑word, or byte level. |
| **BPE** | Starts with characters, repeatedly merges the most frequent adjacent pair. Produces a compact sub‑word vocab that can represent any word by greedy segmentation. |
| **School‑book implementation** | Simple Python code that builds a vocab, merges pairs, and encodes/decodes words. Great for learning but not scalable. |
| **Production‑ready version** | Use the `tokenizers` library (Rust core) → fast, Unicode‑aware, thread‑safe, supports special tokens, can be saved/loaded, and integrates with Hugging Face Transformers. |


---

Below is a **minimal, human‑readable example** of the JSON file that the 🤗 `tokenizers` library writes when you call `tokenizer.save("my_bpe.json")`.  
The file contains **four top‑level sections**:

| Section | What it stores |
|---------|----------------|
| `model` | The actual BPE model – a mapping **token → id** (`vocab`) and the ordered list of **merge operations** (`merges`). |
| `normalizer` | How raw Unicode text is normalised before tokenisation (e.g. NFKC, lower‑casing, stripping). |
| `pre_tokenizer` | The first split step (usually whitespace‑based, but can be more complex). |
| `post_processor` | How special tokens like BOS/EOS are added after the BPE step. |

> **Tip:** Only the `model` part is strictly required for a “vocabulary JSON”.  
> The other sections are optional – if you omit them the tokenizer will fall back to defaults.

---

## 1️⃣  Full‑featured example (≈ 30 k‑style vocab)

```json
{
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3,
      "<mask>": 4,
      "the": 5,
      "▁and": 6,
      "▁to": 7,
      "▁of": 8,
      "▁a": 9,
      "▁in": 10,
      "▁that": 11,
      "▁is": 12,
      "▁it": 13,
      "▁he": 14,
      "▁she": 15,
      "▁was": 16,
      "▁for": 17,
      "▁on": 18,
      "▁with": 19,
      "▁as": 20,
      "▁his": 21,
      "▁her": 22,
      "▁i": 23,
      "▁you": 24,
      "▁we": 25,
      "▁they": 26,
      "▁be": 27,
      "▁at": 28,
      "▁by": 29,
      "▁not": 30,
      "▁from": 31,
      "▁this": 32,
      "▁but": 33,
      "▁or": 34,
      "▁have": 35,
      "▁had": 36,
      "▁were": 37,
      "▁which": 38,
      "▁one": 39,
      "▁all": 40,
      "▁their": 41,
      "▁there": 42,
      "▁when": 43,
      "▁who": 44,
      "▁what": 45,
      "▁so": 46,
      "▁can": 47,
      "▁if": 48,
      "▁would": 49,
      "▁do": 50,
      "▁said": 51,
      "▁about": 52,
      "▁out": 53,
      "▁up": 54,
      "▁more": 55,
      "▁than": 56,
      "▁some": 57,
      "▁into": 58,
      "▁no": 59,
      "▁time": 60,
      "▁just": 61,
      "▁him": 62,
      "▁her": 63,
      "▁my": 64,
      "▁your": 65,
      "▁our": 66,
      "▁their": 67,
      "▁good": 68,
      "▁new": 69,
      "▁first": 70,
      "▁last": 71,
      "▁great": 72,
      "▁little": 73,
      "▁big": 74,
      "▁small": 75,
      "▁old": 76,
      "▁young": 77,
      "▁high": 78,
      "▁low": 79,
      "▁long": 80,
      "▁short": 81,
      "▁right": 82,
      "▁left": 83,
      "▁up": 84,
      "▁down": 85,
      "▁here": 86,
      "▁there": 87,
      "▁where": 88,
      "▁why": 89,
      "▁how": 90,
      "▁because": 91,
      "▁while": 92,
      "▁after": 93,
      "▁before": 94,
      "▁again": 95,
      "▁once": 96,
      "▁twice": 97,
      "▁three": 98,
      "▁four": 99,
      "▁five": 100,
      "▁six": 101,
      "▁seven": 102,
      "▁eight": 103,
      "▁nine": 104,
      "▁ten": 105,
      "▁hundred": 106,
      "▁thousand": 107,
      "▁million": 108,
      "▁billion": 109,
      "▁percent": 110,
      "▁$": 111,
      "▁,": 112,
      "▁.": 113,
      "▁!": 114,
      "▁?": 115,
      "▁'": 116,
      "▁\"": 117,
      "▁(": 118,
      "▁)": 119,
      "▁-": 120,
      "▁/": 121,
      "▁\\": 122,
      "▁:": 123,
      "▁;": 124,
      "▁…": 125,
      "▁😀": 126,
      "▁🚀": 127,
      "▁❤️": 128,
      "▁##": 129,
      "▁##ing": 130,
      "▁##ed": 131,
      "▁##ly": 132,
      "▁##s": 133,
      "▁##tion": 134,
      "▁##ness": 135,
      "▁##able": 136,
      "▁##ment": 137,
      "▁##ist": 138,
      "▁##er": 139,
      "▁##est": 140,
      "▁##ous": 141,
      "▁##ive": 142,
      "▁##ify": 143,
      "▁##ize": 144,
      "▁##al": 145,
      "▁##ic": 146,
      "▁##ify": 147,
      "▁##tion": 148,
      "▁##ness": 149,
      "▁##hood": 150,
      "▁##ship": 151,
      "▁##less": 152,
      "▁##ful": 153,
      "▁##ward": 154,
      "▁##wise": 155,
      "▁##like": 156,
      "▁##ish": 157,
      "▁##y": 158,
      "▁##en": 159,
      "▁##ify": 160,
      "▁##ate": 161,
      "▁##ify": 162,
      "▁##ise": 163,
      "▁##ise": 164,
      "▁##ise": 165,
      "▁##ise": 166,
      "▁##ise": 167,
      "▁##ise": 168,
      "▁##ise": 169,
      "▁##ise": 170,
      "▁##ise": 171,
      "▁##ise": 172,
      "▁##ise": 173,
      "▁##ise": 174,
      "▁##ise": 175,
      "▁##ise": 176,
      "▁##ise": 177,
      "▁##ise": 178,
      "▁##ise": 179,
      "▁##ise": 180,
      "▁##ise": 181,
      "▁##ise": 182,
      "▁##ise": 183,
      "▁##ise": 184,
      "▁##ise": 185,
      "▁##ise": 186,
      "▁##ise": 187,
      "▁##ise": 188,
      "▁##ise": 189,
      "▁##ise": 190,
      "▁##ise": 191,
      "▁##ise": 192,
      "▁##ise": 193,
      "▁##ise": 194,
      "▁##ise": 195,
      "▁##ise": 196,
      "▁##ise": 197,
      "▁##ise": 198,
      "▁##ise": 199,
      "▁##ise": 200
    },
    "merges": [
      ["▁t", "he"],
      ["▁a", "nd"],
      ["▁t", "o"],
      ["▁o", "f"],
      ["▁i", "n"],
      ["▁w", "as"],
      ["▁h", "e"],
      ["▁s", "he"],
      ["▁h", "er"],
      ["▁i", "t"],
      ["▁b", "e"],
      ["▁a", "t"],
      ["▁f", "or"],
      ["▁o", "n"],
      ["▁w", "ith"],
      ["▁a", "s"],
      ["▁h", "is"],
      ["▁h", "er"],
      ["▁y", "ou"],
      ["▁w", "e"],
      ["▁t", "hey"],
      ["▁b", "ut"],
      ["▁o", "r"],
      ["▁h", "ave"],
      ["▁h", "ad"],
      ["▁w", "ere"],
      ["▁w", "hich"],
      ["▁o", "ne"],
      ["▁a", "ll"],
      ["▁t", "heir"],
      ["▁t", "here"],
      ["▁w", "hen"],
      ["▁w", "ho"],
      ["▁w", "hat"],
      ["▁s", "o"],
      ["▁c", "an"],
      ["▁i", "f"],
      ["▁w", "ould"],
      ["▁d", "o"],
      ["▁s", "aid"],
      ["▁a", "bout"],
      ["▁o", "ut"],
      ["▁u", "p"],
      ["▁m", "ore"],
      ["▁t", "han"],
      ["▁s", "ome"],
      ["▁i", "nto"],
      ["▁n", "o"],
      ["▁t", "ime"],
      ["▁j", "ust"],
      ["▁h", "im"],
      ["▁h", "er"],
      ["▁m", "y"],
      ["▁y", "our"],
      ["▁o", "ur"],
      ["▁g", "ood"],
      ["▁n", "ew"],
      ["▁f", "irst"],
      ["▁l", "ast"],
      ["▁g", "reat"],
      ["▁l", "ittle"],
      ["▁b", "ig"],
      ["▁s", "mall"],
      ["▁o", "ld"],
      ["▁y", "oung"],
      ["▁h", "igh"],
      ["▁l", "ow"],
      ["▁l", "ong"],
      ["▁s", "hort"],
      ["▁r", "ight"],
      ["▁l", "eft"],
      ["▁h", "ere"],
      ["▁t", "here"],
      ["▁w", "here"],
      ["▁w", "hy"],
      ["▁h", "ow"],
      ["▁b", "ecause"],
      ["▁w", "hile"],
      ["▁a", "fter"],
      ["▁b", "efore"],
      ["▁a", "gain"],
      ["▁o", "nce"],
      ["▁t", "wice"],
      ["▁t", "hree"],
      ["▁f", "our"],
      ["▁f", "ive"],
      ["▁s", "ix"],
      ["▁s", "even"],
      ["▁e", "ight"],
      ["▁n", "ine"],
      ["▁t", "en"],
      ["▁h", "undred"],
      ["▁t", "housand"],
      ["▁m", "illion"],
      ["▁b", "illion"],
      ["▁p", "ercent"],
      ["▁$", "$"],
      ["▁,", ","],
      ["▁.", "."],
      ["▁!", "!"],
      ["▁?", "?"],
      ["▁'", "'"],
      ["▁\"", "\""],
      ["▁(", "("],
      ["▁)", ")"],
      ["▁-", "-"],
      ["▁/", "/"],
      ["▁\\", "\\"],
      ["▁:", ":"],
      ["▁;", ";"],
      ["▁…", "…"],
      ["▁😀", "😀"],
      ["▁🚀", "🚀"],
      ["▁❤️", "❤️"],
      ["▁##", "##"],
      ["▁##", "ing"],
      ["▁##", "ed"],
      ["▁##", "ly"],
      ["▁##", "s"],
      ["▁##", "tion"],
      ["▁##", "ness"],
      ["▁##", "able"],
      ["▁##", "ment"],
      ["▁##", "ist"],
      ["▁##", "er"],
      ["▁##", "est"],
      ["▁##", "ous"],
      ["▁##", "ive"],
      ["▁##", "ify"],
      ["▁##", "ize"],
      ["▁##", "al"],
      ["▁##", "ic"],
      ["▁##", "hood"],
      ["▁##", "ship"],
      ["▁##", "less"],
      ["▁##", "ful"],
      ["▁##", "ward"],
      ["▁##", "wise"],
      ["▁##", "like"],
      ["▁##", "ish"],
      ["▁##", "y"],
      ["▁##", "en"],
      ["▁##", "ate"]
    ]
  },

  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      { "type": "NFKC" },
      { "type": "Lowercase" },
      { "type": "Strip" }
    ]
  },

  "pre_tokenizer": {
    "type": "Whitespace"
  },

  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      "<s>", "$A", "</s>"
    ],
    "pair": [
      "<s>", "$A", "$B", "</s>"
    ],
    "special_tokens": [
      { "id": 1, "type_id": 0, "token": "<s>" },
      { "id": 2, "type_id": 0, "token": "</s>" }
    ]
  }
}
```

### What you see in the file

| Part | Example entry | Meaning |
|------|---------------|---------|
| **`vocab`** | `"▁and": 6` | Token string → integer ID. The leading `▁` (U+2581, “lower one eighth block”) is the **space marker** that GPT‑2‑style byte‑level BPE uses to indicate a preceding whitespace. |
| **`merges`** | `["▁t","he"]` | The first merge operation learned during training. The list order is **exactly** the order the tokenizer will apply when encoding new text. |
| **`normalizer`** | `{"type":"Lowercase"}` | Convert everything to lower case before any splitting. |
| **`pre_tokenizer`** | `{"type":"Whitespace"}` | Split the raw string on whitespace; punctuation stays attached to the preceding token and will later be broken by BPE merges. |
| **`post_processor`** | `"<s>", "$A", "</s>"` | After BPE, prepend BOS (`<s>`) and append EOS (`</s>`) automatically. `$A` is the placeholder for the token list produced by the BPE step. |

---

## 2️⃣  Tiny “toy” vocab JSON (easy to read)

If you just want to see the **bare minimum** – a mapping of a handful of tokens and a couple of merges – here’s a stripped‑down version that you could even write by hand:

```json
{
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3,
      "the": 4,
      "▁low": 5,
      "est": 6,
      "▁lowest": 7,
      "▁new": 8,
      "er": 9,
      "est</w>": 10,
      "▁newer": 11,
      "▁newest": 12,
      "▁quick": 13,
      "ly": 14,
      "▁quickly": 15,
      "▁hello": 16,
      "▁world": 17,
      "!</w>": 18
    },
    "merges": [
      ["▁", "low"],
      ["low", "est"],
      ["▁", "new"],
      ["new", "er"],
      ["new", "est"],
      ["▁", "quick"],
      ["quick", "ly"],
      ["▁", "hello"],
      ["▁", "world"],
      ["world", "!"]
    ]
  },

  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      { "type": "NFKC" },
      { "type": "Lowercase" }
    ]
  },

  "pre_tokenizer": { "type": "Whitespace" },

  "post_processor": {
    "type": "TemplateProcessing",
    "single": ["<s>", "$A", "</s>"],
    "pair": ["<s>", "$A", "$B", "</s>"],
    "special_tokens": [
      { "id": 1, "type_id": 0, "token": "<s>" },
      { "id": 2, "type_id": 0, "token": "</s>" }
    ]
  }
}
```

*Running this tiny tokenizer on the sentence*  

```text
"The lowest new quickly hello world!"
```  

*produces the following IDs* (using the IDs from the `vocab` above):

```
[1, 4, 7, 11, 15, 16, 17, 18, 2]
#   ^   ^   ^   ^   ^   ^   ^   ^   ^
# <s> the lowest newer quickly hello world! </s>
```

---

## 3️⃣  How to load the JSON in code

```python
from tokenizers import Tokenizer

# Load the file you just saved (fast Rust implementation)
tokenizer = Tokenizer.from_file("my_bpe.json")

# Encode a batch (the post‑processor automatically adds BOS/EOS)
enc = tokenizer.encode("The lowest new quickly hello world!")
print("ids :", enc.ids)          # → [1, 4, 7, 11, 15, 16, 17, 18, 2]
print("tokens :", enc.tokens)    # → ['<s>', 'the', '▁lowest', '▁new', '▁quickly', '▁hello', '▁world', '!</w>', '</s>']
```

That’s it – the JSON file is the **single source of truth** for both training and inference.  
When you ship a model, just bundle `my_bpe.json` together with the checkpoint; any environment that can import `tokenizers` (Python, Rust, C++, Java, Node, …) will be able to reproduce exactly the same token IDs.

---

### TL;DR

* The **vocabulary JSON** contains a `model` block with a `vocab` dictionary (`token → id`) and a `merges` list (ordered BPE merges).  
* It also stores optional `normalizer`, `pre_tokenizer`, and `post_processor` sections so the whole tokenisation pipeline can be reconstructed from a single file.  
* A tiny example (≈ 20 tokens) is shown above, and a more realistic “30 k‑style” snippet demonstrates the exact structure you’ll see in production.  

Feel free to copy the tiny example, edit the token strings, and run `Tokenizer.from_file` to see it in action!


---

A key optimization that separates a simple educational implementation from a high-performance, production-ready one is optimizing the use of merges.

Using a Trie (or a similar optimized structure) is crucial for making the **encoding** step fast and efficient.

Let's break down where a Trie would fit and what its benefits are.

### The Core Problem with the Naive `encode` Function

First, let's look at the bottleneck in the school-book `encode_word` function:

```python
def encode_word(word: str, merges: List[...], vocab: Dict[...]) -> List[int]:
    # ...
    # Apply merges in the same order as training
    for merge in merges:  # This loop is the problem!
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == merge:
                # ... merge and delete ...
            else:
                i += 1
    # ...
```

For every single word we want to encode, this function iterates through the **entire list of merge rules** (which could be thousands long). If `num_merges` is 30,000, this is a huge amount of repeated work.

This is where a Trie provides a much more elegant and performant solution.

### How a Trie Solves the Encoding Problem

Instead of re-applying the merge rules, we can use the **final vocabulary** to build a Trie. Each path from the root to a node in the Trie represents a valid token.

1.  **Build the Vocabulary Trie:**
    Take all the tokens from your final BPE vocabulary (e.g., `l`, `o`, `w`, `lo`, `es`, `wes`, `lowes`, `lowest`, `</w>`) and insert them into a Trie.

    A simplified view of the Trie would look like this:

    ```
         (root)
         /  |  \
        l   w   <
       /   / \   \
      o   e   i   /
     /   / \   \   \
    w   s   d   w   >
     \   \   \   \
      e   t   e   >
       \       \
        s       r
         \
          t
    ```
    *(Each path from the root, like `l`->`o`->`w`->`e`->`s`->`t`, represents a token: "lowest")*

2.  **Greedy Longest-Match Encoding:**
    Now, to tokenize a new word like `"lowest</w>"`, you perform a **greedy longest-prefix match** against this Trie.

    *   **Start at index 0 (`l`).** Traverse the Trie: `l` is a token. `lo` is a token. `low` is not. `lowe` is not. `lowes` is a token. `lowest` is a token. `lowest<` is not.
    *   The longest possible token starting at index 0 is `"lowest"`.
    *   **Result:** Emit the token `"lowest"`. Advance your position in the word by `len("lowest")`.
    *   **Continue from the new position.** In this case, we have `"</w>"` left. The longest match is `"</w>"`.
    *   **Result:** Emit the token `"</w>"`.
    *   **Final tokens:** `["lowest", "</w>"]`.

This process requires only a **single pass** over the input word, making it dramatically faster than the naive loop-over-merges approach.

---

### What about the **Training** Step?

Using a Trie for the training part is less common and more complex. The main bottleneck during training is re-calculating pair frequencies (`get_pair_frequencies`) after every merge. Production tokenizers solve this with a different optimization:

*   Instead of re-scanning the whole corpus, they maintain an index of where each pair occurs.
*   When a pair `(A, B)` is merged into `C`, they only need to update the counts for the pairs immediately adjacent to the merge points. For example, if you had `... X A B Y ...`, you would:
    *   Decrement the count of `(X, A)` and `(B, Y)`.
    *   Increment the count of `(X, C)` and `(C, Y)`.
*   This local update is much faster than a global rescan. This is often managed with a combination of linked lists (to represent the token sequences) and a priority queue (to store the pair frequencies and quickly find the max).

So, the optimizations are typically split:
1.  **Training:** Efficient pair counting using indexed/linked structures and a priority queue.
2.  **Encoding:** A Trie built from the final vocabulary for fast longest-match segmentation.

### Summary: Trie Benefits vs. School-Book Hash Map

| Aspect | School-Book (Hash Map / Dict) | Production (Trie-based Encoding) |
| :--- | :--- | :--- |
| **Data Structure** | `merges`: A `List` of merge rules.<br>`vocab`: A `Dict` mapping string -> ID. | `vocab`: A **Trie** where each path is a valid token. |
| **Encoding Algorithm** | **Re-apply all merge rules** sequentially for each new word. | **Greedy longest-prefix match** against the Trie in a single pass over the word. |
| **Time Complexity (Encoding)**| *O(len(word) * num_merges)* | *O(len(word))*, as Trie lookups are proportional to token length, not vocabulary size. |
| **Benefit** | **Simple to understand and implement.** Clearly demonstrates the BPE merge logic. | **Extremely fast.** This is the method used in libraries like `sentencepiece` and Hugging Face `tokenizers`. |
| **Drawback** | **Very slow.** Unsuitable for any real-world application. | **More complex to implement.** The Trie data structure itself is more involved than a simple list or dictionary. |

In conclusion, your intuition is spot on. The move from a hash-map-based "replay the merges" strategy to a **Trie-based longest-match strategy** is a fundamental step in building a tokenizer that is not just correct, but also performant enough for production systems.