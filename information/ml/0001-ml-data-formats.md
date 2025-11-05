# Fromats for machine learning data


**Key data‑storage formats used in machine learning projects**

| Type | Typical use | Main advantages |
|------|-------------|-----------------|
| **CSV** | Small‑to‑medium tabular data, quick inspection | Human readable, universal support |
| **TSV / TSV‑GZ** | Same as CSV but tab‑delimited, easier when commas appear in fields | Simple, compressible |
| **JSON / JSON‑L** | Hierarchical or nested records, logs, configuration | Human readable, easy to parse in Python/JS |
| **Parquet** | Large column‑oriented tables, analytics pipelines | Columnar compression, fast column pruning, works with Spark, Dask, Pandas |
| **Feather / Arrow IPC** | Fast interchange between Python/R/Julia, in‑memory analytics | Zero‑copy reads, columnar, supports complex types |
| **HDF5** | Multi‑dimensional arrays, hierarchical datasets, scientific data | Random access, compression, good for images, time‑series |
| **TFRecord** | TensorFlow pipelines, large image/audio/video streams | Streamable, works with tf.data, supports sharding |
| **RecordIO / LMDB** | General purpose binary record storage, often used in Caffe, MXNet | Fast sequential reads, good for large image datasets |
| **NPY / NPZ** | Numpy arrays, single‑ or multi‑array files | Native to Python, very fast load |
| **Pickle / Joblib** | Python objects (pre‑processed features, pipelines) | Simple, but not cross‑language and can be insecure |
| **Image formats (JPEG, PNG, WebP, TIFF)** | Raw image data for CV tasks | Widely supported, compression options |
| **Audio formats (WAV, FLAC, OGG)** | Raw audio for speech / audio tasks | Lossless (WAV, FLAC) or compressed (OGG) |
| **Video formats (MP4, MKV, WebM)** | Video streams for action recognition | Standard codecs, can be read frame‑wise |
| **ONNX / PMML / TorchScript** | Serialized trained models (often considered “data” for deployment) | Interoperable inference across frameworks |

**When to choose which format**

* **Exploratory work / small datasets** – CSV, TSV, JSON.  
* **Large tabular data** – Parquet or Feather for columnar speed and compression.  
* **High‑dimensional arrays (images, sensor data)** – HDF5, NPY/NPZ, TFRecord.  
* **Streaming pipelines** – TFRecord, RecordIO, LMDB.  
* **Cross‑language or production deployment** – ONNX, PMML.  

Pick the format that matches the data shape (tabular vs. array), size, required compression, and the tools you plan to use.

These tools cover most needs: raw tensors, tabular datasets, versioning, and integration with big‑data processing frameworks.


Below is a **deep‑dive** into each of the most common data‑storage formats you’ll encounter in machine‑learning (ML) projects.  
For every format you’ll find:

* **What it is / typical use‑case**  
* **Key strengths** – why you would pick it  
* **Weak points / gotchas**  
* **Scalability & performance** – how it behaves on large data, parallel / distributed workloads  
* **Concurrency & consistency** – read‑/write‑access patterns and multi‑process safety  
* **Ecosystem & tooling** – libraries, languages, and platforms that understand it  

I’ve also added a **quick‑compare matrix** at the end and a short **decision guide** for picking the right format for a given workflow.

---

## 1. Tabular‑oriented Formats

| Format | Typical Use | Strengths | Weaknesses | Scalability & Performance | Concurrency | Ecosystem |
|--------|-------------|-----------|------------|---------------------------|-------------|-----------|
| **CSV** (comma‑separated values) | Small‑to‑medium tabular data, quick ad‑hoc inspection | Human‑readable, ubiquitous, works with any spreadsheet or text editor | No schema, ambiguous quoting, no native compression, no random access, poor handling of nested data | Reads/writes are **O(N)** linear; can be gzipped (`*.csv.gz`) for modest size reduction but still streamed line‑by‑line. | Single‑writer, many‑readers (file lock needed for safe concurrent writes). | `pandas.read_csv`, Python `csv`, R `readr`, Excel, many DB import tools. |
| **TSV / TSV‑GZ** | Same as CSV but tab‑delimited (useful when commas appear in fields) | Same as CSV + easier parsing when commas are common | Same limitations as CSV; still no schema | Same as CSV; gzip compression (`*.tsv.gz`) reduces I/O dramatically for large files. | Same as CSV | Same as CSV |
| **JSON** (JavaScript Object Notation) | Hierarchical / nested records, logs, configuration files | Human‑readable, supports nesting, language‑agnostic, easy to stream line‑by‑line (`jsonlines`) | Verbose, no built‑in schema, no columnar compression, whole‑file must be parsed for random access | **O(N)** parsing; can be gzipped (`*.json.gz`). For very large logs, use **JSON‑L** (one JSON object per line) to enable streaming. | Single‑writer, many‑readers; safe with file locks. | `json` module (Python), `orjson`, `rapidjson`, `jq`, many NoSQL DBs (MongoDB). |
| **JSON‑L** (JSON Lines) | Same as JSON but each line is a self‑contained JSON object | Streamable, easy to append, works with line‑based tools (`grep`, `awk`) | Still verbose, no columnar compression | Linear read/write; can be gzipped (`*.jsonl.gz`). | Append‑friendly – multiple processes can safely write if they acquire a lock or use a log‑aggregation service. | Same as JSON + `pandas.read_json(..., lines=True)`. |
| **Parquet** | Large column‑oriented tables, analytics pipelines, data‑lake workloads | Columnar storage → **fast column pruning**, built‑in **dictionary & run‑length compression**, **schema** (metadata) stored in file, **splittable** for parallel reads, works with Spark, Dask, Presto, Hive, Athena, etc. | Not human‑readable, requires libraries to inspect, limited support for deeply nested structures (though it does support complex types). | **Scalable to billions of rows**. Files can be **partitioned** (folder hierarchy) and **row‑group** size tuned (default ~128 MiB) to balance memory vs. I/O. Reads are **vectorized** and can be parallelized across row‑groups. | **Read‑only** concurrency is trivial (multiple processes can read same file). **Write** concurrency is limited – you cannot safely have many writers to the same file; instead write separate files (e.g., per partition) and later merge. | `pyarrow.parquet`, `fastparquet`, Spark `DataFrame.write.parquet`, Pandas `to_parquet`, Hive/Impala, AWS Athena, Google BigQuery external tables. |
| **Feather / Arrow IPC** | Fast interchange between Python/R/Julia, in‑memory analytics, small‑to‑medium columnar data | **Zero‑copy** reads/writes using Apache Arrow memory format → **sub‑millisecond** load for millions of rows, supports complex nested types, cross‑language. | Not ideal for very large datasets (>10 GB) because the whole file is usually read into memory; limited compression (LZ4/ZSTD) but still columnar. | Works well for **single‑node** workloads; can be sharded across many files for larger data. | Multiple readers are safe; concurrent writes require separate files. | `pyarrow.feather`, `pandas.read_feather`, R `arrow::read_feather`. |
| **Delta Lake** (open‑source) | ACID‑compliant data lake on top of Parquet | **Transactional** (commit/rollback), **schema evolution**, **time‑travel** (versioned reads), **optimistic concurrency control**, **vacuum** for cleanup. | Requires a Delta‑compatible engine (Spark, Databricks, Delta‑Rust, Delta‑Python). | Same columnar performance as Parquet + added **metadata log** for fast snapshot reads. Scales to petabytes when stored on S3/ADLS. | **Concurrent writers** are supported via **optimistic concurrency** – each writer creates a new transaction log entry; conflicts are detected and retried. | `delta-rs`, `pyspark.sql`, `deltalake` Python package. |
| **Apache Iceberg** | Similar to Delta Lake – table format for large analytic datasets | **Schema evolution**, **partition spec evolution**, **snapshot isolation**, **hidden partitions**, **metadata pruning**. | Still maturing in the Python ecosystem (though growing). | Same as Parquet + metadata tables for fast planning. | Supports **concurrent writes** via **append‑only** and **overwrite** modes with conflict detection. | Spark, Flink, Trino, Presto, `iceberg-python`. |

---

## 2. Array‑Oriented / Tensor Formats

| Format | Typical Use | Strengths | Weaknesses | Scalability & Performance | Concurrency | Ecosystem |
|--------|-------------|-----------|------------|---------------------------|-------------|-----------|
| **NPY / NPZ** | Single‑ or multi‑array files, quick dump of NumPy arrays | Native NumPy format → **instant load**, supports **memory‑mapping** (`np.load(..., mmap_mode='r')`) for out‑of‑core access, `npz` bundles multiple arrays with zip compression. | Not columnar, limited to NumPy‑compatible dtypes, no schema beyond dtype/shape, not splittable for parallel reads. | Good for **medium‑size** arrays (up to a few GB). Memory‑mapping enables **random access** without loading whole file. | Read‑only concurrency is safe; writes must be exclusive. | NumPy, `np.save`, `np.load`. |
| **HDF5** (Hierarchical Data Format v5) | Multi‑dimensional scientific data, images, time‑series, hierarchical groups | **Hierarchical namespace** (datasets inside groups), **chunked storage** → random access, **built‑in compression** (gzip, LZF, szip), **partial I/O**, **metadata** attributes. | Complex API, file locking required for safe concurrent writes, single‑writer‑multiple‑reader (SWMR) mode only in newer HDF5 (≥1.10). | Scales to **hundreds of GB** in a single file; can be sharded across many files for petabyte‑scale. Chunk size tuning critical for performance. | **SWMR** (single‑writer‑multiple‑reader) mode enables concurrent reads while a writer updates; otherwise only one writer at a time. | `h5py` (Python), `pytables`, MATLAB, R `rhdf5`, C++ HDF5 library. |
| **TFRecord** | TensorFlow pipelines, large image/audio/video streams | Simple **record‑oriented** binary format → streamable, works with `tf.data` for efficient sharding, **supports compression** (`GZIP`, `ZLIB`). | TensorFlow‑centric (though can be read by other libs), no built‑in schema (you must define `tf.train.Example` protobuf). | Designed for **very large** datasets (hundreds of GB–TB). Files can be split into **shards** and read in parallel across workers. | Write concurrency is limited – typically one process writes a shard; you generate many shards in parallel and later concatenate. Reads are parallelizable. | TensorFlow `tf.io.TFRecordWriter`, `tf.data.TFRecordDataset`. |
| **RecordIO** (MXNet) | General binary record storage, often for image datasets | Simple length‑prefixed record format → fast sequential reads, works with MXNet `DataIter`. | Limited tooling outside MXNet, no columnar compression. | Works well for **large image collections** when sharded; each shard is a single file. | One writer per shard; multiple readers can read shards concurrently. | MXNet `mx.io.ImageRecordIter`. |
| **LMDB** (Lightning Memory‑Mapped Database) | Key‑value store for large image/audio datasets (e.g., Caffe, MXNet) | **Memory‑mapped B‑tree**, **zero‑copy reads**, **transactions**, **read‑only concurrency** (many readers), **fast random access**. | Write performance can be slower due to sync; file size must be pre‑allocated (or set large maxsize). | Handles **hundreds of millions** of small records efficiently; works well on SSDs. | **Multiple readers** are lock‑free; **single writer** at a time (but writes are fast). | `lmdb` Python binding, Caffe `convert_imageset`. |
| **Zarr** | Cloud‑native chunked N‑dimensional arrays (similar to HDF5) | **Chunked, compressed, cloud‑friendly** (stores each chunk as a separate object in S3/GS/ADLS), **metadata in JSON**, **concurrent read/write** via atomic chunk writes, **supports many compressors** (Blosc, Zstd). | Slightly higher latency for many tiny reads (each chunk is a separate HTTP request). | Scales to **petabyte‑scale** because each chunk is independent; can be accessed by many workers in parallel. | **Concurrent writers** are safe as long as they write to distinct chunks; atomic replace semantics avoid corruption. | `zarr` Python, `xarray` integration, Dask array, `zarr-py`. |
| **N5** | Similar to Zarr, used in bio‑imaging (e.g., BigDataViewer) | Chunked, hierarchical, supports multiple back‑ends (local FS, S3, GCS). | Smaller community than Zarr. | Same scaling properties as Zarr. | Same concurrency model. | `n5` Java library, `z5py` Python. |

---

## 3. Media Formats (Raw Input for CV / Audio / Video)

| Format | Typical Use | Strengths | Weaknesses | Scalability | Concurrency | Ecosystem |
|--------|-------------|-----------|------------|-------------|-------------|-----------|
| **JPEG / PNG / WebP / TIFF** | Image datasets (classification, detection, segmentation) | Widely supported, hardware‑accelerated decoders, lossy (JPEG) vs. lossless (PNG, TIFF) options, WebP offers better compression. | No built‑in indexing; each file is a separate I/O operation → many small files can overwhelm file‑system metadata. | Works for **any size** of dataset; for >10 M images you typically move to a container format (TFRecord, LMDB, Zarr) or a cloud bucket with parallel read. | Read‑only concurrency trivial; writes usually done offline (pre‑processing). | OpenCV, Pillow, TensorFlow `tf.io.decode_image`, PyTorch `torchvision.io`. |
| **TFRecord / RecordIO / LMDB** (binary containers) | Store images (or audio) as raw bytes inside a single file or key‑value store | Reduces file‑system overhead, enables **sharding** for parallel reads, can store labels together with data. | Requires custom loader; not directly viewable. | Designed for **TB‑scale** image/audio corpora. | Same as underlying format (single‑writer, many‑readers). | See sections above. |
| **WAV / FLAC / OGG** | Audio datasets (speech, music) | WAV = raw PCM (lossless, easy), FLAC = lossless compression, OGG = lossy but high quality. | Large file sizes for raw PCM; codecs may need extra libraries. | Usually stored as individual files; for massive corpora you can pack them into TFRecord/LMDB/Zarr. | Same as image case – many small files can stress FS; container formats help. | `librosa`, `torchaudio`, TensorFlow `tf.audio.decode_wav`. |
| **MP4 / MKV / WebM** | Video datasets (action recognition, video‑question‑answering) | Standard codecs, hardware‑accelerated decoding, streaming support. | Random access to frames is expensive; variable‑bit‑rate can cause uneven read times. | Large (GB‑scale) per video; typical pipelines extract frames and store them in TFRecord/Zarr. | Same as audio – many small files vs. container trade‑off. | OpenCV `VideoCapture`, `ffmpeg`, `decord`, `torchvision.io`. |

---

## 4. Model‑Serialization Formats (Often considered “data” for deployment)

| Format | Typical Use | Strengths | Weaknesses | Scalability | Concurrency | Ecosystem |
|--------|-------------|-----------|------------|-------------|-------------|-----------|
| **ONNX** (Open Neural Network Exchange) | Interoperable model exchange, inference across frameworks | **Framework‑agnostic**, supports many operators, can be optimized with ONNX Runtime, quantization, and hardware accelerators. | Not all custom ops are supported; version mismatches can be tricky. | Model size up to **hundreds of MB**; can be sharded (multiple ONNX files) for ensembles. | Read‑only concurrency safe; writing is a one‑off export step. | `onnx`, `onnxruntime`, `torch.onnx.export`, `tf2onnx`. |
| **PMML** (Predictive Model Markup Language) | Traditional statistical / tree models, rule‑based models | XML‑based, language‑agnostic, supported by many scoring engines. | Limited support for deep‑learning architectures. | Usually small (KB‑MB). | Same as ONNX. | `sklearn2pmml`, `jpmml`. |
| **TorchScript** | PyTorch model serialization for production | Captures both model graph and Python code, can be run without Python interpreter (`torch.jit.save`). | PyTorch‑specific, limited cross‑framework portability. | Same as ONNX. | Same as ONNX. | `torch.jit`. |
| **SavedModel** (TensorFlow) | TensorFlow serving, TF‑Lite conversion | Stores graph, variables, signatures; works with TensorFlow Serving, TF‑Lite, TF‑JS. | TensorFlow‑centric; larger disk footprint. | Same as ONNX. | Same as ONNX. | `tf.saved_model`. |

---

## 5. Summary Matrix

| Category | Format | Columnar? | Compression | Random Access | Schema | Concurrency (R/W) | Typical Size Range | Best‑Fit Scenarios |
|----------|--------|-----------|-------------|---------------|--------|--------------------|--------------------|--------------------|
| **Tabular** | CSV/TSV | No | None (gzip optional) | Linear scan only | None | R: many, W: single (file lock) | < 1 GB (human‑editable) | Quick prototyping, small datasets |
| **Tabular** | JSON/JSON‑L | No | None (gzip) | Linear (JSON‑L can be streamed) | None | Same as CSV | < 1 GB | Nested records, logs |
| **Tabular** | Parquet | Yes | Column‑wise (Snappy, ZSTD, GZIP) | Row‑group + column pruning → fast random column reads | Embedded (metadata) | R: many, W: single per file (write‑once) | 10 GB – PB | Data‑lake analytics, Spark/Dask pipelines |
| **Tabular** | Feather/Arrow IPC | Yes | LZ4/ZSTD (optional) | Whole‑file memory‑mapped → fast column reads | Embedded | R: many, W: single | < 10 GB (in‑memory) | Cross‑language data exchange, quick prototyping |
| **Tabular** | Delta Lake | Yes (Parquet underneath) | Same as Parquet | Same + transaction log for versioned reads | Full schema + versioning | Optimistic concurrency (multiple writers) | PB‑scale | ACID data lake, incremental pipelines |
| **Tabular** | Iceberg | Yes (Parquet) | Same as Parquet | Same + hidden partition pruning | Full schema + versioning | Optimistic concurrency | PB‑scale | Similar to Delta, multi‑engine |
| **Array** | NPY/NPZ | No (flat) | None (npz zip) | Memory‑mapped → random | dtype/shape only | R: many, W: single | < 10 GB | Quick dump of NumPy arrays |
| **Array** | HDF5 | Yes (chunked) | GZIP/LZF/others | Chunk‑level random | Hierarchical metadata | SWMR (single‑writer‑multiple‑reader) | 10 GB – hundreds GB | Scientific data, hierarchical datasets |
| **Array** | TFRecord | No (record) | GZIP/ZLIB (optional) | Sequential (shard‑level) | Defined by protobuf | One writer per shard; many readers | TB‑scale | TensorFlow pipelines |
| **Array** | LMDB | Yes (key‑value) | None (optional compression) | Key‑based random | None (you define keys) | R: many, W: single (fast) | Hundreds M records | Large image/audio DBs |
| **Array** | Zarr | Yes (chunked) | Blosc/ZSTD/LZ4/etc. | Chunk‑level random (cloud‑friendly) | JSON metadata | R: many, W: many (different chunks) | PB‑scale | Cloud‑native, Dask/Xarray workflows |
| **Media** | JPEG/PNG/… | No | Lossy/Lossless | File‑level random | None | R: many, W: offline | Any (many small files) | Raw image datasets |
| **Model** | ONNX | No | None (binary) | Whole‑model load | Embedded graph | R: many, W: single export | MB‑hundreds MB | Cross‑framework inference |
| **Model** | TorchScript | No | None | Whole‑model load | Embedded graph | R: many, W: single export | MB‑hundreds MB | PyTorch production |

---

## 6. Choosing the Right Format – Decision Guide

Below is a **flowchart‑style checklist** you can follow when you start a new ML project.

| Question | Recommended Format(s) |
|----------|-----------------------|
| **Is the data primarily tabular (rows × columns) and you need fast analytics?** | Parquet (or Delta/Iceberg if you need ACID/versioning). |
| **Do you need a human‑editable file for quick inspection or sharing with non‑technical stakeholders?** | CSV/TSV (maybe gzipped) or JSON/JSON‑L for nested data. |
| **Will you exchange data between Python, R, and Julia on a single node?** | Feather / Arrow IPC (zero‑copy). |
| **Do you have multi‑dimensional scientific arrays (e.g., satellite images, sensor cubes) that need random access and compression?** | HDF5 (if you stay on a single node) or Zarr (if you need cloud‑scale, parallel writes). |
| **Are you building a TensorFlow data pipeline that reads millions of images/audio files?** | TFRecord (sharded) or Zarr/LMDB for better random access. |
| **Do you need a key‑value store for fast random reads of small items (e.g., image patches)?** | LMDB (single‑writer, many‑readers) or Zarr (if you need concurrent writes). |
| **Will the dataset be stored in a cloud object store (S3, GCS, ADLS) and accessed by many workers?** | Parquet (columnar) for analytics, Zarr for chunked arrays, Delta/Iceberg for ACID. |
| **Do you need versioned, transactional updates (e.g., incremental feature engineering, data‑pipeline re‑runs)?** | Delta Lake or Iceberg (both sit on top of Parquet). |
| **Is the dataset composed of many tiny files (e.g., JPEGs) that cause file‑system bottlenecks?** | Pack them into TFRecord, LMDB, or Zarr to reduce metadata overhead. |
| **Do you need to ship a model to a different framework or edge device?** | ONNX (cross‑framework) or TorchScript/TF‑SavedModel (framework‑specific). |
| **Do you need to store both data and its schema together for reproducibility?** | Parquet (schema embedded) or HDF5/Zarr (metadata). |
| **Will you be using Spark, Presto, or Athena for ad‑hoc SQL queries?** | Parquet (or Delta/Iceberg for transactional). |
| **Do you need to read/write from multiple processes simultaneously on the same file?** | Use **SWMR HDF5**, **Delta/Iceberg**, or **Zarr** (chunk‑level concurrency). |

---

## 7. Practical Tips & Gotchas

| Tip | Why it matters |
|-----|----------------|
| **Chunk size matters** – In Parquet, HDF5, Zarr, LMDB the size of each chunk (row‑group, HDF5 chunk, Zarr chunk) determines the trade‑off between I/O latency and compression efficiency. Small chunks → fast random reads but lower compression; large chunks → better compression but higher memory usage. |
| **Avoid “many tiny files”** – Filesystems (ext4, NTFS, XFS) struggle with >10 M files. Use container formats (TFRecord, LMDB, Zarr) or store raw media in an object store and keep only a manifest (CSV/Parquet) pointing to the URIs. |
| **Use compression wisely** – ZSTD (level 3‑5) often gives the best speed‑compression ratio for Parquet and Zarr. For HDF5, LZF is fast but lower ratio; gzip gives higher ratio but slower. |
| **Leverage column pruning** – When you only need a subset of columns, Parquet/Feather/Arrow can read just those columns, dramatically reducing I/O. |
| **Version your data** – With Delta/Iceberg you can time‑travel to any previous snapshot. For plain Parquet, consider a **folder‑per‑date** or **folder‑per‑version** layout and keep a manifest file. |
| **Read‑only vs. write‑heavy workloads** – If you have many concurrent writers (e.g., streaming ingestion), choose a format that supports **optimistic concurrency** (Delta/Iceberg) or **chunk‑level atomic writes** (Zarr). |
| **Memory‑map when possible** – `np.load(..., mmap_mode='r')` for NPY/NPZ and `pyarrow.feather` for Feather allow you to work with data larger than RAM without loading everything. |
| **Cross‑language compatibility** – Arrow/Feather, Parquet, and ONNX are the only formats that truly work across Python, Java/Scala, C++, R, and Julia without conversion. |
| **Security** – Never unpickle data from an untrusted source. Prefer `joblib` with `mmap_mode='r'` for read‑only, or use a format with explicit schema (Parquet, Arrow). |
| **Testing** – When you switch formats, validate that a round‑trip (`write → read`) yields identical tensors/arrays (use checksums or hash of the data). |

---

## 8. Example Code Snippets

Below are minimal examples that illustrate **reading & writing** for the most common formats. They can be used as a starting point for pipelines.

### 8.1 Parquet (with compression & partitioning)

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Sample DataFrame
df = pd.DataFrame({
    "user_id": range(1_000_000),
    "event_ts": pd.date_range("2024-01-01", periods=1_000_000, freq="S"),
    "category": ["A", "B", "C", "D"] * 250_000,
    "value": np.random.rand(1_000_000)
})

# Write partitioned by category, using ZSTD compression
pq.write_to_dataset(
    pa.Table.from_pandas(df),
    root_path="s3://my-bucket/events/",
    partition_cols=["category"],
    compression="zstd",
    use_dictionary=True
)
```

### 8.2 Feather / Arrow IPC

```python
import pandas as pd
import pyarrow.feather as feather

df = pd.read_parquet("large.parquet")   # assume already in memory
feather.write_feather(df, "data.feather", compression="zstd")
# Zero‑copy read
df2 = feather.read_feather("data.feather")
```

### 8.3 HDF5 with chunking & compression

```python
import h5py
import numpy as np

data = np.random.randn(5000, 5000).astype('float32')
with h5py.File('mydata.h5', 'w') as f:
    dset = f.create_dataset(
        'images',
        data=data,
        chunks=(256, 256),          # 256×256 tiles
        compression='gzip',
        compression_opts=4
    )
# Random read of a tile
with h5py.File('mydata.h5', 'r') as f:
    tile = f['images'][1000:1256, 2000:2256]   # reads only needed chunks
```

### 8.4 TFRecord (sharded)

```python
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'label': _bytes_feature(np.int64(label).tobytes()),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Write 10 shards
for shard_id in range(10):
    filename = f'train-{shard_id:05d}-of-00010.tfrecord'
    with tf.io.TFRecordWriter(filename, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for i in range(1000):  # 1000 examples per shard
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            lbl = np.random.randint(0, 10)
            writer.write(serialize_example(img, lbl))
```

### 8.5 Zarr (cloud‑native)

```python
import zarr, numpy as np, dask.array as da

# Create a Zarr array on S3 (requires s3fs)
store = zarr.storage.S3Store('my-bucket/zarr-dataset')
root = zarr.group(store=store)

# Chunked 3‑D array (e.g., video frames)
shape = (1000, 256, 256)   # 1000 frames of 256×256
chunks = (100, 256, 256)   # one chunk per 100 frames
z = root.create_dataset('frames', shape=shape, chunks=chunks,
                        dtype='uint8', compressor=zarr.Blosc(cname='zstd', clevel=3))

# Write using Dask for parallelism
dask_arr = da.random.randint(0, 256, size=shape, chunks=chunks, dtype='uint8')
da.to_zarr(dask_arr, z, overwrite=True)
```

---

## 9. Closing Thoughts

* **No single format wins everywhere** – pick the one that matches **data shape**, **size**, **access pattern**, and **tooling**.  
* For **large, columnar analytics** → Parquet (or Delta/Iceberg for ACID).  
* For **high‑performance in‑memory exchange** → Feather / Arrow IPC.  
* For **hierarchical scientific data** → HDF5 (single node) or Zarr (cloud‑scale).  
* For **TensorFlow pipelines** → TFRecord; for **PyTorch / MXNet** → LMDB or RecordIO.  
* When you anticipate **concurrent writes** or **incremental updates**, lean on **Delta Lake**, **Iceberg**, or **Zarr** which were built for that scenario.  

By aligning the format with the **characteristics of your workload**, you’ll avoid I/O bottlenecks, keep your pipelines portable, and make future scaling (both in data volume and team size) much smoother. Happy modeling!