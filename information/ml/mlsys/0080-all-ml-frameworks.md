# List of machine learning frameworks by category

Machine learning today is built on a rich ecosystem of specialized frameworks that cater to every stage of development—from model conception to large‑scale deployment. For deep‑learning and neural‑network research, TensorFlow, PyTorch, and JAX dominate thanks to their flexible APIs, extensive tooling, and strong hardware support, while higher‑level libraries such as Keras, FastAI, and PyTorch Lightning streamline model construction and training pipelines. Classical machine‑learning workloads continue to rely on mature libraries like scikit‑learn, XGBoost, LightGBM, and CatBoost, complemented by scalable platforms such as Apache Spark MLlib and H2O.ai for big‑data environments. AutoML solutions—including Auto‑Keras, Auto‑sklearn, TPOT, and H2O AutoML—automate hyper‑parameter search and model selection, accelerating the path from data to production.

Serving and inference have similarly converged around dedicated platforms that optimize latency, throughput, and resource utilization for both general models and large language models (LLMs). Production‑grade servers such as TensorFlow Serving, TorchServe, NVIDIA Triton, and KServe (KFServing) provide robust model‑versioning and scaling, while specialized LLM engines—vLLM, Hugging Face Text Generation Inference, DeepSpeed‑MII, TensorRT‑LLM, and FasterTransformer—deliver high‑throughput generation with advanced quantization (e.g., GPTQ, AWQ, Bitsandbytes, QLoRA) and sharding techniques. Training‑time frameworks like Hugging Face Transformers, DeepSpeed, Megatron‑LM, and ColossalAI further accelerate LLM development through tensor‑parallelism, ZeRO optimizations, and low‑precision (4‑bit/8‑bit) training, ensuring that modern AI pipelines can move seamlessly from research prototypes to production‑grade services.


## List all ML frameworks

**Deep‑learning / neural‑network frameworks**
- TensorFlow  
- PyTorch  
- Keras (high‑level API, runs on TensorFlow, Theano, CNTK)  
- MXNet / Gluon  
- Caffe / Caffe2  
- Theano (now discontinued)  
- Microsoft CNTK (aka Cognitive Toolkit)  
- JAX (Google)  
- Chainer  
- PaddlePaddle (Baidu)  
- Deeplearning4j (Java)  
- ONNX Runtime (inference)  
- FastAI (built on PyTorch)  

**Classical machine‑learning libraries**
- scikit‑learn (Python)  
- XGBoost  
- LightGBM  
- CatBoost  
- H2O.ai (including H2O‑AutoML)  
- Apache Spark MLlib  
- Apache Mahout  
- Shogun  
- Vowpal Wabbit  
- Dlib (C++)  
- libsvm / liblinear  

**AutoML / hyper‑parameter search tools**
- Auto‑Keras  
- Auto‑sklearn  
- TPOT  
- H2O AutoML  
- Google Cloud AutoML (service)  

**Time‑series & statistical modeling**
- Prophet (Facebook)  
- Statsmodels  

**Other notable frameworks**
- JuliaFlux (Flux.jl)  
- MXNet‑GluonTS (forecasting)  
- TensorFlow.js (browser)  
- TensorFlow Lite / TensorFlow Mobile (edge)  
- PyTorch Lightning (structured training)  
- Horovod (distributed training)  

*The list is not exhaustive but covers the most widely used machine‑learning frameworks across different languages and application domains.*


## Essential Deeplearning frameworks

- TensorFlow Serving  
- TorchServe  
- ONNX Runtime Server  
- NVIDIA Triton Inference Server  
- KFServing (now KServe)  
- Seldon Core  
- MLflow Model Serving  
- BentoML  
- OpenVINO Model Server  
- Paddle Serving


## All the essential LLM serving frameworks


- vLLM – high‑throughput, low‑latency inference engine with paged attention  
- Text Generation Inference (TGI) – Hugging Face‑optimized server for decoder‑only models  
- DeepSpeed‑MII – multi‑instance inference with ZeRO‑offload and tensor parallelism  
- TensorRT‑LLM – NVIDIA‑accelerated inference for FP8/FP16/INT8 quantized models  
- FasterTransformer – CUDA‑based transformer kernels for fast generation  
- Triton Inference Server – model‑agnostic serving with GPU scheduling, ensemble support, and model‑versioning  
- TorchServe – production‑ready serving for PyTorch models, supports custom handlers  
- TensorFlow Serving – native serving for TensorFlow‑based LLMs  
- Ray Serve – scalable Python‑level serving, easy integration with LangChain or custom pipelines  
- SageMaker Inference (AWS) – managed endpoint service with automatic scaling and multi‑model support  
- Azure Machine Learning Managed Endpoints – hosted inference with GPU autoscaling  
- Google Vertex AI Prediction – managed serving with GPU/TPU options  

These frameworks cover the most common deployment scenarios: single‑GPU low‑latency serving, multi‑GPU/cluster scaling, hardware‑specific acceleration, and managed cloud endpoints.


## Essential LLM training frameworks

- PyTorch  
- TensorFlow  
- JAX  
- Hugging Face Transformers  
- DeepSpeed  
- Megatron‑LM  
- FairScale (ZeRO)  
- ColossalAI  
- Bitsandbytes (for 4‑bit/8‑bit training)  
- NVIDIA NeMo  
- OpenAI RLHF (for fine‑tuning)

## Essential LLM Quantization

- Unsloth
- GPTQ (and its variants)
- AWQ (Activation‑aware Weight Quantization)
- BitsandBytes (8‑bit/4‑bit optimizers)
- QLoRA (Quantized LoRA fine‑tuning)
- LLM.int8 (int8 inference)
- DeepSpeed ZeRO‑Quant (ZeRO + quantization)
- TensorRT‑LLM (GPU‑accelerated int8/float16)
- ONNX Runtime Quantization (static/dynamic)


## Tiered view

### Tier 1: The Most Popular & Widely-Used Frameworks

These are the dominant forces in the ML world today, especially for deep learning. Most new projects will use one of these.

1.  **PyTorch (Meta)**: Currently the leading framework in the research community and gaining massive adoption in the industry. Known for its Python-first approach, flexibility, and easy-to-use interface.
    *   **Ecosystem:**
        *   **PyTorch Lightning:** A high-level wrapper for PyTorch that organizes code and removes boilerplate, making experiments more reproducible.
        *   **fast.ai:** An even higher-level library built on PyTorch, designed to make deep learning accessible to everyone.
        *   **TorchVision, TorchAudio, TorchText:** Official libraries for common datasets and models in computer vision, audio, and NLP.

2.  **TensorFlow (Google)**: The other industry giant. It was dominant for years, especially for production deployment, and has a massive ecosystem.
    *   **Ecosystem:**
        *   **Keras:** The official high-level API for TensorFlow. It's user-friendly and the recommended way to build models in TensorFlow.
        *   **TensorFlow Extended (TFX):** An end-to-end platform for deploying production ML pipelines.
        *   **TensorFlow Lite (TFLite):** A lightweight version for deploying models on mobile and embedded devices.
        *   **TensorFlow.js:** A library for training and deploying models in JavaScript environments (like web browsers).

3.  **Scikit-learn**: The undisputed king for **classical machine learning** (i.e., non-deep learning). If you're doing regression, classification, clustering, or dimensionality reduction with tabular data, this is your go-to library. It's known for its simple, consistent API and excellent documentation.

---

### Specialized & High-Performance Frameworks

These are gaining popularity for specific use cases or for pushing the boundaries of performance.

4.  **JAX (Google)**: Not a full framework, but a library for high-performance numerical computing and machine learning research. It combines automatic differentiation (`grad`) and JIT compilation (`jit`) for incredible speed on GPUs/TPUs. It's becoming the foundation for new ML libraries.
    *   **Built on JAX:**
        *   **Flax (Google)**: A popular and flexible neural network library for JAX.
        *   **Haiku (DeepMind)**: A simple neural network library for JAX, modeled after Sonnet (a DeepMind library for TensorFlow).

5.  **Hugging Face Transformers**: While technically a library, it functions as a framework for **Natural Language Processing (NLP)**. It provides thousands of pre-trained models (like BERT, GPT-2, T5) and makes them easy to use with PyTorch or TensorFlow. It's the standard for almost any NLP task today.

---

### Gradient Boosting Frameworks

These are so dominant for **tabular data competitions** (like on Kaggle) and many industry applications that they deserve their own category. They often outperform deep learning models on structured data.

6.  **XGBoost (eXtreme Gradient Boosting)**: The original high-performance gradient boosting library. Known for its speed and accuracy.
7.  **LightGBM (Light Gradient Boosting Machine)**: A faster, more memory-efficient alternative from Microsoft. Excellent for large datasets.
8.  **CatBoost**: A gradient boosting library from Yandex that excels at handling categorical features automatically.

---

### Frameworks for Big Data & Distributed Computing

These are designed to work with massive datasets that don't fit on a single machine.

9.  **Apache Spark MLlib**: The machine learning library for Apache Spark. It allows you to train models on huge datasets stored in a distributed cluster.
10. **Ray**: An open-source framework for scaling any computation, including ML workloads.
    *   **RLlib**: A library for reinforcement learning that runs on Ray.
    *   **Tune**: A library for hyperparameter tuning that runs on Ray.

---

### MLOps & Deployment Frameworks

These focus on the full lifecycle of a model, from training to production deployment and monitoring.

11. **MLflow**: An open-source platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment.
12. **Kubeflow**: The "ML toolkit for Kubernetes." It aims to make deploying and managing ML systems on Kubernetes simple, portable, and scalable.
13. **ONNX (Open Neural Network Exchange)**: Not a framework, but a standard format for representing ML models. It allows you to train a model in one framework (like PyTorch) and deploy it in another (like ONNX Runtime), improving interoperability.
14. **Amazon SageMaker**: A fully managed cloud service that provides tools to build, train, and deploy ML models at scale. It integrates with many of the frameworks listed above.

---

### Frameworks in Other Languages

While Python dominates, other languages have their own ML ecosystems.

15. **ML.NET (Microsoft)**: An open-source, cross-platform ML framework for **.NET developers**.
16. **Deeplearning4j (DL4J)**: A popular deep learning library for the **Java Virtual Machine (JVM)**.
17. **Weka**: A collection of ML algorithms for data mining tasks written in **Java**. Features a popular graphical user interface.
18. **Caret** and **Tidymodels**: Two popular frameworks in the **R** programming language for building and evaluating models.

---

### Historically Significant / Less Common Frameworks

These were once major players or serve niche roles. They are important for understanding the history of deep learning.

19. **Theano**: One of the earliest and most influential deep learning frameworks (now discontinued). It pioneered the concept of the computation graph that TensorFlow later adopted.
20. **Caffe / Caffe2**: Known for its blazing-fast performance in computer vision. Caffe2 was later merged into PyTorch.
21. **Apache MXNet**: A flexible and efficient deep learning framework. It was heavily backed by Amazon Web Services (AWS) but has since seen its popularity decline in favor of PyTorch and TensorFlow.
22. **CNTK (Microsoft Cognitive Toolkit)**: Microsoft's open-source deep learning framework. Development has largely ceased as Microsoft has shifted its focus to contributing to PyTorch.


