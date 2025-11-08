3.  **Design the Data Curation and Filtering Pipeline for a Next-Generation Foundation Model.**


# System Design

## Analysis

**Problem statement**

The next generation foundation model requires data for training, post-training,
instruction-tuning, chat-training, specific fine-tuning, alignment, safety. From
available datasets we need to curate and filter relevant data for each define
purpose.

### Scope

**Assumptions**

- New records are added to the dataset and are enqueued for processing
- The dataset is accessible through a data-lake / distributed filesystem
    as a readonly resource
- Rules for filtering , curation, and destination are defined ahead of time
- Assume that the data-curation pipeline is collocated with the training cluster
- Assume 10TB on storage per dataset processing
- Assume 1MB/s of peak data incoming
- Assume order of 10 parallel running pipelines

**Inside**

- Dataset changes during processing, records can be added/removed, 
    batch+online mode of operation
- Data relevant explainability, we need to relate data to training purpose 
    related model performance 
- Provide means to define rules for filtering/curation to users
- Python library to program the pipeline, data-scientists and ml-engineers
    need to have a way to programmatically configure the pipeline
- Extensibility: the pipeline needs to be extensible with custom microservices

**Outside**

- Data collection, we assume that the dataset/s are available.
- Rules definition, programmability
- Regional considerations
- Monitoring, observability, e.g. querying the logs , or registry
- Feedback loops
- Data considerations, specifically in what format is the data stored, what technology
    is used, such as json / csv / ELL . 
- Backup , data or service backup, global failures and catastrophic events

## Functional considerations

**High level architecture**

A filtering pipeline is a sequence of operations ( curations / filters ) run
on a data record. The result of that record can be stored back in the datastore
destination, and the meta-data with respect to the relevant record stored
in a database. Each operation of the pipeline run needs to execute in a two phase 
commit manner: Create and lock database destination record, finish the job and
write the  

```text
                                                    [Logging service]

[Control-plane-API]-->[Service Orchestrator]<------>[ Registry ]
                        |                               |
                        |                               |
[Data monitors]----------[AMQ]<------>|-----------|-----
                                      | Curators  | 
                                      |  W[1,2,..]|<---->[Process cache]
                                      |  T[...]   |
                                      |-----------|
                                            |               
                                            |               
                               [-- Records Cache --]
                                            |
                               [--- Data-store ---]


```


**Data flows**

```text
Data store -> Curators ... -> Data-store
```

**Components details**

* Control plane API: Provides REST API for client applications: 
    1. allowing programmability or curators, e.g. via an agreed upon syntax, 
    2. configuration of pipeline runs ( could have multiple in parallel)
    3. controlling execution of a pipeline run , etc. 

* Service orchestrator: Receives a job definition and creates a schedule of work
    as separate jobs. These are then placed into the message queue to be processed.
    Keeps track of completion by checking the datajob registry ( either poll/callbacks ).
    Segregates and organizes the work in batches and monitors progress. Stores
    ongoing state in the job registry.

* Curators: Curators are lambda functions defined via the control-api. Are triggered 
    from events on scheduled on the AMQ automagically.

* Records Cache: Different pipeline runs may require to operate on the same
    data records , so we can choose our prefered technology of this cache. This
    can be a two way cache both write and read , as curators will have checkpointing
    and may need to save intermediate state for records. 

* Process Cache: Stores intermediate artifacts, e.g. processing an audio file may
    result in producing embeddings that can then be used by another lambda but are 
    not part of the final processed dataset.

* Data store: Keeps the data records, can receive new records from other sources,
    receives the produced dataset after curation which can be in place. 

* Data monitors: Notify for incoming data changes and records that need to be 
    processed for permanent running jobs. Service orchestrator consumes the messages
    and updates the appropriate pipeline job schedule by issuing relevant events.

* Registry: Keeps track of job and data metadata , can be made up of multiple 
    replicas and sharded across topics or size.

* Logging service: Used by every component to log operations as needed, ( e.g. splunk )

## Non functional considerations

**Scalability**

Fully scalable horizontally and vertically. Curators can be sized as needed, and 
have multiples replicas per type. Data-store technology can be chosen to be read-heavy,
behind a CDN or other technologies. AMQ pipelining allows to scale-out according to load
of jobs running in parallel and ml-engineers or data-scientists working in parallel.

Service orchestrator is a bottleneck but mainly does work at start of each job
and subsequently monitors completion of work. But that is also not a demanding
operation as it mostly relates to checking the completion on intervals and rescheduling
missing jobs. 

Registry, can be a bottleneck especially if many small jobs need to be providing 
meta-data updates, but can be scaled if we choose a no-sql solution, such as 
mongo-db. 

**Reliability**

Hihgly available and fault tolerant. If a curator fails it can be protected through
Transactional API on the AMQ and service scheduler control on the final state 
in the registry.

**Risks and Variables**

Minimal risk of some jobs failing and ending up in the DLQ but the service 
orchestrator can monitor and cross check with the job registry and reschedule the job.

Orchestrator failures need to be durable and the service have cold-replicas.


**Summary**

This is a full-blown AP distributed pipeline service geared towards high availability 
and eventually consistent processing with reconciliation of failures. It is 
diagonally scalable with small future modifications. 





This is a system design of a candidate for a senior staff software engineer at
Google. This was the first technical screening 30-minute sessions.
What do you think ? Would you give it a pass? and how would you improve. 
What questions would you ask, and what would you like to see in the next stage if
this was a pass?


---
This is an excellent question and a great piece of work to analyze. It's a very realistic scenario for a senior/staff level interview.

Let's break this down from the perspective of an interviewer at a company like Google.

### Overall Impression

This is a strong response for a 30-minute initial screening. The candidate demonstrates a solid understanding of distributed systems principles and follows a structured approach. They cover the essential bases: requirements, components, data flow, and non-functionals. The design is a classic, scalable, event-driven architecture, which is a very sound choice.

**Verdict: A comfortable Pass.**

The candidate has proven they have the foundational knowledge of system design. They didn't make any major errors, the system is plausible, and they thought about scale and reliability. They've earned a follow-up interview.

---

### Detailed Critique: Strengths and Weaknesses

#### Strengths:

1.  **Structured Thinking:** The candidate didn't just jump into drawing boxes. They started with `Analysis` -> `Scope` -> `Functional` -> `Non-Functional`. This is exactly what interviewers want to see.
2.  **Quantification:** The candidate provided specific numbers (`10TB`, `1MB/s`, `10 parallel pipelines`). This is a huge plus. It grounds the design in reality and shows they're not just thinking in the abstract. It moves the conversation from "it needs to be scalable" to "it needs to scale *to this degree*."
3.  **Solid Architectural Pattern:** The event-driven architecture using a message queue (AMQ), an orchestrator, and stateless workers (Curators) is a robust and highly scalable pattern. It's a textbook solution for this kind of decoupled processing workload.
4.  **User-Centric Features:** Mentioning a "Python library to program the pipeline" and an API for configuration shows empathy for the end-users (Data Scientists, ML Engineers), which is a sign of a mature engineer.
5.  **Awareness of Bottlenecks:** The candidate correctly identified the Service Orchestrator and the Registry as potential bottlenecks and suggested ways to mitigate this (scaling the registry, acknowledging the orchestrator's less-intensive role post-scheduling).

#### Areas for Improvement & Deeper Probing

This is where we separate a "good senior" answer from a "great staff" answer. The current design is a **generic data processing engine**. It's excellent, but it's not yet specific to the unique, messy, and critical challenges of **Foundation Model data curation**.

Here's how to elevate it:

**1. The "What" is Missing: Focus on the Curation Steps**

The design expertly describes *how* to process data but barely touches on *what* the processing steps are. The "Curators" are black boxes. A staff-level candidate should be ableto name and describe the actual curation tasks.

*   **Improvement:** Explicitly state that the pipeline is a Directed Acyclic Graph (DAG) of specific curation tasks. Give examples:
    *   **Deduplication:** Mention both exact hash-based deduplication and more complex semantic/fuzzy deduplication (e.g., using MinHash, SimHash, or embedding-based clustering). This is a massive task at this scale.
    *   **Quality Filtering:** How do you define "high quality"? Mention filtering based on heuristics (e.g., length, symbol-to-word ratio, stop-word count) and model-based filtering (e.g., using a perplexity model to discard nonsensical text).
    *   **PII & Toxicity Removal:** Mention using named-entity recognition (NER) models for PII and classifier models for toxicity. Discuss the trade-off of false positives/negatives.
    *   **Data Mixing:** Foundation models aren't trained on a monolith. They're trained on a specific "recipe" of data (e.g., 60% web text, 20% books, 15% code, 5% conversation). The pipeline must be able to sample from curated sources to create these final training mixes.

**2. Data Lineage Is Underspecified but CRITICAL**

The candidate mentions "data relevant explainability" and a "Registry," but this is the most critical part of the whole system and needs more detail. Reproducibility is non-negotiable.

*   **Improvement:** Detail the "Registry" as a **Data Lineage Store**.
    *   For every single output record, this store must be able to answer:
        *   What source record(s) did it come from?
        *   What sequence of "Curator" functions was applied?
        *   What was the exact version/commit hash of the code for each function?
        *   What were the parameters for each function (e.g., the toxicity threshold was `0.85`)?
    *   This allows you to trace a bad model behavior back to the exact data and rule that produced it. It also allows you to reproduce the exact dataset years later. The schema for this is non-trivial (e.g., a graph database like Neo4j might be a good fit).

**3. Designing for Iteration and Cost**

Researchers constantly change their minds. "Let's lower the perplexity filter from 50 to 45." Reprocessing 10TB of data for one parameter change is prohibitively expensive and slow.

*   **Improvement:** The "Process Cache" and "Records Cache" are good starts, but they can be made more explicit.
    *   Design the pipeline so that the output of each major curation step is **materialized** back to the data store (e.g., S3/GCS) with a deterministic name (e.g., `s3://bucket/step=dedup/hash=<input_hash>/data.parquet`).
    *   The orchestrator can then check if the output for a given step and a given input already exists. This enables **incremental computation** and massive cost savings. The "Process Cache" becomes this intermediate, content-addressable storage. This is a core concept in tools like DVC and Dagster.

**4. Human-in-the-Loop**

Not all filtering can be automated. Quality is subjective.

*   **Improvement:** Add a "Human Review" stage to the pipeline.
    *   When a curator is uncertain (e.g., a toxicity score is between 0.6 and 0.8), it can flag the record and route it to a separate queue.
    *   This queue feeds a simple UI where human annotators can provide a label (e.g., "toxic" / "not toxic").
    *   These human-generated labels are then fed back into the system, both to correct the specific record and potentially to fine-tune the classifier models.

---

### Questions I Would Ask in the Follow-up

Based on the initial design, my goal would be to probe the areas of improvement listed above.

1.  **On Curation Specifics:** "This is a great framework. Let's make it concrete. Imagine our source is a 100TB dump of Common Crawl. What would be the first three 'Curator' microservices in your pipeline, and what specific logic would they implement?"
2.  **On Data Lineage:** "A year after training, we find our model has a bias against a certain demographic. How would you use your system, specifically the 'Registry', to debug this? What query would you run to trace this behavior back to the source data?"
3.  **On Iteration & Cost:** "A data scientist wants to test a new PII removal rule. Your full dataset is 50TB. How can they do this without incurring the cost and time of re-running the entire pipeline from scratch? Walk me through the flow."
4.  **On Data Flow & Scale:** "Let's talk about the data moving between the 'Curators'. Are you passing the full 1MB text document in the message queue? What are the implications of that choice versus passing a pointer to the data in the Data Store?"
5.  **On Failure & State:** "You mentioned a two-phase commit for the database record. Can you elaborate on the transaction boundary? What happens if a curator processes a record, writes its output to the datastore, but fails before it can update the 'Registry'?"

### What I'd Want to See in the Next Stage

If this was the first screen, I'd want the next round (the 45-60 min deep dive) to focus on:

*   **Depth in one area:** Pick one component, like the "Deduplication Service," and design it in detail. Discuss algorithms (MinHash vs. embeddings), data structures, and how to scale it to 10TB+.
*   **API Design:** Sketch out the REST API for the `Control-plane-API` or the schema for the `Registry`.
*   **Trade-offs:** Discuss technology choices with more depth. Why a message queue over a workflow orchestrator like Airflow/Dagster? Why a NoSQL DB for the registry vs. a graph DB? What are the pros and cons of each in this specific context?
*   **Operational Maturity:** How do you handle schema evolution for the data? How do you version the Curator functions? How do you monitor data quality degradation over time?

In summary, the candidate built a great car chassis. The next step is to prove they also know how to build the engine, transmission, and navigation system that are specific to the F1 race they're about to enter.