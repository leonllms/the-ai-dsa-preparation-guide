# Writing a system design document
 
**System‑Design Document – What to Include and How to Write It**

---

### 1. Title & Metadata
- Document title (e.g., “Design of XYZ Service”)
- Author, date, version
- Audience (engineers, managers, reviewers)

### 2. Problem Statement
- Brief description of the business problem or user need.
- Success criteria (latency, throughput, availability, cost, etc.).

### 3. Scope & Assumptions
- What is in‑scope vs. out‑of‑scope.
- Key assumptions (traffic volume, data size, deployment environment, third‑party services).

### 4. High‑Level Architecture
- One‑page overview of major components (client, API gateway, services, databases, cache, message bus, etc.).
- Use simple ASCII diagrams if graphics are not allowed, e.g.:

```
Client -> Load Balancer -> API Gateway -> Service A
                                   -> Service B -> DB
                                   -> Service C -> Cache
```

### 5. Component Details
For each major component include:
- **Purpose** – what it does.
- **Key responsibilities** – CRUD, business logic, etc.
- **Interfaces** – REST/GRPC endpoints, request/response schemas.
- **Data storage** – type (SQL, NoSQL, blob), schema highlights, partitioning.
- **Dependencies** – other services, external APIs.

### 6. Data Model
- Entity‑relationship description (tables, collections, primary keys).
- Important indexes and access patterns.
- Data flow (write path, read path, eventual consistency points).

### 7. Scalability & Performance
- Load estimates (RPS, concurrent users, data volume).
- Horizontal scaling strategy (stateless services, sharding, partition keys).
- Caching layers (in‑memory, CDN) and cache‑invalidation approach.
- Bottleneck identification and mitigation (back‑pressure, rate limiting).

### 8. Reliability & Fault Tolerance
- Redundancy (multi‑AZ, multi‑region).
- Failure handling (retries, circuit breakers, fallback).
- Data durability (replication factor, backup schedule).
- Monitoring & alerting (metrics, logs, health checks).

### 9. Security
- Authentication & authorization (OAuth, JWT, RBAC).
- Data protection (encryption at rest & in transit).
- Input validation, rate limiting, audit logging.

### 10. Trade‑offs & Alternatives
- List major design choices (e.g., SQL vs NoSQL, synchronous vs async).
- Pros/cons, why the chosen option fits the requirements.

### 11. Deployment & Operations
- CI/CD pipeline overview.
- Infrastructure as code (Terraform, CloudFormation) mention.
- Rolling upgrade / canary strategy.

### 12. Open Questions & Risks
- Items needing clarification.
- Potential risks and mitigation plans.

### 13. References
- Links to related design docs, API specs, external services.

---

#### Writing Tips
- **Use clear headings** and numbered sections for easy navigation.
- **Keep paragraphs short**; bullet points are preferred.
- **Define acronyms** on first use.
- **Include simple ASCII diagrams** where visual aid is needed.
- **Highlight key numbers** (e.g., “target 99.9% uptime”, “≤ 100 ms latency”) in bold.
- **Review for completeness**: does a reader unfamiliar with the problem understand the solution end‑to‑end?

Following this structure ensures the document conveys the full design while staying readable in plain‑text format.