# Imprimer

> *"To imprint a mental pattern."*

**Imprimer** is a prompt control and observability platform that treats LLM prompts as **control inputs to a dynamical system**.

**In one line:**
It measures how much control your prompt has over a model’s output—and helps you improve it.

Inspired by *The Society of Mind*, Imprimer assumes intelligence is emergent—and prompts don’t instruct a model, they **activate configurations** within it. This system makes those activations measurable, comparable, and optimizable.



## 🔬 Theoretical Foundation

Based on *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023).

Key result:

> For prompts ≤ 10 tokens, the correct next token is reachable ~97% of the time.

**Imprimer’s goal:**
Close the gap between your current prompt and that theoretical ceiling.



## ✨ System Highlights

| Metric             | Value          | Detail                               |
| :-- | :- | :-- |
| **Architecture**   | 2 Services     | Go gateway + Python ML engine (gRPC) |
| **Observability**  | 3 Metrics      | Reachability, latency, length        |
| **Baseline Score** | ~0.65–0.69     | On `qwen2.5:1.5b`                    |
| **Security**       | Injection Scan | Pre-execution filtering              |
| **Tracing**        | Distributed    | 1 trace ID per request               |
| **Deployment**     | 1 Command      | `docker compose up --build`          |



## ⚙️ Architecture

Strict separation between infrastructure and model logic via `.proto`.

* **Go Gateway (8080)**
  Auth, tracing, audit logging, routing

* **Python Engine (50051)**
  LLM execution, scoring, security, persistence

### Request Flow

```
POST /prompt
 ├── Gateway (Auth, Trace)
 │    └── gRPC Boundary
 └── Engine
      ├── Injection Scan
      ├── Model Execution (A/B)
      ├── Scoring
      └── Storage (SQLite)
```



## 🧮 Control Score (Reachability)

For each token:

* $p = \exp(\text{logprob})$
* $\text{total} = \sum_{i=1}^{5} \exp(\text{lp}_i)$
* $\text{certainty} = \frac{p}{\text{total}}$

**Reachability Index = average certainty across tokens**

* **1.0** → deterministic output
* **~0.2** → weak control

Over time, Imprimer builds memory:

```
GET /best?task=summarize
```

→ returns the highest-control prompt historically.



### Why It Matters

* Prompt engineering is mostly trial-and-error → now measurable
* Small prompt changes can drastically shift outputs → now optimizable
* Teams forget what worked → now persistent and queryable


## 🚀 Quickstart

### Prerequisites

* Docker
* Ollama (`qwen2.5:1.5b`)
* `OLLAMA_HOST=0.0.0.0`

### Run

```bash
docker compose up --build
```

### Evaluate Prompts

```bash
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "task": "summarize",
    "input": "Your input text here",
    "variant_a": "Summarize this in one sentence: {input}",
    "variant_b": "You are an expert writer. Give a precise one sentence summary of: {input}",
    "backend": "ollama"
  }'
```

### Query Best Prompt

```bash
curl "http://localhost:8080/best?task=summarize"
```



## 🗺️ Roadmap

**Phase 1 - Optimization**

* Prompt optimizer (`/optimize`) — Bayesian search
* Task-specific scoring weights

**Phase 2 - Observability**

* Prometheus metrics (`/metrics`)
* LLM-based evaluation

**Phase 3 - Intelligence**

- Multi-agent prompt optimization via LangGraph (generator, evaluator, optimizer loops)
- Stateful prompt refinement and adaptive control workflows
- LoRA escalation when controllability plateaus
- Transition from A/B testing to graph-based optimization pipelines

**Phase 4 - Scale**

* PostgreSQL backend
* Auth integration (JWT)
* Air-gapped deployment



## 📚 References

* *What's the Magic Word? A Control Theory of LLM Prompting* (2023)
* *The Society of Mind* (1986)
* OWASP Top 10 for LLM Applications
