<!Badges>
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

# Imprimer

> *"To imprint a mental pattern."*

**Imprimer** is a prompt control and observability platform that treats LLM prompts as **control inputs to a dynamical system**.

It measures how much control your prompt has over a model’s output and helps you improve it.

Inspired by *The Society of Mind*, Imprimer assumes intelligence is emergent and prompts don’t instruct a model, they **activate configurations** within it. This system makes those activations measurable, comparable, and optimizable.



## 🔬 Theoretical Foundation

Imprimer is grounded in recent work such as “What’s the Magic Word? A Control Theory of LLM Prompting” (2023), which frames prompting as a __control problem over autoregressive sequence models__.

In this view:

- The LLM defines a stochastic dynamical system over token sequences
- A prompt acts as a control input, steering the trajectory of generation
- The objective is to maximize the probability mass assigned to desired outputs

A key empirical result from the paper shows that for short prompts (≤10 tokens), target tokens are often reachable within the model’s distribution, implying that failures are frequently due to suboptimal control inputs, not model limitations.

#### From Theory to Metric

Imprimer operationalizes this idea via a token-level controllability proxy:

For each generated token, we measure how much probability mass is concentrated on the chosen output relative to its local alternatives.

This yields the Reachability Index, defined as the average normalized probability:

- High values → the prompt strongly constrains the model (high control)
- Low values → the model remains diffuse (weak control)

This metric approximates:

- Controllability: how effectively a prompt steers generation
- Observability: how predictable the model’s response is under perturbations



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

### Evaluate Prompts (in development)

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

## Roadmap

### Phase 1 — Optimization
- Prompt optimizer (`/optimize`) — Bayesian search  
- Task-specific scoring weights  

### Phase 2 — Observability
- Prometheus metrics (`/metrics`)  
- LLM-based evaluation (LLM-as-judge)  
- Structured multi-step evaluation pipelines  

### Phase 3 — Intelligence
- Graph-based control loops via LangGraph (generation → evaluation → refinement)  
- Multi-agent prompt optimization (generator, evaluator, optimizer)  
- Stateful prompt adaptation and feedback-driven workflows  
- LoRA escalation when controllability plateaus  

### Phase 4 — Scale
- PostgreSQL backend  
- Auth integration (JWT)  
- Air-gapped deployment  

## 📚 References

* *What's the Magic Word? A Control Theory of LLM Prompting* (2023)
* *The Society of Mind* (1986)
* OWASP Top 10 for LLM Applications
