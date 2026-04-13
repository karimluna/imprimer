<p align="center">
  <h1 align="center">Imprimer</h1>
  <img src="docs/assets/imprimer.drawio.png" height=190 style="display: block; margin: 0 auto;">
  <p align="center">Prompt control and observability platform for LLMs</p>
  <p align="center">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Go-1.25.6-00ADD8?logo=go" alt="Go">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/gRPC-contract--first-00897B" alt="gRPC">
  </p>
</p>

---

> *"To imprint a mental pattern."*
>
> Inspired by Minsky's *The Society of Mind* (1986): a prompt does not instruct a unified intelligence — it **activates a specific configuration** of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable over time.

---

## What it does

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**.

Given a task and two prompt variants, Imprimer asks: which prompt gives you more control over the model's output distribution? It measures this with a **Reachability Index** grounded in the paper *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023) — the first mathematically rigorous analysis of prompt controllability over autoregressive sequence models.

Every evaluation is persisted. Over time, the system learns which prompts control each task most effectively and surfaces that knowledge through the `best` command and `/best` endpoint.


## Theoretical foundation

The LLM defines a stochastic dynamical system over token sequences. A prompt acts as a **control input** $u$ that steers the trajectory of generation toward a desired output region.

For each generated token, Imprimer measures how much probability mass concentrates on the chosen output relative to its local alternatives:

$$p = \exp(\text{logprob}) \qquad \text{total} = \sum_{i=1}^{5} \exp(\text{lp}_i) \qquad \text{certainty} = \frac{p}{\text{total}}$$

The **Reachability Index** is the average certainty across all output tokens:

| Score | Meaning |
|---|---|
| `1.0` | Deterministic — the prompt leaves the model no uncertainty |
| `~0.65–0.69` | Observed on `qwen2.5:1.5b` with default prompts |
| `0.97` | Paper's theoretical upper bound for prompts ≤ 10 tokens |
| `~0.2` | Weak control — five tokens equally likely at each position |

The gap between your prompt's reachability and `0.97` is the optimization target.


## Architecture

Imprimer is two services connected by a gRPC contract. The proto file is the single source of truth — Go and Python never share code, only the contract.

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / curl                           │
│              imprimer evaluate / optimize / best             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP :8080
┌──────────────────────────▼──────────────────────────────────┐
│                      Go Gateway                              │
│   Auth · Audit (trace ID) · Rate limit · Prometheus metrics  │
│   /prompt  /optimize  /best  /metrics  /health               │
└──────────────────────────┬──────────────────────────────────┘
                           │ gRPC :50051
┌──────────────────────────▼──────────────────────────────────┐
│                    Python Engine                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Security   │  │   Chains     │  │    Optimizer       │  │
│  │  (inject.)  │  │  (LangChain) │  │  (Optuna TPE)      │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Scorer    │  │    Judge     │  │    Registry        │  │
│  │ (reachabil.)│  │ (LLM-as-j.)  │  │   (SQLite)         │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐   │
│  │           Observability (structured JSON trace)        │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Go handles:** HTTP ingress, authentication, audit logging, Prometheus metrics, gRPC routing. Go's goroutine model handles concurrent LLM requests efficiently.

**Python handles:** LLM inference (Ollama or OpenAI), logprob extraction, reachability computation, LLM-as-judge scoring, Optuna optimization, injection scanning, registry persistence.

**The boundary:** `proto/imprimer.proto` — three RPCs, never more complexity than needed.

---

## Quickstart

### Prerequisites

- Docker Desktop
- Ollama with `qwen2.5:1.5b`: `ollama pull qwen2.5:1.5b`
- Ollama listening on all interfaces (required for Docker):

```bash
# Set permanently, then restart Ollama from system tray
export OLLAMA_HOST=0.0.0.0
```

### Start the stack

```bash
docker compose up --build
```

Gateway on `:8080`. Engine on `:50051` (internal).

### Install the CLI

```bash
go install github.com/BalorLC3/Imprimer/gateway/cmd/imprimer@latest
```

Or build locally:

```bash
go install ./gateway/cmd/imprimer/
```

---

## CLI reference

The primary interface to Imprimer. All commands talk to the gateway over HTTP — works with Docker or a remote deployment.

### Global flags

```
--gateway string   Gateway URL (default "http://localhost:8080")
--api-key string   Bearer token (or set IMPRIMER_API_KEY)
--json             Raw JSON output instead of formatted text
```

---

### `imprimer evaluate`

Run two prompt variants against an input and compare their reachability scores.

```bash
imprimer evaluate \
  --task summarize \
  --input "Minsky argued intelligence emerges from many small agents none of which is intelligent alone" \
  --a "Summarize this in one sentence: {input}" \
  --b "You are an expert writer. Give a precise one sentence summary of: {input}" \
  --backend ollama
```

Output:
```
  Trace ID  9bc004ea-4b8a-4dc2-860d-1bf08ac0014f
  Winner    variant b

  Variant A     score=0.456  latency=4557ms
  Minsky posited that intelligence arises from the collective action of numerous simple agents.

  Variant B     score=0.492  latency=1353ms
  Minsky posited that intelligence arises from the collective action of numerous simple,
  interconnected agents rather than from any single agent's inherent intelligence.
```

| Flag | Description |
|---|---|
| `--task` | Task type: `summarize`, `classify`, `extract`, etc. |
| `--input` | Input text. Use `{input}` as placeholder in templates. |
| `--a` | First prompt template |
| `--b` | Second prompt template |
| `--backend` | `ollama` (default, local) or `openai` |

---

### `imprimer optimize`

Run Bayesian optimization (Optuna TPE) over a mutation space to find the prompt that maximizes reachability + similarity to an expected output.

Each trial costs one LLM inference call. TPE bootstraps with random exploration for the first `n/4` trials, then exploits patterns — which mutations scored highest — for the remainder.

```bash
imprimer optimize \
  --task summarize \
  --prompt "Summarize this in one sentence: {input}" \
  --input "Minsky argued intelligence emerges from many small agents" \
  --expected "Minsky argued that intelligence is an emergent property of simple agents." \
  --trials 20 \
  --backend ollama
```

Output:
```
  Running 20 optimization trials for task 'summarize'...
  Base prompt: Summarize this in one sentence: {input}

  Trials run          20
  Baseline score      0.6320  (reachability 0.6502)
  Best score          0.7140  (reachability 0.6891)
  Improvement         +0.0820

  Best prompt:
  You are an expert. Summarize this in one sentence: {input}
  Be concise.
```

| Flag | Description |
|---|---|
| `--prompt` | Base prompt template to optimize |
| `--input` | Example input for scoring |
| `--expected` | Expected output for similarity scoring |
| `--trials` | Number of optimization trials (default 20) |
| `--backend` | `ollama` or `openai` |

**Mutation space searched by the optimizer:**

| Mutation | Transformation |
|---|---|
| `concise` | Appends "Be concise." |
| `precise` | Appends "Be precise and factual." |
| `structured` | Appends "Return structured output." |
| `stepbystep` | Appends "Think step by step before answering." |
| `expert` | Prepends "You are an expert." |
| `no_fluff` | Appends "Avoid unnecessary words." |
| `rewrite_sum` | Replaces "Summarize" with "Concisely summarize" |
| `rewrite_exp` | Replaces "Explain" with "Clearly explain" |

---

### `imprimer best`

Query the registry for the prompt that achieved the highest average reachability for a given task across all historical evaluations.

```bash
imprimer best --task summarize
```

Output:
```
  Task                summarize
  Evaluations         12
  Avg reachability    0.6891
  Avg score           0.7140

  Best prompt:
  You are an expert. Summarize this in one sentence: {input}
  Be concise.
```

This is the feedback loop closing. After running `evaluate` and `optimize` multiple times, `best` surfaces what the system has learned — no manual review required.

| Flag | Description |
|---|---|
| `--task` | Task type to query |
| `--limit` | Number of recent evaluations to sample (default 10) |

---

## API reference

The CLI wraps these endpoints. Use them directly with curl or any HTTP client.

### `POST /prompt`

Evaluate two prompt variants.

```bash
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "task": "summarize",
    "input": "Your input text",
    "variant_a": "Summarize this: {input}",
    "variant_b": "You are an expert. Summarize this: {input}",
    "backend": "ollama"
  }'
```

```json
{
  "trace_id": "9bc004ea-...",
  "winner": "b",
  "output_a": "...",
  "output_b": "...",
  "latency_a_ms": 4557.78,
  "latency_b_ms": 1353.53,
  "score_a": 0.456,
  "score_b": 0.492
}
```

### `POST /optimize`

Run Bayesian optimization.

```bash
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "summarize",
    "base_prompt": "Summarize this in one sentence: {input}",
    "input_example": "Minsky argued intelligence emerges from many small agents",
    "expected_output": "Minsky argued that intelligence is an emergent property of simple agents.",
    "n_trials": 20,
    "backend": "ollama"
  }'
```

```json
{
  "best_prompt": "You are an expert. Summarize this in one sentence: {input}\nBe concise.",
  "best_score": 0.714,
  "best_reachability": 0.689,
  "baseline_score": 0.632,
  "baseline_reachability": 0.650,
  "improvement": 0.082,
  "trials_run": 20
}
```

### `GET /best`

```bash
curl "http://localhost:8080/best?task=summarize&limit=10"
```

### `GET /metrics`

Prometheus-compatible metrics endpoint.

```bash
curl http://localhost:8080/metrics
```

```
imprimer_evaluations_total{task="summarize"} 12
imprimer_avg_reachability{task="summarize"} 0.6891
imprimer_avg_judge_score{task="summarize"} 0.7340
imprimer_optimization_improvement{task="summarize"} 0.0820
```

Scrape with Prometheus and visualize in Grafana for a TensorBoard-style view of prompt improvement over time.

### `GET /health`

```bash
curl http://localhost:8080/health
# {"status":"ok","service":"imprimer-gateway"}
```

---

## Observability

Every request generates a structured JSON trace line in the engine log with a trace ID that correlates across both services:

```json
{
  "trace_id": "9bc004ea-4b8a-4dc2-860d-1bf08ac0014f",
  "task": "summarize",
  "backend": "ollama",
  "winner": "b",
  "reachability_a": 0.6421,
  "reachability_b": 0.6891,
  "score_a": 0.456,
  "score_b": 0.492,
  "latency_a_ms": 4557.78,
  "latency_b_ms": 1353.53,
  "timestamp": "2026-04-10T20:45:18Z"
}
```

The Go gateway logs every request with method, path, duration, and the same trace ID:

```
trace=9bc004ea method=POST path=/prompt duration=7.2s
```

One UUID. Complete picture across both services.

---

## Security

Every request passes through the security layer before any LLM interaction:

- **Prompt injection detection** — 9 regex patterns covering OWASP LLM Top 10 LLM01
- **PII detection** — SSN, credit card, email patterns flagged in audit log
- **Auth middleware** — Bearer token validation (set `IMPRIMER_API_KEY`)
- **Least privilege** — engine container has no write access to host filesystem

ISO 27001 alignment: A.9 (access control), A.12.6 (vulnerability management), A.14.2 (security in development).

---

## Development

### Run locally without Docker

```bash
# Terminal 1 — Python engine
cd engine
python main.py

# Terminal 2 — Go gateway
go run ./gateway/cmd/main.go

# Terminal 3 — CLI
imprimer evaluate --task summarize --input "..." --a "..." --b "..."
```

### Regenerate proto after editing `proto/imprimer.proto`

```bash
# Python
python -m grpc_tools.protoc \
  -I proto --python_out=engine --grpc_python_out=engine proto/imprimer.proto

# Go
mkdir -p gateway/gen
protoc -I proto \
  --go_out=gateway/gen --go-grpc_out=gateway/gen \
  --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative \
  proto/imprimer.proto
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ENGINE_ADDR` | `localhost:50051` | Engine address for gateway |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Model for generation |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL |
| `JUDGE_MODEL` | `$OLLAMA_MODEL` | Separate model for LLM-as-judge |
| `OPENAI_API_KEY` | — | Required for `--backend openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `IMPRIMER_API_KEY` | — | Gateway bearer token (leave empty to disable) |

---

## Roadmap

### Phase 1 — Optimization ✓
- [x] Prompt optimizer (`/optimize`) — Optuna TPE Bayesian search
- [x] LLM-as-judge scoring — optional quality signal via `--judge` flag
- [ ] Task-specific scoring weights via config

### Phase 2 — Observability ✓
- [x] Prometheus metrics (`/metrics`) — reachability, judge score, improvement
- [x] Structured JSON audit trace — correlated by trace ID across services
- [ ] Grafana dashboard for reachability trends over time
- [ ] `/trend?task=X` — reachability over time endpoint

### Phase 3 — Intelligence (next)
- [ ] LangGraph control loop — generation → evaluation → refinement graph
- [ ] Multi-agent optimization — generator, evaluator, optimizer as separate nodes
- [ ] Stateful prompt adaptation — graph cycles until reachability threshold met
- [ ] LoRA escalation — when optimizer plateaus below threshold, trigger fine-tuning

### Phase 4 — Scale
- [ ] PostgreSQL backend — replace SQLite for multi-instance deployments
- [ ] JWT authentication — scoped access per team
- [ ] Air-gapped deployment — single binary with embedded model for plant-floor use
- [ ] `imprimer ui` — TensorBoard-style dashboard reading from registry

---

## References

- *What's the Magic Word? A Control Theory of LLM Prompting* — Bhargava et al., 2023 · [arxiv.org/abs/2310.04444](https://arxiv.org/abs/2310.04444)
- *The Society of Mind* — Marvin Minsky, 1986
- *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* — Khattab et al., 2023
- OWASP Top 10 for LLM Applications · [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- Optuna: A Next-generation Hyperparameter Optimization Framework · [optuna.org](https://optuna.org)

---

## License

MIT
