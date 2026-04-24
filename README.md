<p align="center">
  <h2 align="center">Imprimer: Control and Optimization for LLMs</h2>
  <div>
    <p align="center">
      <img src="docs/assets/imprimer.drawio.png" height="190"/>
    </p>
  </div>
  <p align="center">Prompt control and observability platform for LLMs</p>
  <p align="center">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Go-1.25.6-00ADD8?logo=go" alt="Go">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/gRPC-contract--first-00897B" alt="gRPC">
  </p>
</p>

> *"To imprint a mental pattern."* — Inspired by Minsky's *The Society of Mind*: a prompt doesn't instruct a unified intelligence, it **activates a specific configuration** of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable.



## What it does

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**: given two prompt variants, which gives you more control over the model's output distribution?

It measures this with a **Reachability Index** grounded in *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023)—the first rigorous analysis of prompt controllability over autoregressive models. Every evaluation is persisted; the system learns which prompts control each task most effectively, surfaced via `best` command and `/best` endpoint.

**Demo:** [imprimer](https://balor78-imprimer.hf.space/) (Reflective Prompt Evolution). For heavy use, run locally: `python -m demo.app` or use the CLI.



## Theoretical Foundation

An LLM is a stochastic dynamical system over token sequences. A prompt is a **control input** $u$ steering generation toward desired output $y^*$. The core question: *Can the model produce the desired output without fighting its own prior distribution?*

### Token-level reachability

For each generated token with logprob $\ell$, a soft reachability score via sigmoid:

$$r = \sigma\big(\alpha (\ell - \tau)\big)$$

- $\tau = \log(0.1)$ → token needs ≥10% probability to be naturally reachable
- $\alpha$ → sharpness of reachable/unreachable separation

$r \approx 1$: token in high-probability region. $r \approx 0$: prompt is fighting the model.

### Sequence-level Reachability Index

$$R = \frac{1}{T} \sum_{t=1}^{T} r_t$$

| Score | Meaning |
|---|---|
| `~1.0` | Follows model's natural trajectory |
| `~0.6–0.8` | Good prompt-model alignment |
| `~0.3–0.5` | Partial control, fighting the model |
| `<0.3` | Largely unnatural output |

### Optimization objective

Maximize semantic alignment with $y^*$ while keeping outputs within the model's reachable region.

<p align="center">
  <img src="docs/assets/llmcontrol.drawio.png" height="240" alt="LLMs control framework">
</p>



## Architecture

Two services connected by gRPC. The proto file is the single source of truth—Go and Python share no code, only the contract. The orchestrator cycle integrates a **reflective agent** pattern: the evaluation node analyzes outputs and returns a feedback signal, creating the improvement loop.

<p align="center">
  <img src="docs/assets/show-arch.drawio.png" height="350" alt="Architecture diagram">
</p>

| Layer | Responsibility |
|---|---|
| **Go** | HTTP ingress, auth, audit logging, Prometheus metrics, gRPC routing |
| **Python** | LLM inference (Ollama/OpenAI/HuggingFace), logprob extraction, reachability computation, reflections, Optuna/RPE optimization, injection scanning, registry persistence |
| **Contract** | `proto/imprimer.proto` — three RPCs, minimal surface |

CLI integrated for immediate use. See [Imprimer CLI](./docs/cli-imprimer.md).



## Controlling Small Models

Qwen2.5:1.5b (no fine-tuning) classifying spam via **Reflective Prompt Evolution**—the system discovers the optimal prompt autonomously.

<p align="center">
  <img src="docs/examples/optimization-imprimer.png" height="400" alt="optimization" width="360" />
  &nbsp;
  <img src="docs/examples/stability-imprimer.png" height="400" alt="stability analysis" width="360"/>
</p>

Scoring is **task-aware and backend-adaptive**, routing through different strategies depending on available signals (logprobs, embeddings, etc.).



## Optimization

### CLI path: Bayesian search (Optuna TPE)

`imprimer optimize` — searches structured linguistic mutations via spaCy dependency parsing:

- **VerbMutator**: root verb rewrites (`summarize` → `distill`, `condense`)
- **NounMutator**: primary object noun chunk rewrites
- **ModalityMutator**: mood shifts (imperative → directive → interrogative)

One dimension at a time across graph iterations. Optuna's TPE builds a surrogate over the mutation space.

### UI path: Reflective Prompt Evolution (RPE)

The LLM generates its own variant prompts based on current best + verbal feedback—open-ended search discovering transformations spaCy cannot.

**Semantic Self-Consistency (SSC):** same prompt run K times at temperature > 0, measuring average pairwise semantic similarity. High SSC = reliable steering. Low SSC = too much variance.

Both paths share the same **LangGraph outer loop**: generator → evaluator → controller, cycling until score exceeds baseline or iteration cap.


## API Call Cost

### CLI path (Optuna, default `--trials 20`, `--max-iterations 3`)

| Step | Per iteration | 3 iterations |
|---|---|---|
| Optuna trials | 20 | 60 |
| Evaluator + Feedback | 2 | 6 |
| **Total** | **22** | **~66** |

### UI path (RPE, default `n_variants=3`, `ssc_runs=2`, `max_iterations=3`)

| Step | Per iteration | 3 iterations |
|---|---|---|
| Variant generation | 1 | 3 |
| SSC scoring (N×K, parallel) | 6 | 18 |
| Evaluator + Feedback | 2 | 6 |
| **Total** | **9** | **~27** |

### Cost by backend (per 1K tokens)

| Backend | Cost | Logprobs | Notes |
|---|---|---|---|
| **Ollama (local)** | Free | ✅ Full | `qwen2.5:1.5b` runs on CPU |
| **OpenAI `gpt-4o-mini`** | ~\$0.15i / \$0.60o per 1M | ✅ Full | UI: ~\$0.001–0.003 per run |
| **HuggingFace Inference** | Free (rate-limited) / ~\$9/mo | ❌ None | Falls back to similarity scoring |

**Reduce cost:** lower `--trials`, `n_variants`, `ssc_runs`, or `--max-iterations`. With Ollama, cost is zero.



## Quickstart

### Prerequisites

- Docker Desktop
- Ollama with `qwen2.5:1.5b`: `ollama pull qwen2.5:1.5b`
- Ollama on all interfaces (required for Docker):

```bash
export OLLAMA_HOST=0.0.0.0
# Then restart Ollama from system tray
```

### Start the stack

```bash
docker compose up --build
```

Gateway on `:8080`. Engine on `:50051` (internal).

### Install the CLI

```bash
go install github.com/BalorLC3/Imprimer/gateway/cmd/imprimer@latest
# Or locally:
go install ./gateway/cmd/imprimer/
```



## Security

Every request passes through the security layer before any LLM interaction:

- **Prompt injection detection**: 9 regex patterns (OWASP LLM Top 10 LLM01)
- **PII detection**: SSN, credit card, email flagged in audit log
- **Auth middleware**: Bearer token validation (`IMPRIMER_API_KEY`)
- **Least privilege**: engine container has no host write access

ISO 27001 alignment: A.9, A.12.6, A.14.2.



## Development

### Run locally without Docker

```bash
# Terminal 1: Python engine
cd engine && python main.py

# Terminal 2: Go gateway
go run ./gateway/cmd/main.go

# Terminal 3: CLI
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



## Roadmap

- **`imprimer ui`**: TensorBoard-style dashboard reading from the registry
- **Fine-tuning escalation**: LoRA when optimizer plateaus on complex tasks


## References

This work draws inspiration from [What's the magic world?: A Control Theory of LLM Prompting](https://arxiv.org/abs/2310.04444) and [Optimizing Acquisition Functions](https://arxiv.org/html/2505.17151). The ideas presented here are shaped by these foundational perspectives on control, but the synthesis and applications are my own.


