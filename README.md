<p align="center">
  <h2 align="center">Imprimer: Control and Optimization for LLMs</h2>
  <div>
    <p align="center">
      <img src="docs/assets/imprimer.png" height="190"/>
    </p>
  </div>
  <p align="center">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Go-1.25.6-00ADD8?logo=go" alt="Go">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/gRPC-contract--first-00897B" alt="gRPC">
  </p>
</p>


Most prompt engineering is trial and error. Imprimer treats it as a **control problem**: given two prompt variants, which gives you more control over the model's output distribution?

It measures this with a **Reachability Index** grounded in *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023)—the first rigorous analysis of prompt controllability over autoregressive models. Every evaluation is persisted; the system learns which prompts control each task most effectively, surfaced via `best` command and `/best` endpoint.

**Demo:** [imprimer](https://balor78-imprimer.hf.space/) (Reflective Prompt Evolution). For heavy use, run locally: `python -m demo.app` or use the CLI.



## Theoretical Foundation

An LLM is a stochastic dynamical system over token sequences. A prompt is a **control input** $u$ steering generation toward a desired output $y^*$. The question Imprimer answers: *can the model produce the desired output without fighting its own prior distribution?*

### Token-level reachability

For each generated token with logprob $\ell$, a soft reachability score via sigmoid:

$$r = \sigma\big(\alpha (\ell - \tau)\big)$$

- $\tau = \log(0.1)$ → token needs ≥10% probability to be naturally reachable
- $\alpha$ → sharpness of reachable/unreachable separation

$r \approx 1$: token sits in the model's high-probability region. $r \approx 0$: the prompt is fighting the model.

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

Two services connected by gRPC. The proto file is the single source of truth — Go and Python share no code, only the contract.

<p align="center">
  <img src="docs/assets/architecture.png" height="350" alt="Architecture diagram">
</p>

| Layer | Responsibility |
|---|---|
| **Go** | HTTP ingress, auth, audit logging, Prometheus metrics, gRPC routing |
| **Python** | LLM inference (Ollama/OpenAI/HuggingFace), logprob extraction, reachability computation, reflections, Optuna/RPE optimization, injection scanning, registry persistence |
| **Contract** | `proto/imprimer.proto` — three RPCs, minimal surface |

Both backends (Ollama and OpenAI) are routed through a single `ChatOpenAI` factory. Ollama exposes an OpenAI-compatible API at `/v1`, so no branching is required in the inference layer.

Both Ollama and OpenAI route through a single `ChatOpenAI` factory. Ollama exposes an OpenAI-compatible API at `/v1`, so there is no branching in the inference layer.


## Optimization Loop

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

**Prerequisites:** Docker Desktop, Ollama with `qwen2.5:1.5b` pulled.

```bash
# Ollama must listen on all interfaces for Docker networking
export OLLAMA_HOST=0.0.0.0
# Restart Ollama from the system tray, then:

docker compose up --build
```

Gateway on `:8080`. Engine on `:50051` (internal). Gradio UI on `:7860`.

**CLI**

```bash
go install github.com/BalorLC3/Imprimer/gateway/cmd/imprimer@latest

imprimer evaluate --task classify --input "Win a free iPhone now!" \
  --a "Classify as spam or not spam: {input}" \
  --b "You are a spam classifier. Output only 'spam' or 'not spam': {input}"

imprimer optimize --task classify --prompt "Classify this: {input}" \
  --input "Win a free iPhone now!" --expected "spam" --iterations 3
```

See [CLI reference](./docs/cli-imprimer.md) for all commands.

**Local (no Docker)**

```bash
cd engine && python main.py        # gRPC engine on :50051
go run ./gateway/cmd/main.go       # HTTP gateway on :8080
python -m demo.app                 # Gradio UI on :7860
```

**OpenAI backend**

```bash
export OPENAI_API_KEY=sk-...
# Set BACKEND_ID = ModelBackend.OPENAI in demo/app.py
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Model to optimize for |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `EMBEDDER_MODEL` | `all-MiniLM-L6-v2` | Similarity scoring |

| Backend | Cost | Logprobs |
|---|---|---|
| **Ollama (local)** | Free | ✅ Full |
| **OpenAI `gpt-4o-mini`** | ~$0.001–0.003 per full run | ✅ Full |


## Security

Every request passes through the security layer before any LLM interaction: prompt injection detection (9 patterns, OWASP LLM01), PII flagging in audit logs, Bearer token auth, and a least-privilege engine container with no host write access.


## Roadmap

- **`imprimer ui`**: TensorBoard-style dashboard reading from the registry
- **Fine-tuning escalation**: LoRA when optimizer plateaus on complex tasks


## References

This work draws inspiration from [What's the magic world?: A Control Theory of LLM Prompting](https://arxiv.org/abs/2310.04444) and [Optimizing Acquisition Functions](https://arxiv.org/html/2505.17151). The ideas presented here are shaped by these foundational perspectives on control, but the synthesis and applications are my own.


