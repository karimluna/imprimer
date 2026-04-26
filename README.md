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


Most prompt engineering is trial and error. Imprimer treats it as a **control problem**: given a prompt and a task, which configuration of the model's output distribution gives you the most control? It measures this with a **Reachability Index**, derived from token-level logprobs and optimizes it automatically using GRPO and RiOT residuals. Every result is persisted; the system learns which prompts control each task most

## Theoretical Foundation

An LLM is a stochastic dynamical system over token sequences. A prompt is a **control input** $u$ steering generation toward a desired output $y^*$. The question Imprimer answers: *can the model produce the desired output without fighting its own prior distribution?*

### Token-level reachability

For each generated token with logprob $\ell$, a soft reachability score via sigmoid:

$$r = \sigma\big(\alpha (\ell - \tau)\big)$$

- $\tau = \log(0.40)$ — token needs ≥ 40% probability mass to be naturally reachable  
- $\alpha$ — sharpness of the reachable / unreachable boundary

$r \approx 1$: token sits in the model's high-probability region. $r \approx 0$: the prompt is fighting the model.

### Sequence-level Reachability Index

$$R = \frac{1}{T} \sum_{t=1}^{T} r_t$$

| Score | Meaning |
|---|---|
| `~1.0` | Follows the model's natural trajectory |
| `0.6 – 0.8` | Good prompt-model alignment |
| `0.3 – 0.5` | Partial control, fighting the model |
| `< 0.3` | Largely unnatural output |

### GRPO reward shaping (ELPR)

Rather than a binary pass/fail threshold, the optimizer applies **Exponential Linear Proximity Reward** relative to the group mean across candidates:

$$\text{ELPR}(s, \bar{s}) = \sigma\big(\beta (s - \bar{s})\big)$$

$\bar{s}$ is the mean reachability across the current candidate group, a value-function-free baseline. No critic model needed. A variant just above the group mean scores ~0.55; a strong outlier approaches 1.0. The signal is continuous and never saturates.

### Scoring formula

$$\text{score} = 0.60 \times R + 0.28 \times \text{quality} + 0.12 \times \text{latency}$$

Reachability carries the primary weight as the theoretically grounded control signal. Quality is task-aware: cosine similarity to a reference for deterministic tasks (classify, extract, qa), lexical diversity + length heuristic for open-ended tasks (summarize, reason, code, creative). Latency penalizes responses over 1 second.


## Architecture

Two services connected by gRPC. The proto file is the single source of truth — Go and Python share no code, only the contract.

<p align="center">
  <img src="docs/assets/architecture.png" height="350" alt="Architecture diagram">
</p>

| Layer | Responsibility |
|---|---|
| **Go** | HTTP ingress, auth, audit logging, Prometheus metrics, gRPC routing |
| **Python** | LLM inference, logprob extraction, reachability scoring, GRPO optimization, RiOT residual extraction, registry persistence |
| **Contract** | `proto/imprimer.proto` — three RPCs, minimal surface |

Both Ollama and OpenAI route through a single `ChatOpenAI` factory. Ollama exposes an OpenAI-compatible API at `/v1`, so there is no branching in the inference layer.


## Optimization Loop

The optimizer runs a LangGraph cycle: `generator → evaluator → controller → (loop | done)`.

<p align="center">
  <img src="docs/assets/llmcontrol.png" height="240" alt="LLM control framework">
</p>

**Generator: GRPO step.**
Generates `n_variants` candidate prompts anchored to the current best. Injects the RiOT residual, structural constraints extracted from prior winning prompts so provenn constraints are never overwritten. Scores all candidates in parallel via logprob reachability and task similarity, applies ELPR group-relative reward shaping, and returns the winner.

**Evaluator: promotion and reflection.**
The winner's score is a cache hit (GRPO already executed this call at temp=0 so cost is zero). The evaluator promotes the candidate to global best if its reachability exceeds the current record, extracts a new residual from the winning prompt, and generates a two-sentence verbal reflection explaining what changed and why. That reflection feeds directly into the next generation step.

**Controller: termination.**
Increments the cycle counter. Stops when reachability ≥ target or the cycle cap is reached.

### RiOT Residual Connection

Across cycles, a prompt can undergo **semantic drift** — improving one dimension inadvertently overwrites constraints that were working elsewhere. RiOT prevents this by extracting structural lines from the winning prompt (output format rules, persona anchors, priming instructions) and explicitly preserving them in the next generation prompt.

```
Cycle 1 → Winner A → extract residual R(A)
Cycle 2 → Generate with R(A) injected → Winner B  (drift-free)
Cycle 3 → Generate with R(B) injected → ...
```

### Call budget per cycle

| Step | Calls | Execution |
|---|---|---|
| Variant generation | 1 `call_llm` | serial |
| GRPO scoring | N `run_variant` | parallel |
| Evaluator score | 0 (cache hit) | - |
| Feedback reflection | 1 `call_llm` | serial |
| **Total (N = 2)** | **4** | **3 wall-clock steps** |



## In Practice

Qwen2.5:1.5b (no fine-tuning) classifying spam, the optimizer discovers the optimal prompt autonomously across cycles, preserving proven structure while exploring new directions.

<p align="center">
  <img src="docs/examples/optimization-imprimer.png" height="400" width="360" alt="Optimization run"/>
  &nbsp;
  <img src="docs/examples/stability-imprimer.png" height="400" width="360" alt="Stability analysis"/>
</p>

The **Stability** tab samples the same prompt N times at temperature > 0 and reports reachability, pairwise output similarity, and variance, a diagnostic view of how consistently a prompt steers the model before committing to an optimization run.


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

- **Dashboard**: TensorBoard-style UI reading from the prompt registry
- **Fine-tuning escalation**: LoRA trigger when the optimizer plateaus on complex tasks  
- **Model routing**: automatic cascade to a larger model when reachability is consistently low across cycles



## References

- [What's the Magic Word? A Control Theory of LLM Prompting](https://arxiv.org/abs/2310.04444), Bhargava et al., 2023
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300), DeepSeek, 2024
- [RiOT: Residual Iterative Optimization for Text](https://arxiv.org/abs/2504.12345), arXiv, 2025

*The synthesis and application of these ideas to prompt control are original from my analysis.*