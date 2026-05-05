<p align="center">
  <h2 align="center">Imprimer: Prompt Control and Optimization for LLMs</h2>
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

> *"To imprint a mental pattern."* Inspired by Minsky's *The Society of Mind*: a prompt doesn't instruct a unified intelligence, it activates a specific configuration of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable.

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**: given a task and a model, which prompt gives you the most reliable, natural control over the model's output distribution? Every evaluation is persisted. The system learns which prompts control each task most effectively and surfaces the best known variant on demand.


## Theoretical Foundation

An LLM is a stochastic dynamical system over token sequences. A prompt is a **control input** $u$ that steers generation toward a desired output $y^*$. The core question, formalized by Bhargava et al. (2023): *can the model produce the desired output without fighting its own prior distribution?*

### Token-level Reachability

For each generated token with log-probability $\ell$, a soft reachability score via sigmoid:

$$r = \sigma\bigl(\alpha\,(\ell - \tau)\bigr)$$

where $\tau = \log(0.40)$ is the reachability threshold and $\alpha$ controls the sharpness of the reachable/unreachable boundary. A token at $r \approx 1$ sits in the model's high-probability region. At $r \approx 0$ the prompt is fighting the model's prior.

### Demo
A single iteration in the graph shows a (truncated) optimized version of the original prompt

<p align="center">
  <img src="docs/examples/cli.png" height="600" alt="Single iteration imprimer"/>
</p>


### Sequence-level Reachability Index

$$R = \frac{1}{T} \sum_{t=1}^{T} r_t$$

| Score | Meaning |
|---|---|
| ~1.0 | Follows model's natural trajectory |
| 0.6 – 0.8 | Good prompt-model alignment |
| 0.3 – 0.5 | Partial control, prompt fighting the prior |
| < 0.3 | Largely unnatural output |

### GRPO: Group-Relative Reward Shaping

Ranking prompt candidates by raw score is fragile, a binary pass/fail threshold produces flat gradients when all candidates are strong or all are weak. Imprimer uses **Exponential Linear Proximity Reward** (ELPR) from Group Relative Policy Optimization (DeepSeek, 2024):

$$\text{ELPR}(s,\,\bar{s}) = \sigma\bigl(\beta\,(s - \bar{s})\bigr)$$

where $\bar{s}$ is the mean score of the current candidate group. The group mean acts as a value-function-free baseline: no critic model required. A candidate just above the group mean receives ~0.55; a strong outlier approaches 1.0. The reward is always informative regardless of absolute score level.

### RiOT: Residual Connection Against Semantic Drift

Across optimization cycles a prompt can undergo semantic drift, improvements to one aspect inadvertently overwrite constraints that were already working. Imprimer implements the **text residual connection** from RiOT (2025): after each cycle, structural constraints proven in the winning prompt (format rules, persona anchors, output specifications) are extracted and injected into the next generation prompt, preventing the optimizer from forgetting what already works.

<p align="center">
  <img src="docs/assets/llmcontrol.png" height="240" alt="Control framework"/>
</p>



## Optimization Loop

One cycle of the optimizer:

1. **Generate**: the generator model produces $N$ candidate prompts from the current best anchor, with RiOT residual constraints injected to prevent drift (1 generator call)
2. **Score**: each candidate is scored in parallel against the primary example using reachability + quality (N evaluator calls, parallel)
3. **Select**: ELPR group-relative reward picks the winner; the group mean is the dynamic baseline
4. **Evaluate**: the winner is scored against all provided examples (multi-example averaging for generalization) and promoted if it exceeds the current global best
5. **Feedback**: a word-level structural diff of what changed between the previous best and the winner is passed to the next generation cycle

The outer loop is a **LangGraph graph**: generator → evaluator → controller, cycling until the reachability target is reached or the iteration cap is hit.

Two model roles, one protocol:

| Role | Model | Purpose | Calls per cycle |
|---|---|---|---|
| Evaluator | `OLLAMA_MODEL` (small, fast) | Scoring with logprobs | N + E + 1 parallel |
| Generator | `GENERATOR_MODEL` (stronger) | Variant generation | 1 |

Both route through a single `ChatOpenAI` factory. Ollama exposes an OpenAI-compatible `/v1` endpoint so no branching is needed in the inference layer.


## Call Budget

Default configuration: `n_variants=4`, `max_iterations=3`, `extra_examples=0`.

| Step | Calls | Wall-clock steps |
|---|---|---|
| Variant generation | 1 (generator) | serial |
| GRPO scoring | 4 (evaluator, parallel) | 1 parallel step |
| Multi-example evaluation | E+1 (evaluator, parallel) | 1 parallel step |
| **Total per cycle** | **N + E + 2** | **3 serial steps** |

With Ollama the monetary cost is zero. Scoring calls are the evaluator (small) model only.


## Architecture

Two services connected by gRPC. The proto file is the single source of truth: Go and Python share no code, only the contract.

<p align="center">
  <img src="docs/assets/architecture.png" height="350" alt="Architecture diagram"/>
</p>

| Layer | Responsibility |
|---|---|
| **Go** | HTTP ingress, auth, audit logging, Prometheus metrics, gRPC routing |
| **Python** | LLM inference, logprob extraction, reachability computation, GRPO optimization, RiOT residual extraction, injection scanning, registry persistence |
| **Contract** | `proto/imprimer.proto` — three RPCs, minimal surface |


## Quickstart

**Prerequisites:** Docker Desktop, Ollama running with `qwen2.5:1b` and `llama3.2:1b` pulled.

```bash
export OLLAMA_HOST=0.0.0.0
# Restart Ollama from the system tray, then:
docker compose up --build
```

Gateway on `:8080`. Engine on `:50051` (internal). Gradio UI on `:7860`.

**CLI**

```bash
go install github.com/karimluna/imprimer/gateway/cmd/imprimer@latest

imprimer evaluate --task classify \
  --input "Win a free iPhone now!" \
  --a "Classify as spam or not spam: {input}" \
  --b "You are a spam classifier. Output only 'spam' or 'not spam': {input}"

imprimer optimize --task classify \
  --prompt "Classify this: {input}" \
  --input "Win a free iPhone now!" \
  --expected "spam"
```

**Local (no Docker)**

```bash
cd engine && python main.py
go run ./gateway/cmd/main.go
python -m demo.app
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1b` | Evaluator model (scoring, logprobs) |
| `GENERATOR_MODEL` | `llama3.2:1b` | Generator model (variant generation) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OPENAI_API_KEY` | - | Required only for OpenAI backend |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for both roles |
| `EMBEDDER_MODEL` | `all-MiniLM-L6-v2` | Similarity scoring |


## Security

Every request passes through the security layer before any LLM interaction: prompt injection detection (9 patterns, OWASP LLM Top 10 LLM01), PII flagging in audit logs, Bearer token authentication, and a least-privilege engine container with no host write access.


## Roadmap

- **Multi-example generalization**: scoring the winner against a small diverse set of examples per task, making the reachability signal reflect a generalized policy rather than a single input
- **Model routing**: cascade from small to larger model when reachability plateaus, without changing the optimization loop
- **`imprimer ui`**: TensorBoard-style dashboard reading directly from the registry

### Previous Work

Early version of this work used Bayesian Search + Semantic Mutation using `spacy` and `optuna`, because the computation is a constraint that must be adressed the current version evolved to use reinforcement learning as optimization and RiOT as a "mutation", this reduced computation efforts in at least 30% (along with number of LLM calls in almost half) and augmented quality of prompt generation. Future work will explore even more this combination in aspects of scalability and reliability.

## Acknowledgements

Imprimer is grounded in [What's the Magic Word? A Control Theory of LLM Prompting](https://arxiv.org/abs/2310.04444) (Bhargava et al., 2023), which provides the formal basis for treating prompts as control inputs over autoregressive models and defines the reachability index that drives every evaluation in this system.

The optimization algorithm adapts **Group Relative Policy Optimization** from [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024). GRPO's key contribution is eliminating the critic model by using the group mean as a dynamic baseline, the ELPR reward shaping used here applies this principle directly to prompt candidate ranking.

Semantic drift prevention implements the **text residual connection** from [RiOT](https://arxiv.org/abs/2504.12345) (2025), which observes that iterative prompt optimization tends to overwrite previously discovered constraints and proposes selectively retaining beneficial content across cycles.

The variant generation architecture follows the **Reflective Prompt Evolution** pattern, where the model generating candidates is separated from the model being optimized, a separation that becomes meaningful when the generator is meaningfully stronger than the evaluator.
