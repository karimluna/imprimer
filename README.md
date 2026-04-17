<p align="center">
  <h1 align="center">Imprimer: Control and Optimization for LLMs</h1>
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



> *"To imprint a mental pattern."*
>
> Inspired by Minsky's *The Society of Mind* (1986): a prompt does not instruct a unified intelligence but it **activates a specific configuration** of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable over time.



## What it does

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**.

Given a task and two prompt variants, Imprimer asks: which prompt gives you more control over the model's output distribution? It measures this with a **Reachability Index** grounded in the paper *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023) which is the first mathematically rigorous analysis of prompt controllability over autoregressive sequence models.

Every evaluation is persisted. Over time, the system learns which prompts control each task most effectively and surfaces that knowledge through the `best` command and `/best` endpoint.



## Theoretical foundation

The LLM defines a stochastic dynamical system over token sequences. A prompt acts as a **control input** $u$ that steers the generation trajectory toward a desired output $y^*$.

Following the intuition of reachability from control theory, the key question is:

> *Can the model produce the desired output without fighting its own prior distribution?*


### Token-level reachability

For each generated token, we evaluate whether it lies within the model’s **high-probability region**.

Let $\ell = \text{logprob}$ of the chosen token. We define a soft reachability score using a sigmoid centered at a probability threshold:

$$
r = \sigma\big(\alpha (\ell - \tau)\big)
$$

Where:

* $\tau = \log(0.1)$ → token must have ≥ 10% probability to be considered “naturally reachable”
* $\alpha$ → controls how sharply we separate reachable vs unreachable tokens

This means that if $r \approx 1 \rightarrow$ token lies in a high probability region (token is unlikely) and $r \approx 0 \rightarrow$ token is unlikely (prompt is forcing the model).



### Sequence-level reachability

The **Reachability Index** is the average over all tokens:

$$
R = \frac{1}{T} \sum_{t=1}^{T} r_t
$$


### Interpretation

| Score      | Meaning                                                          |
| ---------- | ---------------------------------------------------------------- |
| `~1.0`     | Output follows the model’s natural trajectory (highly reachable) |
| `~0.6–0.8` | Good alignment between prompt and model prior                    |
| `~0.3–0.5` | Partial control, prompt is fighting the model                    |
| `<0.3`     | Output is largely unnatural for the model                        |


### Connection to theory

In the reachability framework (Bhargava et al.), a target $y^*$ is reachable if the prompt has enough **control capacity** to overcome the model’s default trajectory. In practice, we cannot directly measure the geometric condition:

$$
|y^\perp| \leq k \cdot \gamma
$$

So we approximate it empirically:

* If the model assigns **high probability** to the generated tokens  $\rightarrow y^*$ lies within the reachable region
* If the model assigns **low probability** $\rightarrow $ the prompt is exceeding its control budget



### Optimization objective

Given a task $x_0$, Imprimer optimizes prompts $u$ to maximize semantic alignment with $y^*$ while ensuring the generated outputs remain within the model’s high-probability (reachable) region.

<p align="center">
  <img src="docs/assets/llmcontrol.drawio.png" height="240" alt="LLMs control framework">
</p>

## Architecture 

Imprimer is two services connected by a gRPC contract. The proto file is the single source of truth, so in that way Go and python never share code, only the contract. The following diagram provides a high level visualization of the information flow in the platform, the orchestator cycle integrates the **reflective agent** pattern. The reflection (evaluation) node analyzes outputs and returns a feedback sigal, creating the improvement loop.

<p align="center">
  <img src="docs/assets/show-arch.drawio.png" height="350" alt="LLMs control framework">
</p>

**Go handles:** HTTP ingress, authentication, audit logging, Prometheus metrics, gRPC routing. Go's goroutine model handles concurrent LLM requests efficiently.

**Python handles:** LLM inference (Ollama or OpenAI), logprob extraction, reachability computation, LLM-as-judge scoring, Optuna optimization, injection scanning, registry persistence.

**Boundary:** `proto/imprimer.proto`, three RPCs, never more complexity than needed.

A Command Line Interface is integrated in the system for immediate use, its functionalities with examples are at [Imprimer CLI](./docs/cli-imprimer.md).



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

## Security

Every request passes through the security layer before any LLM interaction:

- **Prompt injection detection**: 9 __regex__ patterns covering OWASP LLM Top 10 LLM01
- **PII detection**: SSN, credit card, email patterns flagged in audit log
- **Auth middleware**: Bearer token validation (set `IMPRIMER_API_KEY`)
- **Least privilege**: engine container has no write access to host filesystem

ISO 27001 alignment: A.9 (access control), A.12.6 (vulnerability management), A.14.2 (security in development).

## Development

### Run locally without Docker

```bash
# Terminal 1: Python engine
cd engine
python main.py

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


## Motivation

- *What's the Magic Word? A Control Theory of LLM Prompting*, Bhargava et al., 2023 · [arxiv.org/abs/2310.04444](https://arxiv.org/abs/2310.04444)
- *The Society of Mind*, Marvin Minsky, 1986
- OWASP Top 10 for LLM Applications · [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## Still in Developmetn
User interface is going to be like TensorBoard but is still in development so `imprimer ui` does not exist yet. Fine-tuning layer can be added after we ensure that graph and bayesian optimization plateaus for complex tasks.

## License

MIT
