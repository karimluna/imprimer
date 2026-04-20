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

> *"To imprint a mental pattern."*
>
> Inspired by Minsky's *The Society of Mind* (1986): a prompt does not instruct a unified intelligence but it **activates a specific configuration** of the model's internal society. Imprimer makes that activation measurable, comparable, and improvable over time. 

---

## What it does

Most prompt engineering is trial and error. Imprimer treats it as a **control problem**.

Given a task and two prompt variants, Imprimer asks: which prompt gives you more control over the model's output distribution? It measures this with a **Reachability Index** grounded in the paper *"What's the Magic Word? A Control Theory of LLM Prompting"* (Bhargava et al., 2023), the first mathematically rigorous analysis of prompt controllability over autoregressive sequence models.

Every evaluation is persisted. Over time, the system learns which prompts control each task most effectively and surfaces that knowledge through the `best` command and `/best` endpoint.

A __demo__ for the Reflective Optimizaiton Evolution version is available at [imprimer](https://balor78-imprimer.hf.space/), because the project requires a large number of LLM calls I recommend using local models and running with `python -m demo.app` or the CLI directly.

## Theoretical foundation

The LLM can be defined as a stochastic dynamical system over token sequences. A prompt acts as a **control input** $u$ that steers the generation trajectory toward a desired output $y^*$.

Following the intuition of reachability from control theory, the key question is:

> *Can the model produce the desired output without fighting its own prior distribution?*

### Token-level reachability

For each generated token, we evaluate whether it lies within the model's **high-probability region**.

Let $\ell = \text{logprob}$ of the chosen token. We define a soft reachability score using a sigmoid centered at a probability threshold:

$$r = \sigma\big(\alpha (\ell - \tau)\big)$$

Where:
- $\tau = \log(0.1)$ → token must have ≥ 10% probability to be considered naturally reachable
- $\alpha$ → controls how sharply we separate reachable vs unreachable tokens

This means $r \approx 1 \rightarrow$ token lies in a high probability region and $r \approx 0 \rightarrow$ token is unlikely, the prompt is fighting the model.

### Sequence-level reachability

The **Reachability Index** is the average over all tokens:

$$R = \frac{1}{T} \sum_{t=1}^{T} r_t$$

### Interpretation

| Score | Meaning |
|---|---|
| `~1.0` | Output follows the model's natural trajectory (highly reachable) |
| `~0.6–0.8` | Good alignment between prompt and model prior |
| `~0.3–0.5` | Partial control, prompt is fighting the model |
| `<0.3` | Output is largely unnatural for the model |

### Connection to theory

In the reachability framework (Bhargava et al.), a target $y^*$ is reachable if the prompt has enough **control capacity** to overcome the model's default trajectory. We approximate this empirically: if the model assigns high probability to the generated tokens, $y^*$ lies within the reachable region. If the model assigns low probability, the prompt is exceeding its control budget.

### Optimization objective

Given a task $x_0$, Imprimer optimizes prompts $u$ to maximize semantic alignment with $y^*$ while ensuring generated outputs remain within the model's high-probability (reachable) region.

<p align="center">
  <img src="docs/assets/llmcontrol.drawio.png" height="240" alt="LLMs control framework">
</p>

## Architecture

Imprimer is two services connected by a gRPC contract. The proto file is the single source of truth, Go and Python never share code, only the contract. The diagram below shows the information flow. The orchestrator cycle integrates the **reflective agent** pattern: the reflection (evaluation) node analyzes outputs and returns a feedback signal, creating the improvement loop.

<p align="center">
  <img src="docs/assets/show-arch.drawio.png" height="350" alt="Architecture diagram">
</p>

**Go handles:** HTTP ingress, authentication, audit logging, Prometheus metrics, gRPC routing.

**Python handles:** LLM inference (Ollama, OpenAI, HuggingFace), logprob extraction, reachability computation, LLM-as-judge scoring, Optuna/RPE optimization, injection scanning, registry persistence.

**Boundary:** `proto/imprimer.proto`, three RPCs, never more complexity than needed.

A Command Line Interface is integrated for immediate use. See [Imprimer CLI](./docs/cli-imprimer.md).

## Controlling Small Models
Here we can see an example of the capabilities of the platform in simple tasks, for example using Qwen2.5:1.5B (a really tiny model) in Ollama without fine-tuning, we get a model that can classify an email as spam with confidence, we first run optimization with some prompt, the system is in charge of getting the best prompt via __Reflective Prompt Evolution__!. In the right image we can see how now it classifies the email as spam.

<p align="center">
  <img src="docs/examples/optimization-imprimer.png" height="400" alt="optimization" width="360" />
  &nbsp;
  <img src="docs/examples/stability-imprimer.png" height="400" alt="stability analysis" width="360"/>
</p>   

#### Scoring function

Imprimer uses a **task-aware, backend-adaptive scoring function** that routes through four scenarios depending on what signals are available.


## Optimization

Imprimer has two optimization paths depending on the interface:

### CLI path: Bayesian search (Optuna TPE)

Used when calling `imprimer optimize` from the command line. Searches over structured linguistic mutations of the base prompt using spaCy dependency parsing:

- **VerbMutator**: rewrites the root verb (`summarize` → `distill`, `condense`, `extract`)
- **NounMutator**: rewrites the primary object noun chunk
- **ModalityMutator**: shifts surface mood (imperative → directive → interrogative)

Mutations are searched one dimension at a time across graph iterations, preventing combinatorial explosion. Optuna's TPE sampler builds a surrogate model over the mutation space and exploits patterns across trials.

### UI path: Reflective Prompt Evolution (RPE)

Used in the Gradio interface. Instead of predefined mutations, the LLM generates its own variant prompts based on the current best and verbal feedback from prior rounds, an open-ended search that can discover transformations spaCy mutations cannot.


**Semantic Self-Consistency (SSC):** run the same prompt K times at temperature > 0 and measure average pairwise semantic similarity of the K outputs. High SSC → the prompt reliably steers the model to similar outputs. Low SSC → the prompt leaves too much to chance.

Both paths use the same **LangGraph outer loop**: generator → evaluator → controller, cycling until the score exceeds the baseline by a margin or the iteration cap is hit.


## API call cost

Actual cost per optimization run depends on the path, backend, and settings.

### CLI path (Bayesian, default `--trials 20`, `--max-iterations 3`)

| Step | Calls per iteration | 3 iterations total |
|---|---|---|
| Optuna trials | 20 | 60 |
| Evaluator (graph node) | 1 | 3 |
| Feedback | 1 | 3 |
| **Total** | **22** | **~66** |

### UI path (RPE, default `n_variants=3`, `ssc_runs=2`, `max_iterations=3`)

| Step | Calls per iteration | 3 iterations total |
|---|---|---|
| Variant generation (all N in 1 call) | 1 | 3 |
| SSC scoring (N × K, executed in parallel) | N×K | N×K×3 |
| Evaluator (graph node) | 1 | 3 |
| Feedback | 1 | 3 |
| **Total (N=3, K=2)** | **9** | **~27** |

### Cost by backend (approximate, per 1000 tokens)

| Backend | Cost | Logprobs | Notes |
|---|---|---|---|
| **Ollama (local)** | Free | ✅ Full | Requires local GPU or CPU inference. `qwen2.5:1.5b` runs on CPU. |
| **OpenAI `gpt-4o-mini`** | ~\$0.15 input / \$0.60 output per 1M tokens | ✅ Full | UI path: ~27 calls × ~200 tokens avg = ~\$0.001–0.003 per optimization run |
| **HuggingFace Inference API** | Free tier: rate-limited. Pro: ~\$9/month | ❌ None | No logprobs → reachability falls back to similarity. SSC still works. |

### Cost reduction tips

- Reduce `--trials` to 10 for faster CLI runs (loses optimizer precision)
- Reduce `n_variants` to 2 and `ssc_runs` to 1 in the UI (loses SSC reliability)
- Use `--max-iterations 1` when the base prompt is already reasonable
- With Ollama, cost is zero but latency scales with concurrent threads
```

### Key Changes Made in this Block:
1. **Changed `n_variants=6` to `n_variants=3`** to match your Gradio UI defaults.
2. **Updated the Math:** `1 + (3*2) + 1 + 1 = 9` calls per iteration. `9 * 3 = 27` total calls.
3. **Added "executed in parallel"** to the SSC scoring note, which is a great selling point now that we refactored it.
4. **Updated the OpenAI cost estimate** to reflect the ~27 calls of the UI path instead of mixing it with the 66 calls of the CLI path.

## Quickstart

### Prerequisites

- Docker Desktop
- Ollama with `qwen2.5:1.5b`: `ollama pull qwen2.5:1.5b`
- Ollama listening on all interfaces (required for Docker):

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
```

Or build locally:

```bash
go install ./gateway/cmd/imprimer/
```


## Security

Every request passes through the security layer before any LLM interaction:

- **Prompt injection detection**: 9 regex patterns covering OWASP LLM Top 10 LLM01
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


## Still in development

The user interface (`imprimer ui`) is planned as a TensorBoard-style dashboard reading directly from the registry, still in development. Fine-tuning escalation (LoRA when the optimizer plateaus) is the next planned addition after the optimization loop stabilizes on complex tasks.


## Motivation

- *What's the Magic Word? A Control Theory of LLM Prompting*, Bhargava et al., 2023 · [arxiv.org/abs/2310.04444](https://arxiv.org/abs/2310.04444)
- *The Society of Mind*, Marvin Minsky, 1986
- OWASP Top 10 for LLM Applications · [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)


## License

MIT
