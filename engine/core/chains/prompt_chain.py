from dataclasses import dataclass, field
from enum import Enum
import time
import os

from langchain_core.prompts import PromptTemplate


class ModelBackend(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class VariantResult:
    text: str
    latency_ms: float
    # logprobs is populated only when backend=OPENAI.
    # Ollama does not yet return logprobs through LangChain,
    # so the reachability scorer falls back to 0.5 neutral when empty.
    # In the interview: "logprobs require API access — on-premise deployments
    # trade controllability measurement for data sovereignty."
    logprobs: list = field(default_factory=list)


def _build_llm(backend: ModelBackend):
    """
    Builds the LLM instance for the requested backend.
    This is the only place in the codebase that knows which
    model provider exists — everything else receives an llm object
    and calls .invoke() on it. Swapping providers requires
    changing nothing outside this function.
    """
    if backend == ModelBackend.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            logprobs=True,
            top_logprobs=5,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if backend == ModelBackend.OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=0,
            # base_url can be overridden for remote Ollama instances
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(f"Unknown backend: {backend}")


def _extract_logprobs(response, backend: ModelBackend) -> list:
    """
    Extracts token logprobs from the response metadata.
    Returns empty list if unavailable — scorer handles that gracefully.
    """
    if backend != ModelBackend.OPENAI:
        return []

    try:
        lp_content = response.response_metadata.get("logprobs", {})
        if not lp_content or "content" not in lp_content:
            return []

        return [
            {
                "token": token_data["token"],
                "logprob": token_data["logprob"],
                "top": [
                    {"token": t["token"], "logprob": t["logprob"]}
                    for t in token_data.get("top_logprobs", [])
                ],
            }
            for token_data in lp_content["content"]
        ]
    except (AttributeError, KeyError, TypeError):
        return []


def run_variant(
    template: str,
    input_text: str,
    task: str,
    backend: ModelBackend = ModelBackend.OPENAI,
) -> VariantResult:
    """
    Runs one prompt variant and returns what the model produced.

    In Minsky's framing this activates one candidate mind —
    a specific configuration of the model's internal society —
    and observes what it produces under that prompt's control input.

    backend selects the model provider:
      ModelBackend.OPENAI  — uses OpenAI API, returns logprobs for
                             full controllability scoring
      ModelBackend.OLLAMA  — uses local Ollama, no logprobs,
                             reachability scorer degrades to neutral 0.5
                             appropriate for sensitive/on-premise data

    The backend selection is the ISO 27001 data classification decision
    made explicit in code: confidential data → OLLAMA, public data → OPENAI.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["task", "input"]
    )

    llm = _build_llm(backend)
    chain = prompt | llm

    start = time.time()
    response = chain.invoke({"task": task, "input": input_text})
    elapsed_ms = (time.time() - start) * 1000

    return VariantResult(
        text=response.content,
        latency_ms=round(elapsed_ms, 2),
        logprobs=_extract_logprobs(response, backend),
    )