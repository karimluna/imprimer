"""
Unified inference layer.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.create_logger import get_logger

logger = get_logger(__name__)

TASK_MAX_TOKENS: dict[str, int] = {
    "classify":          10,
    "extract":           50,
    "summarize":        100,
    "reasoning":        150,
    "creative_writing": 500,
    "code_generation":  300,
    "rewrite":          100,
    "roleplay":         150,
    "qa":                50,
    "translate":        150,
}

_CACHE_MAX = 512
_VARIANT_CACHE: OrderedDict = OrderedDict()


class ModelBackend(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class VariantResult:
    text: str
    latency_ms: float
    logprobs: list = field(default_factory=list)


def _build_chat_client(
    backend: ModelBackend,
    model_env_var: str,
    default_model: str,
    temperature: float = 0.0,
    max_tokens: int = 150,
    with_logprobs: bool = True,
) -> ChatOpenAI:
    logprob_kwargs = {"logprobs": True, "top_logprobs": 5} if with_logprobs else {}

    if backend == ModelBackend.OLLAMA:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return ChatOpenAI(
            model=os.getenv(model_env_var, default_model),
            base_url=f"{base}/v1",
            api_key="ollama",
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **logprob_kwargs,
        )

    if backend == ModelBackend.OPENAI:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **logprob_kwargs,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def _extract_logprobs(response) -> list:
    try:
        lp_content = response.response_metadata.get("logprobs", {})
        if not lp_content or "content" not in lp_content:
            return []
        return [
            {
                "token":  td["token"],
                "logprob": td["logprob"],
                "top": [
                    {"token": t["token"], "logprob": t["logprob"]}
                    for t in td.get("top_logprobs", [])
                ],
            }
            for td in lp_content["content"]
        ]
    except (AttributeError, KeyError, TypeError):
        return []


def _render_prompt(template: str, task: str, input_text: str) -> str:
    if "{input}" in template and "{task}" in template:
        return template.format(task=task, input=input_text)
    if "{input}" in template:
        return template.replace("{input}", input_text)
    if "{task}" in template:
        return template.replace("{task}", task)
    return template


def run_variant(
    template: str,
    input_text: str = "",
    task: str = "",
    backend: ModelBackend = ModelBackend.OLLAMA,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> VariantResult:
    """
    Evaluator model (OLLAMA_MODEL / small, fast, logprobs).
    Used for GRPO scoring, evaluator promotion, baseline, stability analysis.
    Cache is LRU-bounded to 512 entries.
    """
    cache_key = hashlib.sha256(
        json.dumps(
            {"tpl": template, "inp": input_text, "task": task,
             "be": backend.value, "temp": temperature},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if use_cache and temperature == 0.0 and cache_key in _VARIANT_CACHE:
        _VARIANT_CACHE.move_to_end(cache_key)
        return _VARIANT_CACHE[cache_key]

    rendered   = _render_prompt(template, task, input_text)
    max_tokens = TASK_MAX_TOKENS.get(task, 150)

    try:
        llm = _build_chat_client(
            backend,
            model_env_var="OLLAMA_MODEL",
            default_model="llama3.2:1b",
            temperature=temperature,
            max_tokens=max_tokens,
            with_logprobs=True,
        )
        start    = time.time()
        response = llm.invoke([HumanMessage(content=rendered)])
        elapsed  = round((time.time() - start) * 1000, 2)
        result   = VariantResult(
            text=response.content.strip(),
            latency_ms=elapsed,
            logprobs=_extract_logprobs(response),
        )
    except Exception as exc:
        logger.error(f"run_variant failed backend={backend.value} task={task}: {exc}")
        result = VariantResult(text="", latency_ms=0.0, logprobs=[])

    if use_cache and temperature == 0.0:
        _VARIANT_CACHE[cache_key] = result
        if len(_VARIANT_CACHE) > _CACHE_MAX:
            _VARIANT_CACHE.popitem(last=False)

    return result


def run_variants_parallel(
    templates: list[str],
    input_text: str,
    task: str,
    backend: ModelBackend,
    max_workers: int = 4,
    temperature: float = 0.0,
) -> list[VariantResult]:
    results: list[VariantResult | None] = [None] * len(templates)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_variant, tpl, input_text, task, backend, temperature): idx
            for idx, tpl in enumerate(templates)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"parallel variant {idx} failed: {e}")
                results[idx] = VariantResult(text="", latency_ms=0.0, logprobs=[])
    return results  # type: ignore[return-value]


def call_llm(
    prompt_text: str,
    backend: ModelBackend,
    temperature: float = 0.3,
    max_tokens: int = 300,
) -> str:
    """
    Generator model (GENERATOR_MODEL / stronger, no logprobs).
    Called exactly once per cycle for variant generation.
    Raises on failure, caller decides how to handle.
    """
    try:
        llm = _build_chat_client(
            backend,
            model_env_var="GENERATOR_MODEL",
            default_model="qwen2.5:1.5b",
            temperature=temperature,
            max_tokens=max_tokens,
            with_logprobs=False,
        )
        return llm.invoke([HumanMessage(content=prompt_text)]).content.strip()
    except Exception as exc:
        logger.error(f"call_llm (generator) failed backend={backend.value}: {exc}")
        raise