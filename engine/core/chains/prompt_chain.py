from dataclasses import dataclass, field
from enum import Enum
import time
import os
import math
import requests

from langchain_core.prompts import PromptTemplate


class ModelBackend(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    # ... further compability is easy to add


@dataclass
class VariantResult:
    text: str
    latency_ms: float
    logprobs: list = field(default_factory=list)


def _build_openai_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        logprobs=True,
        top_logprobs=5,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _extract_openai_logprobs(response) -> list:
    """
    Extracts logprobs from a LangChain OpenAI response.
    Returns a list of dicts with 'token', 'logprob', and 'top' keys.
    """
    try:
        lp_content = response.response_metadata.get("logprobs", {})
        if not lp_content or "content" not in lp_content:
            return []
        return [
            {
                "token": td["token"],
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


def _run_ollama(prompt_text: str) -> VariantResult:
    """
    Calls Ollama /api/chat with logprobs enabled.
    
    We normalize this into the same shape as the OpenAI extractor
    so the scorer receives identical input regardless of backend.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    if not model:
        raise RuntimeError(
            "OLLAMA_MODEL is not configured. Set OLLAMA_MODEL to a loaded model "
            "such as 'qwen2.5:1.5b'."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 50,
        }
    }

    start = time.time()
    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    elapsed_ms = (time.time() - start) * 1000

    data = resp.json()

    # Text lives under message.content in /api/chat
    text = data.get("message", {}).get("content", "")

    # Logprobs are at the top level 
    raw = data.get("logprobs") or []
    logprobs = [
        {
            "token": entry.get("token", ""),
            "logprob": entry.get("logprob", -10.0),
            "top": [
                {"token": t["token"], "logprob": t["logprob"]}
                for t in entry.get("top_logprobs", [])
            ],
        }
        for entry in raw
    ]

    return VariantResult(
        text=text,
        latency_ms=round(elapsed_ms, 2),
        logprobs=logprobs,
    )


def run_variant(
    template: str,
    input_text: str,
    task: str,
    backend: ModelBackend = ModelBackend.OLLAMA,
) -> VariantResult:
    """
    Runs one prompt variant and returns what the model produced.

    Backend selection becomes purely a data sovereignty decision:
      OLLAMA  - data never leaves the machine, full logprobs
      OPENAI  - external API, full logprobs, stronger base model

    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["task", "input"]
    )
    
    # Render the prompt template to a plain string for Ollama
    # (Ollama's /api/generate expects a string, not a message list)
    rendered = prompt.format(task=task, input=input_text)

    if backend == ModelBackend.OLLAMA:
        return _run_ollama(rendered)

    if backend == ModelBackend.OPENAI:
        llm = _build_openai_llm()
        chain = prompt | llm
        start = time.time()
        response = chain.invoke({"task": task, "input": input_text})
        elapsed_ms = (time.time() - start) * 1000
        return VariantResult(
            text=response.content,
            latency_ms=round(elapsed_ms, 2),
            logprobs=_extract_openai_logprobs(response),
        )

    raise ValueError(f"Unknown backend: {backend}")