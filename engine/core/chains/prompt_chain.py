from dataclasses import dataclass, field
from enum import Enum
import time
import os
import requests
import hashlib
import json


from langchain_core.prompts import PromptTemplate # to rid off i/o bounding in ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.create_logger import get_logger


logger = get_logger(__name__)


TASK_MAX_TOKENS = {
    "classify": 10,
    "extract": 50,
    "summarize": 100,
    "reasoning": 150,
    "creative_writing": 500,
    "code_generation": 300,
    "rewrite": 100,
    "roleplay": 150,
    "qa": 50,
    "translate": 150,
}



_VARIANT_CACHE = {}

class ModelBackend(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    # ... further compability is easy to add


@dataclass
class VariantResult:
    text: str
    latency_ms: float
    logprobs: list = field(default_factory=list)

def _normalize_template(template: str) -> str:
    """
    Ensures the {input} placeholder exists in the template.
    If the user forgot it, automatically append it to the end.
    """
    if "{input}" not in template:
        # Append it cleanly with a separator
        return f"{template.strip()}\n\n{{input}}"
    return template


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



def _build_huggingface_llm():
    """
    Builds and returns a reusable Hugging Face InferenceClient.
    Defaults to Zephyr or Llama-3, which are free on the Serverless API.
    """
    from huggingface_hub import InferenceClient
    
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set in your environment.")
        
    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    # We bind the model ID to the client here, so it acts just like your OpenAI llm object
    return InferenceClient(model=model_id, token=token)


def _extract_hf_api_logprobs(response) -> list:
    """
    Extracts logprobs from the Hugging Face API ChatCompletion response.
    Returns [] safely if the response doesn't contain logprob data or if
    the model/provider doesn't support logprobs.
    """
    try:
        lp_content = response.choices[0].logprobs.content
        if not lp_content:
            return []
            
        return [
            {
                "token": td.token,
                "logprob": td.logprob,
                "top": [
                    {"token": t.token, "logprob": t.logprob}
                    for t in getattr(td, "top_logprobs", [])
                ],
            }
            for td in lp_content
        ]
    except (AttributeError, KeyError, TypeError, IndexError):
        return []


def _run_ollama(prompt_text: str, temperature: float = 0.0, max_tokens: int = 150) -> VariantResult:
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
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 50,
            "num_predict": max_tokens
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


def run_variants_parallel(
    templates: list[str],
    input_text: str,
    task: str,
    backend: ModelBackend,
    max_workers: int = 4,
    temperature: float = 0.0,
) -> list[VariantResult]:
    """
    Executes multiple prompt variants in parallel using threads.
    Bypasses GIL since requests.post is I/O bound.
    """
    results = [None] * len(templates)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs to the thread pool
        future_to_index = {
            executor.submit(run_variant, tpl, input_text, task, backend, temperature): idx 
            for idx, tpl in enumerate(templates)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"parallel variant failed: {e}")
                # Fallback to a failed result so the list maintains order
                results[idx] = VariantResult(text="", latency_ms=0, logprobs=[])
                
    return results


def run_variant(
    template: str,
    input_text: str,
    task: str,
    backend: ModelBackend,
    temperature: float = 0.0,
) -> VariantResult:
    """
    Runs one prompt variant and returns what the model produced.

    Backend selection becomes purely a data sovereignty decision:
      OLLAMA      - data never leaves the machine, full logprobs
      OPENAI      - external API, full logprobs, stronger base model
      HUGGINGFACE - external API, logprobs when supported by provider

    """
    cache_state = json.dumps({
        "template": template, 
        "input_text": input_text, 
        "task": task, 
        "backend": backend.value,
        "temperature": temperature,
    }, sort_keys=True)

    key = hashlib.sha256(cache_state.encode('utf-8')).hexdigest()
    
    if key in _VARIANT_CACHE:
        return _VARIANT_CACHE[key]
    
    safe_template = _normalize_template(template) # normalizing to ensure {input} exists

    prompt = PromptTemplate(
        template=safe_template,
        input_variables=["task", "input"]
    )

    max_tokens = TASK_MAX_TOKENS.get(task, 150)
    # Render the prompt template to a plain string for Ollama
    # (Ollama's /api/generate expects a string, not a message list)
    rendered = prompt.format(task=task, input=input_text)

    if backend == ModelBackend.OLLAMA:
        result = _run_ollama(prompt_text=rendered, temperature=temperature, max_tokens=max_tokens)

    elif backend == ModelBackend.OPENAI:
        llm = _build_openai_llm()
        chain = prompt | llm
        start = time.time()
        response = chain.invoke({"task": task, "input": input_text})
        elapsed_ms = (time.time() - start) * 1000
        result = VariantResult(
            text=response.content,
            latency_ms=round(elapsed_ms, 2),
            logprobs=_extract_openai_logprobs(response),
        )


    elif backend == ModelBackend.HUGGINGFACE:
        client = _build_huggingface_llm()
        
        start = time.time()
        response = client.chat_completion(
            messages=[{"role": "user", "content": rendered}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )
        elapsed_ms = (time.time() - start) * 1000
        
        text = response.choices[0].message.content
        
        result = VariantResult(
            text=text,
            latency_ms=round(elapsed_ms, 2),
            logprobs=_extract_hf_api_logprobs(response),
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # caching to avoid trials and iterations
    _VARIANT_CACHE[key] = result
    return result