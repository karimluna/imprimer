"""
Stability analyzer, multi-run sampling with variance and confidence metrics.
"""

import math
from dataclasses import dataclass
import os

from core.chains.prompt_chain import ModelBackend
from core.evaluator.embedder import pairwise_similarity
from core.evaluator.scorer import _compute_reachability
from utils.create_logger import get_logger
import requests

logger = get_logger(__name__)

@dataclass
class TokenConfidence:
    token: str
    logprob: float
    certainty: float


@dataclass
class StabilityResult:
    outputs: list[str]
    avg_reachability: float
    variance: float
    avg_similarity: float
    stability_score: float
    token_confidence: list[TokenConfidence]  # from the first run
    recommendation: str = ""


def _run_with_temperature(
    prompt_text: str,
    backend: ModelBackend,
    temperature: float,
) -> tuple[str, list]:
    """
    Runs one inference call with a specific temperature.
    """
    if backend == ModelBackend.OLLAMA:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        # We avoid reuse the run_variant in prompt_chain.py because it 
        # has a bit difference that make more annoying integrate both
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
            }
        }

        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")
        raw = data.get("logprobs") or []
        logprobs = [
            {
                "token": e.get("token", ""),
                "logprob": e.get("logprob", -10.0),
                "top": [
                    {"token": t["token"], "logprob": t["logprob"]}
                    for t in e.get("top_logprobs", [])
                ],
            }
            for e in raw
        ]
        return text, logprobs

    elif backend == ModelBackend.OPENAI:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )
        response = llm.invoke(prompt_text)
        text = response.content

        logprobs = []
        try:
            lp = response.response_metadata.get("logprobs", {})
            if lp and "content" in lp:
                logprobs = [
                    {
                        "token": td["token"],
                        "logprob": td["logprob"],
                        "top": [
                            {"token": t["token"], "logprob": t["logprob"]}
                            for t in td.get("top_logprobs", [])
                        ],
                    }
                    for td in lp["content"]
                ]
        except Exception:
            pass

        return text, logprobs
    
    elif backend == ModelBackend.HUGGINGFACE:
        from huggingface_hub import InferenceClient
        
        token = os.environ.get("HF_TOKEN")
        model_id = os.environ.get("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
        client = InferenceClient(model=model_id, token=token)

        formatted =  f"""### Instruction: 
{prompt_text} 

### Response:
"""
        text = client.text_generation(
            formatted,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
        )

        logprobs = [] # sadly there are no logprobs from inference hf
        return text, logprobs

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _pairwise_similarity(outputs: list[str]) -> float:
    return pairwise_similarity(outputs)


def analyze(
    prompt: str,
    input_text: str,
    task: str,
    backend: ModelBackend,
    n_runs: int = 5,
    temperature: float = 0.7,
) -> StabilityResult:
    """
    Runs the prompt N times and measures output stability.
    """
    full_prompt = f"Your task is {task}\n:{prompt}\n\nInput: {input_text}" if input_text else prompt

    outputs = []
    reachabilities = []
    all_logprobs = []

    for i in range(n_runs):
        text, logprobs = _run_with_temperature(full_prompt, backend, temperature)
        reachability = _compute_reachability(logprobs)

        outputs.append(text)
        reachabilities.append(reachability)
        all_logprobs.append(logprobs)

        logger.info(
            f"stability run={i+1}/{n_runs} "
            f"reachability={reachability:.4f} "
            f"output_len={len(text)}"
        )

    avg_reachability = round(sum(reachabilities) / len(reachabilities), 4)

    # Variance of reachability across runs
    mean = avg_reachability
    variance = round(
        sum((r - mean) ** 2 for r in reachabilities) / len(reachabilities), 6
    )

    avg_similarity = _pairwise_similarity(outputs)

    # Composite stability score
    normalized_variance = min(1.0, variance / 0.1)
    stability_score = round(
        0.4 * avg_reachability +
        0.4 * avg_similarity +
        0.2 * (1.0 - normalized_variance),
        4
    )

    # Token confidence from first run - used for visualization
    token_confidence = []
    if all_logprobs:
        for entry in all_logprobs[0]:
            chosen_prob = math.exp(entry["logprob"])
            top_probs = [math.exp(t["logprob"]) for t in entry.get("top", [])]
            total = sum(top_probs) or chosen_prob
            certainty = round(chosen_prob / total, 4) if total > 0 else 0.5
            token_confidence.append(TokenConfidence(
                token=entry["token"],
                logprob=round(entry["logprob"], 4),
                certainty=certainty,
            ))

    if stability_score >= 0.80:
        recommendation = "🟢 Prompt is highly stable. Behavior is constrained and ready for production."
    elif stability_score >= 0.60:
        recommendation = "🟡 Prompt is moderately stable. Consider running an optimization cycle to tighten the constraints."
    else:
        recommendation = "🔴 Prompt is unstable. High risk of drift or hallucination. Optimization is strongly recommended."

    logger.info(
        f"stability complete "
        f"avg_reachability={avg_reachability:.4f} "
        f"variance={variance:.6f} "
        f"avg_similarity={avg_similarity:.4f} "
        f"stability_score={stability_score:.4f}"
    )

    return StabilityResult(
        outputs=outputs,
        avg_reachability=avg_reachability,
        variance=variance,
        avg_similarity=avg_similarity,
        stability_score=stability_score,
        token_confidence=token_confidence,
        recommendation=recommendation, # Make sure your StabilityResult dataclass expects this field!
    )