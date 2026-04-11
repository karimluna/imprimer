"""
LLM-as-judge scores output quality on a 0.0-1.0 scale.
"""
import hashlib
import json
import re
import os
import requests

from core.chains.prompt_chain import ModelBackend, _build_openai_llm
from utils.create_logger import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT = """You are an impartial evaluator. Score the following AI output.

Task type: {task}
Input given to the AI: {input}
AI output to evaluate: {output}

Score the output on these three dimensions from 0.0 to 1.0:
- accuracy: does the output correctly address the input for this task?
- completeness: does it cover what the task requires without missing key points?
- conciseness: does it avoid unnecessary words, repetition, or padding?

Rules:
- 1.0 means perfect, 0.0 means completely wrong or missing
- Be strict but fair
- Respond with ONLY a valid JSON object, nothing else

Your response:
{{"accuracy": <score>, "completeness": <score>, "conciseness": <score>}}"""


def _cache_key(task: str, input_text: str, output: str) -> str:
    raw = f"{task}||{input_text}||{output}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_scores(text: str) -> dict:
    """
    Extracts the JSON score object from the judge's response.
    Handles markdown code fences that small models tend to add
    despite being told not to.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    try:
        match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback, extract individual values by key name
    scores = {}
    for key in ("accuracy", "completeness", "conciseness"):
        pattern = rf'"{key}"\s*:\s*([0-9.]+)'
        match = re.search(pattern, cleaned)
        scores[key] = float(match.group(1)) if match else 0.5

    return scores


def _run_judge_ollama(prompt_text: str) -> str:
    """
    Runs the judge prompt through Ollama.
    Reuses the same /api/chat call pattern as _run_ollama in prompt_chain.py
    but without logprobs  the judge only needs the text response.
    """
    base_url = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {
            "temperature": 0,
        }
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _run_judge_openai(prompt_text: str) -> str:
    """
    Runs the judge prompt through OpenAI.
    Uses _build_openai_llm from prompt_chain no logprobs needed.
    """
    llm = _build_openai_llm()
    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else str(response)


# Module-level cache survives across calls within one engine process
_judge_cache: dict[str, float] = {}


def judge(
    task: str,
    input_text: str,
    output: str,
    backend: ModelBackend,
) -> float:
    """
    Scores output quality using a second LLM call as an impartial judge.
    Returns a float between 0.0 and 1.0.

    Scoring weights:
      accuracy      50% getting the answer right is most important
      completeness  30% covering the required content
      conciseness   20% avoiding unnecessary verbosity

    Cache: results are cached by SHA256(task+input+output) so the same
    output is never judged twice in the same engine process.
    """
    cache_key = _cache_key(task, input_text, output)

    if cache_key in _judge_cache:
        cached = _judge_cache[cache_key]
        logger.debug(f"judge cache hit score={cached:.4f}")
        return cached

    prompt_text = JUDGE_PROMPT.format(
        task=task,
        input=input_text,
        output=output,
    )

    try:
        if backend == ModelBackend.OLLAMA:
            raw_text = _run_judge_ollama(prompt_text)
        else:
            raw_text = _run_judge_openai(prompt_text)

        scores = _parse_scores(raw_text)

        accuracy     = max(0.0, min(1.0, float(scores.get("accuracy",     0.5))))
        completeness = max(0.0, min(1.0, float(scores.get("completeness", 0.5))))
        conciseness  = max(0.0, min(1.0, float(scores.get("conciseness",  0.5))))

        combined = round(
            0.50 * accuracy +
            0.30 * completeness +
            0.20 * conciseness,
            4
        )

        logger.info(
            f"judge task={task} "
            f"accuracy={accuracy:.2f} "
            f"completeness={completeness:.2f} "
            f"conciseness={conciseness:.2f} "
            f"combined={combined:.4f}"
        )

        _judge_cache[cache_key] = combined
        return combined

    except Exception as e:
        # Judge failure must never crash the evaluation pipeline.
        # Return neutral 0.5 so scoring continues without the judge signal.
        logger.warning(f"judge failed, returning neutral 0.5: {e}")
        return 0.5