"""
Reflective Prompt Optimization

Replaces the Optuna TPE inner loop with LLM-driven candidate generation.
Instead of searching over predefined mutation keys, the LLM generates its own
variant prompts based on the current best and verbal feedback from prior rounds.
---
Semantic Self-Consistency (SSC): Run the same prompt K times at temperature > 0.
                                 Average pairwise semantic similarity of the K outputs.
                                 High SSC -> prompt reliably steers the model to similar outputs.
                                 Low SSC -> model is uncertain, prompt leaves too much to chance.
---
Here reachability is an optional metric. When the backend supports logprobs
(e.g., ollama and openai). Mostly logprobs are unavailable, so SSC is more stable.
"""

import json
import re
import requests
from dataclasses import dataclass, field
import os
from typing import Optional

from core.chains.prompt_chain import ModelBackend, run_variants_parallel, call_llm
from core.evaluator.scorer import (
    _compute_reachability,
    OPEN_ENDED_TASKS,
    _creative_quality_heuristic,
)
from core.evaluator.embedder import pairwise_similarity, similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)

SSC_RUNS = 2
SSC_TEMPERATURE = 0.8
N_VARIANTS = 5


@dataclass
class RPEResult:
    best_prompt: str
    best_score: float
    best_reachability: float  # 0.5 neutral when logprobs unavailable
    best_ssc: float = (
        0.5  # SSC of the winning variant; fallback control signal when logprobs absent
    )
    history: list = field(default_factory=list)


def _generate_variants(
    base_prompt: str,
    feedback: str,
    n_variants: int,
    backend: ModelBackend,
    task: str,
    current_best_prompt: Optional[str] = None,
) -> list[str]:
    """
    Asks the LLM to generate N improved variants of the anchor prompt.

    Prompt is intentionally short and concrete — small models (1.5B) fail
    on long instructions with technical strategy names. "Change one thing"
    is clearer than "Apply EXACTLY ONE micro-mutation strategy."
    """
    anchor = current_best_prompt if current_best_prompt else base_prompt

    feedback_line = f"\nPrevious feedback: {feedback}\n" if feedback else ""

    # Short, concrete, no jargon for small models
    generation_prompt = (
        f"Improve this AI prompt for the task: {task}\n\n"
        f"Current prompt:\n{anchor}\n"
        f"{feedback_line}\n"
        f"Write {n_variants} improved versions. Rules:\n"
        f"- Keep {{input}} exactly as written\n"
        f"- Change only one thing per version: wording, tone, or instruction style\n"
        f"- No explanations\n\n"
        f'Return as JSON array: ["version 1", "version 2", ...]'
    )

    raw = ""
    try:
        raw = call_llm(
            prompt_text=generation_prompt,
            backend=backend,
            temperature=0.7,
            max_tokens=400,
        )

        # Try strict JSON parse first
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if match:
            try:
                variants = json.loads(match.group())
                valid = [v for v in variants if isinstance(v, str) and v.strip()]
                if valid:
                    logger.info(f"generated {len(valid)} valid variants (JSON)")
                    return valid[:n_variants]
            except json.JSONDecodeError:
                pass

        # Fallback: extract quoted strings line by line
        # Handles cases where the model returns "1. "prompt here"" or similar
        quoted = re.findall(r'"([^"]{10,})"', cleaned)
        valid = [v for v in quoted if v.strip() and v != anchor]
        if valid:
            logger.info(f"generated {len(valid)} valid variants (quoted fallback)")
            return valid[:n_variants]

        # Last fallback: non-empty lines that look like prompts
        lines = [
            l.strip().lstrip("0123456789.-) ")
            for l in cleaned.splitlines()
            if len(l.strip()) > 15
        ]
        valid = [l for l in lines if l and l != anchor]
        if valid:
            logger.info(f"generated {len(valid)} valid variants (line fallback)")
            return valid[:n_variants]

    except Exception as e:
        logger.warning(f"variant generation failed: {e} — using anchor prompt")

    if raw:
        logger.warning(f"all parsers failed. Raw: {raw[:120]}")

    return [anchor]


def _compute_ssc(
    prompt: str,
    input_example: str,
    task: str,
    backend: ModelBackend,
    k: int = SSC_RUNS,
    temperature: float = SSC_TEMPERATURE,
) -> tuple[float, float, str]:
    """
    Semantic Self-Consistency score for one prompt. Runs the prompt K times and
    computes average pairwise semantic similarity. Also returns the average
    reachability if logprobs are available.
    """
    # Create K copies of the same prompt template to run in parallel
    variant_copies = [prompt] * k

    results = run_variants_parallel(
        templates=variant_copies,
        input_text=input_example,
        task=task,
        backend=backend,
        temperature=temperature,
        max_workers=k,  # Run all K simultaneously
    )

    outputs = []
    reachabilities = []

    for r in results:
        if r.text.strip():
            outputs.append(r.text)

        if r.logprobs:
            reachabilities.append(_compute_reachability(r.logprobs))
        else:
            reachabilities.append(0.6)  # Neutral fallback if no logprobs

    if not outputs:
        return 0.0, 0.6, ""

    sample_output = outputs[0]

    ssc = pairwise_similarity(outputs) if len(outputs) > 1 else 0.6
    avg_reach = sum(reachabilities) / len(reachabilities) if reachabilities else 0.6

    return round(ssc, 4), round(avg_reach, 4), sample_output


def run_rpe(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    backend: ModelBackend,
    feedback: str = "",
    n_variants: int = N_VARIANTS,
    ssc_runs: int = SSC_RUNS,
    weights: Optional[dict] = None,
    current_best_prompt: Optional[str] = None,
) -> RPEResult:

    if weights is None:
        if task in OPEN_ENDED_TASKS:
            weights = {"ssc": 0.5, "reach": 0.3, "sim": 0.2}
            logger.info("Using creative weights (prioritizing SSC)")
        else:
            if expected_output:
                weights = {"ssc": 0.2, "reach": 0.2, "sim": 0.6}
                logger.info("Using deterministic weights (prioritizing Similarity)")
            else:
                weights = {"ssc": 0.4, "reach": 0.4, "sim": 0.2}
                logger.info(
                    "Using deterministic weights without reference (SSC+Reachability)"
                )

    from core.evaluator.embedder import similarity as semantic_sim
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(
        f"rpe task={task} "
        f"n_variants={n_variants} "
        f"ssc_runs={ssc_runs} "
        f"backend={backend.value}"
    )

    variants = _generate_variants(
        base_prompt=base_prompt,
        feedback=feedback,
        n_variants=n_variants,
        backend=backend,
        task=task,
        current_best_prompt=current_best_prompt,
    )

    history = []
    best_prompt = current_best_prompt if current_best_prompt else base_prompt
    best_score = -1.0
    best_reachability = 0.5
    best_ssc = 0.5

    # Parallelize the scoring of all variants simultaneously
    # Instead of a sequential `for variant in variants:` loop, we submit
    # all _compute_ssc calls to a thread pool.

    def _score_variant(variant_str):
        """Helper to score a single variant, run inside the thread."""
        ssc, reach, sample_output = _compute_ssc(
            prompt=variant_str,
            input_example=input_example,
            task=task,
            backend=backend,
            k=ssc_runs,
        )

        if task in {"classify", "extract"} and expected_output:
            norm_out = sample_output.strip().lower()
            norm_exp = expected_output.strip().lower()
            # If expected is "yes", and output is "yes, the child can", sim = 1.0
            if norm_exp in norm_out:
                sim = 1.0
            else:
                sim = similarity(sample_output, expected_output)
        elif task in OPEN_ENDED_TASKS:
            sim = _creative_quality_heuristic(sample_output)
        elif expected_output:
            sim = semantic_sim(output=sample_output, expected=expected_output)
        else:
            sim = 0.5

        combined = round(
            weights["ssc"] * ssc + weights["reach"] * reach + weights["sim"] * sim, 4
        )

        return {
            "variant": variant_str,
            "ssc": ssc,
            "reachability": reach,
            "similarity": sim,
            "score": combined,
        }

    # Run all variant evaluations in parallel
    with ThreadPoolExecutor(max_workers=n_variants) as executor:
        # Map futures to their variant index for ordered logging if needed
        future_to_idx = {
            executor.submit(_score_variant, v): i for i, v in enumerate(variants)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()

                logger.info(
                    f"variant={idx} ssc={result['ssc']:.4f} reach={result['reachability']:.4f} "
                    f"sim={result['similarity']:.4f} combined={result['score']:.4f}"
                )

                history.append(result)

                if result["score"] > best_score:
                    best_score = result["score"]
                    best_prompt = result["variant"]
                    best_reachability = result["reachability"]
                    best_ssc = result["ssc"]

            except Exception as e:
                logger.error(f"Variant {idx} scoring failed: {e}")

    return RPEResult(
        best_prompt=best_prompt,
        best_score=best_score,
        best_reachability=best_reachability,
        best_ssc=best_ssc,
        history=history,
    )
