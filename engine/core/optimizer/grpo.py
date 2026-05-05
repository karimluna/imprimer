"""
Group Relative Policy Optimization for prompt selection.

One cycle:
  1. Generate N candidate prompts (1 generator call, already done upstream)
  2. Score each candidate against the PRIMARY example in parallel (N evaluator calls)
  3. Compute ELPR reward relative to the group mean
  4. Return the winner

"""

import math
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import rank_score
from utils.create_logger import get_logger

logger = get_logger(__name__)

GRPO_STEEP = 3.0
N_VARIANTS = 4       


@dataclass
class GRPOResult:
    best_prompt:       str
    best_score:        float
    best_reachability: float
    best_grpo_reward:  float
    group_mean:        float
    group_std:         float
    history: list = field(default_factory=list)


def elpr_reward(score: float, group_mean: float, steep: float = GRPO_STEEP) -> float:
    return round(1.0 / (1.0 + math.exp(-steep * (score - group_mean))), 4)


def _group_stats(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    mean     = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return round(mean, 4), round(math.sqrt(variance), 4)


def run_grpo(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    backend: ModelBackend,
    feedback: str = "",
    n_variants: int = N_VARIANTS,
    current_best_prompt: Optional[str] = None,
    residual_content: str = "",
) -> GRPOResult:
    """
    Generates N variants and selects the best via ELPR group-relative reward.
    Scores each variant against the primary example only (1 call each, parallel).
    """
    from core.optimizer.rpe import _generate_variants_with_residual

    anchor   = current_best_prompt or base_prompt
    variants = _generate_variants_with_residual(
        base_prompt=base_prompt,
        feedback=feedback,
        n_variants=n_variants,
        backend=backend,
        task=task,
        current_best_prompt=anchor,
        residual_content=residual_content,
    )

    if not variants:
        logger.warning("grpo: no variants generated, returning anchor")
        return GRPOResult(
            best_prompt=anchor, best_score=0.0, best_reachability=0.5,
            best_grpo_reward=0.5, group_mean=0.0, group_std=0.0,
        )

    def _score_one(variant_str: str) -> dict:
        result = run_variant(
            template=variant_str,
            input_text=input_example,
            task=task,
            backend=backend,
            temperature=0.0,
        )
        s = rank_score(result, task=task, expected_output=expected_output)
        return {
            "variant":      variant_str,
            "reachability": s.reachability,
            "similarity":   s.similarity,
            "score":        s.combined,
        }

    history: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        future_map = {executor.submit(_score_one, v): i for i, v in enumerate(variants)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                r = future.result()
                history.append(r)
                logger.info(
                    f"grpo variant={idx} reach={r['reachability']:.4f} "
                    f"sim={r['similarity']:.4f} score={r['score']:.4f}"
                )
            except Exception as exc:
                logger.error(f"grpo variant {idx} scoring failed: {exc}")

    if not history:
        return GRPOResult(
            best_prompt=anchor, best_score=0.0, best_reachability=0.5,
            best_grpo_reward=0.5, group_mean=0.0, group_std=0.0, history=[],
        )

    raw_scores      = [h["score"] for h in history]
    g_mean, g_std   = _group_stats(raw_scores)

    for h in history:
        h["grpo_reward"] = elpr_reward(h["score"], g_mean)

    winner = max(history, key=lambda h: h["grpo_reward"])

    logger.info(
        f"grpo complete | mean={g_mean:.4f} std={g_std:.4f} "
        f"winner={winner['score']:.4f} reward={winner['grpo_reward']:.4f} "
        f"prompt={winner['variant'][:60]!r}"
    )

    return GRPOResult(
        best_prompt=winner["variant"],
        best_score=winner["score"],
        best_reachability=winner["reachability"],
        best_grpo_reward=winner["grpo_reward"],
        group_mean=g_mean,
        group_std=g_std,
        history=history,
    )