"""
LangGraph node functions.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.optimizer.state import PromptState
from core.optimizer.grpo import run_grpo
from core.optimizer.rpe import extract_residual_content
from core.chains.prompt_chain import ModelBackend, run_variant, call_llm
from core.evaluator.scorer import rank_score
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _structured_diff(previous: str, current: str) -> str:
    """
    Deterministic word-level diff instead of LLM reflection.
    """
    prev_words = set(previous.lower().split())
    curr_words = set(current.lower().split())
    added   = sorted(curr_words - prev_words)[:10]
    removed = sorted(prev_words - curr_words)[:10]

    parts = []
    if added:
        parts.append(f"Added terms: {', '.join(added)}")
    if removed:
        parts.append(f"Removed terms: {', '.join(removed)}")
    if not parts:
        parts.append("Minor rephrasing, same core intent")
    return ". ".join(parts) + "."



def _judge(
    previous_prompt: str, 
    current_prompt: str, 
    task: str, 
    backend: ModelBackend,
    is_improvement: bool
) -> str:
    """
    Uses the Generator LLM to reflect on semantic changes and provide actionable advice.
    Provides a rich text 'gradient' for the next optimizer cycle.
    """
    direction = "improved" if is_improvement else "degraded"
    
    sys_prompt = f"""You are an expert prompt engineer optimizing a prompt for a '{task}' task.
We just tested a new prompt variant. The performance {direction}.

[Previous Best Prompt]: {previous_prompt}
[New Prompt Tested]: {current_prompt}

Analyze the semantic difference. In ONE concise sentence, explain why the new prompt {direction} performance.
In a SECOND concise sentence, give a direct, actionable instruction for the next iteration to improve it further.
Do not use pleasantries. Keep it extremely brief."""

    try:
        # We use call_llm (the generator model) because evaluation/reflection requires higher reasoning
        reflection = call_llm(
            prompt_text=sys_prompt,
            backend=backend,
            temperature=0.3, # Low temp for analytical reflection
            max_tokens=150
        )
        return reflection.strip()
    except Exception as exc:
        logger.error(f"Judge reflection failed: {exc}")
        # Fallback to the dumb diff if the API hiccups, so we don't crash the graph
        return _structured_diff(previous_prompt, current_prompt)


def _score_across_examples(
    prompt: str,
    primary_input: str,
    primary_expected: str,
    extra_examples: list[dict],
    task: str,
    backend: ModelBackend,
) -> tuple[float, float, float]:
    """
    Runs the prompt against the primary example + all extra_examples
    in parallel and returns averaged (reachability, quality, combined).

    This is only called on the GRPO winner in the evaluator — not during
    ranking (which uses the primary example only for speed).
    """
    all_examples = [{"input": primary_input, "expected": primary_expected}] + list(extra_examples)

    def _score_one(ex: dict):
        r = run_variant(
            template=prompt,
            input_text=ex.get("input", ""),
            task=task,
            backend=backend,
        )
        return rank_score(r, task=task, expected_output=ex.get("expected", ""))

    scores = []
    with ThreadPoolExecutor(max_workers=len(all_examples)) as executor:
        futures = [executor.submit(_score_one, ex) for ex in all_examples]
        for f in as_completed(futures):
            try:
                scores.append(f.result())
            except Exception as exc:
                logger.error(f"multi-example scoring failed: {exc}")

    if not scores:
        return 0.5, 0.5, 0.5

    avg_reach   = round(sum(s.reachability for s in scores) / len(scores), 4)
    avg_quality = round(sum(s.quality for s in scores) / len(scores), 4)
    avg_combined = round(sum(s.combined for s in scores) / len(scores), 4)
    return avg_reach, avg_quality, avg_combined


def generator_node(state: PromptState) -> dict:
    """
    GRPO+RiOT step: generate N variants, score in parallel, return winner.
    Generator model: 1 call. Evaluator model: N parallel calls.
    """
    iteration = state["current_iteration"]
    backend   = ModelBackend(state["backend"])
    anchor    = state["best_prompt"]

    logger.info(
        f"generator iter={iteration} backend={state['backend']} "
        f"n_variants={state['n_variants']} anchor={anchor[:60]!r} "
        f"has_residual={bool(state.get('residual_content'))}"
    )

    result = run_grpo(
        task=state["task"],
        base_prompt=state["base_prompt"],
        input_example=state["input_example"],
        expected_output=state["expected_output"],
        backend=backend,
        feedback=state.get("last_feedback", ""),
        n_variants=state["n_variants"],
        current_best_prompt=anchor,
        residual_content=state.get("residual_content", ""),
    )

    logger.info(
        f"generator iter={iteration} winner_score={result.best_score:.4f} "
        f"reach={result.best_reachability:.4f} "
        f"grpo_reward={result.best_grpo_reward:.4f} "
        f"group_mean={result.group_mean:.4f} std={result.group_std:.4f}"
    )

    return {
        "current_prompt":  result.best_prompt,
        "grpo_group_mean": result.group_mean,
    }


def evaluator_node(state: PromptState) -> dict:
    """
    Authoritative scoring of the generator's winner.

    Scores against primary + extra_examples in parallel, averages.
    The group mean used for promotion reflects generalization, not
    memorization of one input.

    The run_variant call for the primary example is always a cache hit
    (GRPO already ran this exact prompt at temp=0). Extra examples add
    E parallel calls to the evaluator model.
    """
    backend   = ModelBackend(state["backend"])
    iteration = state["current_iteration"]

    old_best_prompt = state.get("best_prompt", state["base_prompt"])

    extra_examples = state.get("extra_examples", [])

    avg_reach, avg_quality, avg_combined = _score_across_examples(
        prompt=state["current_prompt"],
        primary_input=state["input_example"],
        primary_expected=state["expected_output"],
        extra_examples=extra_examples,
        task=state["task"],
        backend=backend,
    )

    has_logprobs = bool(run_variant(
        template=state["current_prompt"],
        input_text=state["input_example"],
        task=state["task"],
        backend=backend,
    ).logprobs)

    logger.info(
        f"evaluator iter={iteration} "
        f"avg_reach={avg_reach:.4f} avg_combined={avg_combined:.4f} "
        f"best_reach={state['best_reachability']:.4f} "
        f"examples={1 + len(extra_examples)}"
    )

    updates: dict = {"current_cycle_reachability": avg_reach}

    if state.get("logprobs_available") is None:
        updates["logprobs_available"] = has_logprobs

    control_signal = avg_reach if (has_logprobs or state.get("logprobs_available")) else avg_combined
    best_control   = state["best_reachability"] if (has_logprobs or state.get("logprobs_available")) else state["best_score"]
    signal_name    = "reachability" if (has_logprobs or state.get("logprobs_available")) else "combined"

    is_new_best = control_signal > best_control

    run_id = state.get("run_id") or str(uuid.uuid4())
    if not state.get("run_id"):
        updates["run_id"] = run_id


    if is_new_best:
        new_residual = extract_residual_content(state["current_prompt"])
        updates.update({
            "best_prompt":       state["current_prompt"],
            "best_reachability": avg_reach,
            "best_score":        avg_combined,
            "residual_content":  new_residual,
        })
        logger.info(
            f"evaluator new best ({signal_name}): "
            f"{best_control:.4f} -> {control_signal:.4f} "
            f"residual_lines={len(new_residual.splitlines())} "
            f"prompt={state['current_prompt'][:60]!r}"
        )

    # run judge
    if state["current_prompt"] != old_best_prompt:
        feedback = _judge(
            previous_prompt=old_best_prompt,
            current_prompt=state["current_prompt"],
            task=state["task"],
            backend=backend,
            is_improvement=is_new_best
        )
        updates["last_feedback"] = feedback
        logger.info(f"feedback (judge): {feedback[:120]!r}")

    return updates


def controller_node(state: PromptState) -> dict:
    iteration      = state["current_iteration"]
    target_reached = state["best_reachability"] >= state["target_score"]

    logger.info(
        f"controller iter={iteration}/{state['max_iterations']} "
        f"best_reach={state['best_reachability']:.4f}/{state['target_score']:.4f} "
        f"target_reached={target_reached}"
    )

    return {
        "current_iteration":    iteration + 1,
        "target_reached":       target_reached,
        "iterations_completed": iteration + 1,
    }


def should_continue(state: PromptState) -> str:
    if state["target_reached"]:
        logger.info(f"graph terminating: target {state['target_score']:.4f} reached")
        return "end"
    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(f"graph terminating: max iterations {state['max_iterations']} reached")
        return "end"
    return "generator"