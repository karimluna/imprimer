'''
LangGraph node functions for the prompt optimization graph.

  generator  : runs Optuna TPE search over one mutation dimension (creative)
  evaluator  : scores the generator's candidate and generates feedback (critical)
  controller : decides whether to cycle back or terminate (executive)

RPE feedback loop:
  After each evaluator run, a brief verbal explanation is generated
  describing why the current best prompt works. This is injected into
  the next generator cycle via the PERSONAS slot, the model learns
  from its own prior successes at a semantic level.
'''

from core.optimizer.state import PromptState
from core.optimizer.bayesian_search import optimize as bayesian_optimize, PERSONAS
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import score as compute_score
from utils.create_logger import get_logger
import os
import requests
from core.optimizer.rpe import run_rpe

logger = get_logger(__name__)


def _generate_feedback(
    base_prompt: str,
    best_prompt: str,
    best_score: float,
    backend: ModelBackend,
) -> str:
    """
    Generates a brief verbal explanation of why the best prompt won.
    This is the RPE feedback signal - passed to the next generator cycle
    so the model builds on what worked rather than searching blindly.

    Uses Ollama or OpenAI depending on backend.
    Falls back to a neutral string on any failure - feedback is optional,
    the graph must never crash because of it.
    """
    feedback_prompt = (
        f"The following prompt scored {best_score:.3f} (higher is better):\n\n"
        f"  Original: {base_prompt}\n"
        f"  Improved: {best_prompt}\n\n"
        f"In two sentences, explain what makes the improved version better. "
        f"Focus on instruction clarity, specificity, or structure. "
        f"Do not add any preamble."
    )

    try:
        if backend == ModelBackend.OLLAMA:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": feedback_prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()

        elif backend == ModelBackend.OPENAI:
            from core.chains.prompt_chain import _build_openai_llm
            llm = _build_openai_llm()
            response = llm.invoke(feedback_prompt)
            return response.content.strip()

        else:
            # HuggingFace or unknown - skip feedback, cost too high
            return ""

    except Exception as e:
        logger.warning(f"feedback generation failed, skipping: {e}")
        return ""



def generator_node(state: PromptState) -> dict:
    """
    Generator — one RPE iteration.

    Generates N candidate prompts via verbalized sampling,
    scores each with SSC + optional reachability,
    returns the best candidate to the evaluator.

    RPE: True
    LLM calls: 1 (generation) + N×K (SSC) + N×1 (similarity comparison)
    With defaults: 1 + 5×2 + 5 = 16 calls per iteration.
    """
    iteration = state["current_iteration"]
    backend = ModelBackend(state["backend"])
    feedback = state.get("last_feedback", "")

    logger.info(
        f"generator iteration={iteration} "
        f"backend={state['backend']} "
        f"use_rpe={state['use_rpe']} "
        f"base_prompt={state['base_prompt'][:60]}"
    )

    if state["use_rpe"]:
        # UI path — open-ended LLM-driven variant generation
        result = run_rpe(
            task=state["task"],
            base_prompt=state["base_prompt"],
            input_example=state["input_example"],
            expected_output=state["expected_output"],
            backend=backend,
            feedback=feedback,
            n_variants=state["n_variants"],
        )
        best_prompt = result.best_prompt
        cycle_history = [{**h, "iteration": iteration} for h in result.history]

    else:
        # CLI path — Optuna TPE over structured spaCy mutations
        from core.optimizer.bayesian_search import optimize as bayesian_optimize, DIMENSION_SEQUENCE
        dimension = DIMENSION_SEQUENCE[iteration % len(DIMENSION_SEQUENCE)]

        extra_personas = [feedback] if feedback else []

        result = bayesian_optimize(
            task=state["task"],
            base_prompt=state["base_prompt"],
            input_example=state["input_example"],
            expected_output=state["expected_output"],
            n_trials=state["n_trials"],
            backend=backend,
            dimension=dimension,
            study_name=f"imprimer_{state['task']}_{dimension}",
            extra_personas=extra_personas,
        )
        best_prompt = result.best_prompt
        cycle_history = [{**h, "iteration": iteration} for h in result.history]

    logger.info(
        f"generator iteration={iteration} "
        f"best_score={result.best_score:.4f} "
        f"best_reachability={result.best_reachability:.4f}"
    )

    return {
        "current_prompt": best_prompt,
        "history": state["history"] + cycle_history,
    }

def evaluator_node(state: PromptState) -> dict:
    """
    Scores the generator's candidate and generates RPE feedback.

    Computes reachability + optional judge score.
    Updates best_prompt (by reachability) and global_best_prompt (by combined).
    Generates verbal feedback for the next generator cycle.
    """
    backend = ModelBackend(state["backend"])
    # original_global_best = state["global_best_score"]

    result = run_variant(
        template=state["current_prompt"],
        input_text=state["input_example"],
        task=state["task"],
        backend=backend,
    )

    s = compute_score(
        result=result,
        task=state["task"],
        input_text=state["input_example"],
        expected_output=state["expected_output"],  # was missing
        use_judge=state["use_judge"],
        backend=backend,
    )

    reachability = s.reachability
    combined = s.combined

    logger.info(
        f"evaluator iteration={state['current_iteration']} "
        f"reachability={reachability:.4f} "
        f"score={combined:.4f} "
        f"current_best_reach={state['best_reachability']:.4f} "
        f"global_best_score={state['global_best_score']:.4f}"
    )

    updates: dict = {}

    # Update reachability-best (used by controller for termination)
    if reachability >= state["best_reachability"]:
        updates.update({
            "best_prompt": state["current_prompt"],
            "best_reachability": reachability,
            "best_score": combined,
        })

    # Update global combined-best (returned to caller)
    if combined > state["global_best_score"]:
        updates.update({
            "global_best_prompt": state["current_prompt"],
            "global_best_score": combined,
            "global_best_reachability": reachability,
        })

    # Generate RPE feedback only when there was a genuine improvement
    
    
    if combined > state["baseline_score"] and state["current_prompt"] != state["base_prompt"]:
        feedback = _generate_feedback(
            base_prompt=state["base_prompt"],
            best_prompt=state["current_prompt"],
            best_score=combined,
            backend=backend,
        )
        if feedback:
            updates["last_feedback"] = feedback
            logger.info(f"feedback generated: {feedback[:80]}...")

    return updates


def controller_node(state: PromptState) -> dict:
    """
    Controller - decides whether to continue or terminate.

    Termination:
      1. best_reachability >= target_score
      2. current_iteration >= max_iterations
    """
    iteration = state["current_iteration"]
    best_score = state["global_best_score"]
    target = state["target_score"]
    max_iter = state["max_iterations"]

    baseline = state["baseline_score"]
    target_reached = (best_score >= target or best_score > baseline) 
    cap_reached = iteration >= max_iter - 1

    logger.info(
        f"controller iteration={iteration}/{max_iter} "
        f"reachability={best_score:.4f}/{target:.4f} "
        f"baseline={baseline:4f}"
        f"target_reached={target_reached} "
        f"cap_reached={cap_reached}"
    )

    return {
        "current_iteration": iteration + 1,
        "target_reached": target_reached,
        "iterations_completed": iteration + 1,
    }


def should_continue(state: PromptState) -> str:
    if state["target_reached"]:
        logger.info(
            f"graph terminating: target reachability "
            f"{state['target_score']:.4f} reached"
        )
        return "end"

    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(
            f"graph terminating: max iterations "
            f"{state['max_iterations']} reached"
        )
        return "end"

    return "generator"