'''
LangGraph node functions for the prompt optimization graph.

  generator  : runs Optuna TPE search over one mutation dimension (creative)
  evaluator  : scores the generator's candidate and generates feedback (critical)
  controller : decides whether to cycle back or terminate (executive)

rpe feedback loop:
  After each evaluator run, a brief verbal explanation is generated
  describing why the current best prompt works. This is injectesd into
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
    current_prompt: str,
    base_score: float,
    current_score: float,
    backend: ModelBackend,
) -> str:
    """
    Generates a brief verbal explanation of why the prompt succeeded or failed.
    """
    is_improvement = current_score > base_score
    
    if is_improvement:
        context = "improved the score"
        instruction = "explain what makes the new version better so we can keep doing it."
    else:
        context = "caused the score to drop"
        instruction = "explain why the new version performed worse so we can avoid this mistake next time."

    feedback_prompt = (
        f"We are optimizing an AI prompt. The previous best prompt scored {base_score:.3f}.\n"
        f"The new prompt scored {current_score:.3f}, which means it {context}.\n\n"
        f"  Previous: {base_prompt}\n"
        f"  New: {current_prompt}\n\n"
        f"In two sentences, {instruction} "
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

    FIX (Problem 3): passes `global_best_prompt` from state as
    `current_best_prompt` into run_rpe so the generator builds on the
    actual winning prompt from prior cycles, not the frozen original.

    RPE: True
    LLM calls: 1 (generation) + N×K (SSC) + N×1 (similarity comparison)
    With defaults: 1 + 5×2 + 5 = 16 calls per iteration.
    """
    iteration = state["current_iteration"]
    backend = ModelBackend(state["backend"])
    feedback = state.get("last_feedback", "")

    # FIX: use the current global best as the evolving anchor for generation
    current_best_prompt = state.get("global_best_prompt", state["base_prompt"])

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
            # FIX: pass the evolving best prompt so variants build on it
            current_best_prompt=current_best_prompt,
        )
        best_prompt = result.best_prompt
        cycle_history = [{**h, "iteration": iteration} for h in result.history]

    else:
        # CLI path — Optuna TPE over structured spaCy mutations
        from core.optimizer.bayesian_search import optimize as bayesian_optimize, DIMENSION_SEQUENCE
        dimension = DIMENSION_SEQUENCE[iteration % len(DIMENSION_SEQUENCE)]

        # Parse raw feedback string into structured Optuna hints
        reflection_hints = {}
        if feedback:
            f_lower = feedback.lower()
            
            if "verbose" in f_lower or "too long" in f_lower or "concise" in f_lower:
                reflection_hints["output_contract"] = "Output only the answer, no preamble."
            
            if "hedge" in f_lower or "uncertain" in f_lower or "confident" in f_lower:
                reflection_hints["hedging"] = "Be definitive."
                
            if "unprofessional" in f_lower or "expert" in f_lower:
                reflection_hints["persona"] = "You are an expert in this domain."

        hints_to_pass = reflection_hints if reflection_hints else None

        result = bayesian_optimize(
            task=state["task"],
            base_prompt=state["base_prompt"],
            input_example=state["input_example"],
            expected_output=state["expected_output"],
            n_trials=state["n_trials"],
            backend=backend,
            dimension=dimension,
            study_name=f"imprimer_{state['task']}_{dimension}",
            reflection_hints=hints_to_pass, 
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

    result = run_variant(
        template=state["current_prompt"],
        input_text=state["input_example"],
        task=state["task"],
        backend=backend,
    )

    # skip ssc in early iteration
    _use_judge = False if state.get('current_iteration', 0) == 0 else state['use_judge']

    s = compute_score(
        result=result,
        task=state["task"],
        input_text=state["input_example"],
        expected_output=state["expected_output"],
        use_judge=_use_judge,
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

    is_new_best = combined > state["global_best_score"]

    # Update global combined-best (returned to caller)
    if is_new_best:
        updates.update({
            "global_best_prompt": state["current_prompt"],
            "global_best_score": combined,
            "global_best_reachability": reachability,
        })

    # ALWAYS generate feedback to learn from the attempt (Success or Failure)
    if state["current_prompt"] != state["base_prompt"]:
        previous_best_prompt = state.get("global_best_prompt", state["base_prompt"])
        previous_best_score = state.get("global_best_score", state["baseline_score"])
        
        feedback = _generate_feedback(
            base_prompt=previous_best_prompt,
            current_prompt=state["current_prompt"],
            base_score=previous_best_score,
            current_score=combined,
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
    target_reached = best_score >= target 
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
            f"graph terminating: target score "
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