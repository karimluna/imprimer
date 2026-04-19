'''
LangGraph node functions for the prompt optimization graph.

  generator  : runs Optuna TPE search over one mutation dimension (creative)
  evaluator  : scores the generator's candidate and generates feedback (critical)
  controller : decides whether to cycle back or terminate (executive)

rpe feedback loop:
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
        result = run_rpe(
            task=state["task"],
            base_prompt=state["base_prompt"],
            input_example=state["input_example"],
            expected_output=state["expected_output"],
            backend=backend,
            feedback=feedback,
            n_variants=state["n_variants"],
            current_best_prompt=current_best_prompt,
        )
        best_prompt = result.best_prompt
        cycle_history = [{**h, "iteration": iteration} for h in result.history]

        logger.info(
            f"generator iteration={iteration} "
            f"best_score={result.best_score:.4f} "
            f"best_reachability={result.best_reachability:.4f} "
            f"best_ssc={result.best_ssc:.4f}"
        )

        # Surface best_ssc so the evaluator can use it as a promotion signal
        # when logprobs are unavailable (HuggingFace and similar backends).
        return {
            "current_prompt": best_prompt,
            "best_ssc": result.best_ssc,
            "history": state["history"] + cycle_history,
        }

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

    # Bayesian path — no SSC available, keep existing best_ssc
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

    Promotion strategy — reachability-first:
      The evaluator promotes a candidate to global best when its REACHABILITY
      exceeds the current best reachability. This is intentional:

      - Reachability is grounded in the model's actual log-probability
        distribution. It measures whether the prompt steers the model toward
        naturally high-probability outputs — the core metric from the control
        theory framing.
      - combined_score mixes in latency and quality heuristics that are noisy
        on small local models (qwen2.5 1.5b) and can diverge significantly from
        the RPE scoring formula, causing candidates that the generator ranked
        highly to be rejected by the evaluator even when they're genuinely better.
      - global_best_score (for UI display) is still updated alongside so the
        progress bar reflects something meaningful to the user.

    Feedback comparison uses reachability so the LLM gets consistent signal
    about what "better" means across both halves of the loop.
    """
    backend = ModelBackend(state["backend"])

    result = run_variant(
        template=state["current_prompt"],
        input_text=state["input_example"],
        task=state["task"],
        backend=backend,
    )

    # skip judge in iteration 0 to save calls
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

    # -----------------------------------------------------------------------
    # Promotion signal selection — reachability vs SSC
    #
    # When logprobs ARE available (Ollama, OpenAI): promote on reachability.
    #   reachability is grounded in the model's token probability distribution
    #   and is the primary metric from the control theory framing.
    #
    # When logprobs are NOT available (HuggingFace, any backend that returns
    #   empty logprobs): reachability falls back to 0.5 for every candidate,
    #   meaning nothing ever gets promoted. Instead, use SSC (semantic
    #   self-consistency) as the control signal — it's the behavioral proxy
    #   for the same concept and is always computed by the RPE inner loop.
    #
    # logprobs_available is detected here on first use and stored in state
    # so every subsequent node uses the same criterion without re-detecting.
    # -----------------------------------------------------------------------
    has_logprobs = bool(result.logprobs)

    # Persist detection result — once we know, we know for all future cycles
    if "logprobs_available" not in state or state.get("logprobs_available") is None:
        updates["logprobs_available"] = has_logprobs
        logger.info(f"evaluator logprobs_available={has_logprobs} (detected on first run)")
    else:
        has_logprobs = state.get("logprobs_available", has_logprobs)

    if has_logprobs:
        control_signal = reachability
        best_control_signal = state["best_reachability"]
        signal_name = "reachability"
    else:
        # SSC of the generator's winning variant, surfaced via state["best_ssc"]
        # Falls back to 0.5 (neutral) if generator hasn't run yet
        control_signal = s.quality_score  # quality_score = SSC proxy via scorer heuristic
        # Use best_ssc from state — set by generator_node from RPEResult.best_ssc
        best_control_signal = state.get("best_ssc", 0.5)
        signal_name = "ssc"
        logger.info(
            f"evaluator no logprobs — using SSC proxy as control signal "
            f"current={control_signal:.4f} best={best_control_signal:.4f}"
        )

    is_new_best = control_signal > best_control_signal

    if is_new_best:
        updates.update({
            "global_best_prompt": state["current_prompt"],
            "global_best_score": combined,
            "global_best_reachability": reachability,
            "best_prompt": state["current_prompt"],
            "best_reachability": reachability,
            "best_score": combined,
            # Update whichever signal we're tracking
            **({"best_ssc": control_signal} if not has_logprobs else {}),
        })
        logger.info(
            f"evaluator new best ({signal_name}): "
            f"{best_control_signal:.4f} -> {control_signal:.4f} "
            f"prompt={state['current_prompt'][:60]}"
        )

    # ALWAYS generate feedback so the generator learns from every attempt.
    if state["current_prompt"] != state["base_prompt"]:
        previous_best_prompt = state.get("global_best_prompt", state["base_prompt"])
        previous_signal = best_control_signal

        feedback = _generate_feedback(
            base_prompt=previous_best_prompt,
            current_prompt=state["current_prompt"],
            base_score=previous_signal,
            current_score=control_signal,
            backend=backend,
        )
        if feedback:
            updates["last_feedback"] = feedback
            logger.info(f"feedback generated: {feedback[:80]}...")

    # LangGraph can emit None in the stream when a node returns {}.
    # Always include at least one key so the state update is never empty.
    if not updates:
        updates["best_reachability"] = state.get("best_reachability", 0.0)

    return updates

def controller_node(state: PromptState) -> dict:
    """
    Controller - decides whether to continue or terminate.

    Termination:
      1. best_reachability >= target_score  (reachability-first: matches evaluator)
      2. current_iteration >= max_iterations
    """
    iteration = state["current_iteration"]
    # Use reachability as the termination criterion to match the evaluator's
    # promotion logic. global_best_score is a mixed metric and not reliable
    # as a termination signal on small models.
    best_reachability = state["best_reachability"]
    target = state["target_score"]
    max_iter = state["max_iterations"]

    baseline = state["baseline_reachability"]
    target_reached = best_reachability >= target
    cap_reached = iteration >= max_iter - 1

    logger.info(
        f"controller iteration={iteration}/{max_iter} "
        f"best_reachability={best_reachability:.4f}/{target:.4f} "
        f"baseline_reachability={baseline:.4f} "
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