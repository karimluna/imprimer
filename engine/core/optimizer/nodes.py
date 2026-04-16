'''
LangGraph node functions for the prompt optimization graph. Each funcion
is a node.

  generator  : runs Optuna TPE search over the mutation space (creative)
  evaluator  : scores the generator's best candidate          (critical)
  controller : decides whether to cycle back or terminate     (executive)
'''

from core.optimizer.state import PromptState
from core.optimizer.bayesian_search import optimize as bayesian_optimize
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import score as compute_score
from core.evaluator.judge import judge as llm_judge
from utils.create_logger import get_logger

logger = get_logger(__name__)


def generator_node(state: PromptState) -> dict:
    """
    Runs one optimization cycle over the mutation space.

    Always optimizes from base_prompt (the original raw task instruction),
    never from current_prompt. This prevents the decorator layers added by
    build_prompt from being re-wrapped on every graph cycle, which would
    cause persona/constraints to accumulate across iterations.

    The best candidate found is stored in current_prompt for the evaluator,
    but base_prompt is never mutated.
    """
    iteration = state["current_iteration"]
    base = state["base_prompt"]  # always the raw instruction, never decorated

    logger.info(
        f"generator iteration={iteration} "
        f"base_prompt={base[:60]}"
    )

    backend = ModelBackend(state["backend"])

    result = bayesian_optimize(
        task=state["task"],
        base_prompt=base,
        input_example=state["input_example"],
        expected_output=state["expected_output"],
        n_trials=state["n_trials"],
        backend=backend,
        # Pass study name so Optuna resumes across graph cycles —
        # earlier trials from previous cycles inform this one.
        # Note: same study name across iterations is intentional —
        # TPE learns from all previous trials, not just this cycle's.
        study_name=f"imprimer_{state['task']}",
    )

    logger.info(
        f"generator iteration={iteration} "
        f"candidate_score={result.best_score:.4f} "
        f"candidate_reachability={result.best_reachability:.4f}"
    )

    # Append this cycle's history to the global history
    cycle_history = [
        {**h, "iteration": iteration}
        for h in result.history
    ]

    return {
        # current_prompt carries the decorated candidate to the evaluator
        "current_prompt": result.best_prompt,
        "history": state["history"] + cycle_history,
    }


def evaluator_node(state: PromptState) -> dict:
    """
    Scores the generator's candidate prompt.

    Runs the current prompt against the example input and computes:
      - Reachability index (always)
      - LLM-as-judge score (if use_judge=True)

    If neither condition is met, increments the iteration counter
    and returns control to the generator for another cycle. This is
    to prevent the control's loop to run indefinitely.
    """
    backend = ModelBackend(state["backend"])

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
        use_judge=state["use_judge"],
        backend=backend,
    )

    reachability = s.reachability
    combined = s.combined

    logger.info(
        f"evaluator iteration={state['current_iteration']} "
        f"reachability={reachability:.4f} "
        f"score={combined:.4f} "
        f"current_best={state['best_reachability']:.4f}"
    )

    # Only update best if this candidate is strictly better
    if reachability > state["best_reachability"]:
        return {
            "best_prompt": state["current_prompt"],
            "best_reachability": reachability,
            "best_score": combined,
        }

    # Candidate didn't beat the current best — state unchanged
    return {}


def controller_node(state: PromptState) -> dict:
    """
    Controller, decides whether to continue or terminate.

    Termination conditions (either triggers exit):
      1. best_reachability >= target_reachability (target met)
      2. current_iteration >= max_iterations (iteration cap hit)
    """
    iteration = state["current_iteration"]
    best_reach = state["best_reachability"]
    target = state["target_reachability"]
    max_iter = state["max_iterations"]

    target_reached = best_reach >= target
    cap_reached = iteration >= max_iter - 1

    logger.info(
        f"controller iteration={iteration}/{max_iter} "
        f"reachability={best_reach:.4f}/{target:.4f} "
        f"target_reached={target_reached} "
        f"cap_reached={cap_reached}"
    )

    return {
        "current_iteration": iteration + 1,
        "target_reached": target_reached,
        "iterations_completed": iteration + 1,
    }


def should_continue(state: PromptState) -> str:
    """
    Conditional edge function, is called after controller_node
    to decide which node to route next.
    """
    if state["target_reached"]:
        logger.info(
            f"graph terminating: target reachability "
            f"{state['target_reachability']:.4f} reached"
        )
        return "end"

    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(
            f"graph terminating: max iterations "
            f"{state['max_iterations']} reached"
        )
        return "end"

    return "generator"