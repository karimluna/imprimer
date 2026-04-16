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

    Takes the current best prompt as the base and searches for a mutation
    that improves reachability + similarity. Returns the best candidate
    found in this cycle.
    """    
    iteration = state["current_iteration"]
    base = state["current_prompt"]

    logger.info(
      f"generator iteration = {iteration}"
      f"base_prompt = {base[:60]}"
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
        # earlier trials from previous cycles inform this one
        study_name=f"imprimer_{state['task']}_iter{iteration}",
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
    and returns control to the generator for another cycle. THis is
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