"""
LangGraph optimization graph — the outer control loop.

Graph structure:
  generator -> evaluator -> controller -> (generator | END)

State invariants:
  base_prompt    : never mutated. The generator always reads this
                   to prevent decorator accumulation across cycles.
  current_prompt : decorated candidate for the current cycle.
  backend        : always stored as str (ModelBackend.value), never the enum.
                   Nodes call ModelBackend(state["backend"]) to recover it.
  last_feedback  : verbal explanation from the evaluator, injected into
                   the next generator cycle as an extra persona candidate.
"""

from langgraph.graph import StateGraph, END

from core.optimizer.state import PromptState
from core.optimizer.nodes import (
    generator_node,
    evaluator_node,
    controller_node,
    should_continue,
)
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import score as compute_score
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _build_graph() -> StateGraph:
    graph = StateGraph(PromptState)

    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("controller", controller_node)

    graph.add_edge("generator", "evaluator")
    graph.add_edge("evaluator", "controller")

    graph.add_conditional_edges(
        "controller",
        should_continue,
        {
            "generator": "generator",
            "end": END,
        }
    )

    graph.set_entry_point("generator")
    return graph.compile()


_graph = _build_graph()


def optimize(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str = "",
    n_variants: int = 6,
    n_trials: int = 20,
    backend: ModelBackend = ModelBackend.OLLAMA,
    use_judge: bool = False,
    use_rpe: bool = True,
    target_score: float = 0.80, # we are fine with a better score than the baseline, so just updating it based on that is okay
    max_iterations: int = 3,
) -> dict:
    """
    Entry point for the LangGraph optimization loop.

    backend is accepted as ModelBackend enum here for a clean public API,
    but stored as backend.value (str) in the state so all nodes receive
    a consistent serializable type.
    """
    backend_str = backend.value  # always string inside state

    baseline_result = run_variant(
        template=base_prompt,
        input_text=input_example,
        task=task,
        backend=backend,
    )

    baseline_score_obj = compute_score(
        result=baseline_result,
        task=task,
        input_text=input_example,
        expected_output=expected_output,
        use_judge=use_judge,
        backend=backend,
    )
    baseline_score = baseline_score_obj.combined
    baseline_reachability = baseline_score_obj.reachability

    logger.info(
        f"graph starting task={task} "
        f"backend={backend_str} "
        f"baseline_reachability={baseline_reachability:.4f} "
        f"target={target_score:.4f} "
        f"max_iterations={max_iterations}"
    )

    initial_state: PromptState = {
        "task": task,
        "input_example": input_example,
        "expected_output": expected_output,
        "backend": backend_str,          # string, not enum
        "use_judge": use_judge,
        "use_rpe": use_rpe,
        "base_prompt": base_prompt,
        "current_prompt": base_prompt,
        "current_iteration": 0,
        "last_feedback": "",             # seeds RPE feedback loop
        "target_score": target_score,
        "max_iterations": max_iterations,
        "n_variants": n_variants,
        "n_trials": n_trials,
        "best_prompt": base_prompt,
        "best_reachability": baseline_reachability,
        "best_score": baseline_score,
        "global_best_prompt": base_prompt,
        "global_best_score": baseline_score,
        "global_best_reachability": baseline_reachability,
        "baseline_score": baseline_score,
        "baseline_reachability": baseline_reachability,
        "history": [],
        "target_reached": False,
        "iterations_completed": 0,
    }

    final_state = _graph.invoke(initial_state)

    improvement = round(
        final_state["global_best_score"] - baseline_score, 4
    )

    logger.info(
        f"graph complete "
        f"iterations={final_state.get('current_iteration', 0)} "
        f"best_reachability={final_state['global_best_reachability']:.4f} "
        f"target_reached={final_state.get('target_reached', False)} "
        f"improvement={improvement:+.4f}"
    )

    return {
        "best_prompt": final_state["global_best_prompt"],
        "best_score": final_state["global_best_score"],
        "best_reachability": final_state["global_best_reachability"],
        "baseline_score": baseline_score,
        "baseline_reachability": baseline_reachability,
        "improvement": improvement,
        "iterations_completed": final_state.get("current_iteration", 0),
        "target_reached": final_state.get("target_reached", False),
        "feedback": final_state.get("last_feedback", "")
    }