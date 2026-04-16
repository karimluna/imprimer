"""
LangGraph optimization graph —> the outer control loop.
This is the top-level entry point for prompt optimization.

Graph structure:
  generator -> evaluator -> controller -> (generator | END)

The inner loop (Optuna TPE) lives inside the generator node.
The outer loop (reachability threshold) is managed by this graph.
Together they implement a two-level control hierarchy:
  - Inner: find the best mutation in this cycle (Optuna)
  - Outer: decide if the result is good enough to stop (LangGraph)
"""

from langgraph.graph import StateGraph, END

from core.optimizer.state import PromptState
from core.optimizer.nodes import (
    generator_node,
    evaluator_node,
    controller_node,
    should_continue
)
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import score as compute_score
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _build_graph() -> StateGraph:
    """"Builds the LangGraph optimization graph. Just called once."""
    graph = StateGraph(PromptState)

    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("controller", controller_node)

    # Linear flow within a cycle
    graph.add_edge("generator", "evaluator")
    graph.add_edge("evaluator", "controller")

    # Conditional edge after controller
    graph.add_conditional_edges(
        "controller", 
        should_continue,
        {
            "generator": "generator",
            "end": END
        }
    )

    graph.set_entry_point("generator")
    return graph.compile()

_graph = _build_graph()



def optimize(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    n_trials: int = 20,
    backend: ModelBackend = ModelBackend.OLLAMA,
    use_judge: bool = False,
    target_reachability: float = 0.80,
    max_iterations: int = 3,
) -> dict:
    """
    Entry point for the LangGraph optimization loop.

    Runs the generation -> evaluation -> controller graph until
    the target reachability is met or max_iterations is exhausted.

    Returns a dict matching OptimizeResponse fields so main.py
    can pass it directly to the gRPC response constructor.

    target_reachability: stop when best reachability >= this value.
                         Default 0.80, below the paper's 0.97 ceiling
                         but achievable with small local models.
    max_iterations: hard cap on graph cycles. Each cycle runs n_trials
                    Optuna trials, so total LLM calls = n_trials × iterations.
                    Default 3 = up to 60 calls with n_trials=20.
    """

    backend_str = backend.value

    # Establish baseline before graph runs
    baseline_result = run_variant(
        template=base_prompt,
        input_text=input_example,
        task=task,
        backend=backend_str
    )

    baseline_score_obj = compute_score(baseline_result)
    baseline_score = baseline_score_obj.combined
    baseline_reachability = baseline_score_obj.reachability

    logger.info(
        f"graph starting task={task} "
        f"baseline_reachability={baseline_reachability:.4f} "
        f"target={target_reachability:.4f} "
        f"max_iterations={max_iterations}"
    )

    initial_state: PromptState = {
        "task": task,
        "input_example": input_example,
        "expected_output": expected_output,
        "backend": backend_str,
        "use_judge": use_judge,
        "target_reachability": target_reachability,
        "max_iterations": max_iterations,
        "n_trials": n_trials,
        "current_prompt": base_prompt,
        "current_iteration": 0,
        "best_prompt": base_prompt,
        "best_reachability": baseline_reachability,
        "best_score": baseline_score,
        "baseline_score": baseline_score,
        "baseline_reachability": baseline_reachability,
        "history": [],
        "target_reached": False,
        "iterations_completed": 0,
    }

    final_state = _graph.invoke(initial_state)

    improvement = round(
        final_state["best_score"] - baseline_score, 4
    )

    logger.info(
        f"graph complete "
        f"iterations={final_state['iterations_completed']} "
        f"best_reachability={final_state['best_reachability']:.4f} "
        f"target_reached={final_state['target_reached']} "
        f"improvement={improvement:+.4f}"
    )

    return {
        "best_prompt": final_state["best_prompt"],
        "best_score": final_state["best_score"],
        "best_reachability": final_state["best_reachability"],
        "baseline_score": baseline_score,
        "baseline_reachability": baseline_reachability,
        "improvement": improvement,
        "trials_run": len(final_state["history"]),
        "iterations_completed": final_state["iterations_completed"],
        "target_reached": final_state["target_reached"],
    }