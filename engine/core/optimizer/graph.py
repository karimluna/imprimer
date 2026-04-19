"""
LangGraph optimization graph, the outer control loop.

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
from typing import Generator

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
    n_variants: int = 3,
    n_trials: int = 20,
    backend: ModelBackend = ModelBackend.OLLAMA,
    use_judge: bool = False,
    use_rpe: bool = True,
    target_score: float = 0.70, 
    max_iterations: int = 5,
) -> Generator[dict, None, None]:
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
        "backend": backend_str,
        "use_judge": use_judge,
        "use_rpe": use_rpe,
        "base_prompt": base_prompt,
        "current_prompt": base_prompt,
        "current_iteration": 0,
        "last_feedback": "",
        "target_score": target_score,
        "max_iterations": max_iterations,
        "n_variants": n_variants,
        "n_trials": n_trials,
        "best_prompt": base_prompt,
        "best_reachability": baseline_reachability,
        "best_ssc": 0.5,               # updated by generator_node after first RPE cycle
        "logprobs_available": None,    # detected by evaluator_node on first run
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

    if use_rpe == False:
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

        best_reach = final_state.get("best_reachability", baseline_reachability)
        reach_improvement = round(best_reach - baseline_reachability, 4)
        yield {
            "best_prompt": final_state["global_best_prompt"],
            "best_score": best_reach,
            "best_reachability": best_reach,
            "baseline_score": baseline_reachability,
            "baseline_reachability": baseline_reachability,
            "improvement": reach_improvement,
            "iterations_completed": final_state.get("current_iteration", 0),
            "target_reached": final_state.get("target_reached", False),
            "feedback": final_state.get("last_feedback", "")
        }
        return

    else:
        current_full_state = initial_state.copy()

        for event in _graph.stream(initial_state):
            for node_name, state_update in event.items():

                # a node returns an empty dict (e.g. evaluator finds no new best).
                # dict.update(None) raises 'NoneType' object is not iterable.
                if state_update is not None:
                    current_full_state.update(state_update)
                
                if node_name == "controller":
                    improvement = round(
                        current_full_state["global_best_score"] - baseline_score, 4
                    )
                    
                    logger.info(
                        f"cycle complete "
                        f"iterations={current_full_state.get('current_iteration', 0)} "
                        f"best_reachability={current_full_state['global_best_reachability']:.4f} "
                        f"target_reached={current_full_state.get('target_reached', False)} "
                        f"improvement={improvement:+.4f}"
                    )

                    # Report best_reachability as the primary progress signal.
                    best_reach = current_full_state.get("best_reachability", baseline_reachability)
                    reach_improvement = round(best_reach - baseline_reachability, 4)

                    yield {
                        "best_prompt": current_full_state["global_best_prompt"],
                        "best_score": best_reach,           # reachability as primary score
                        "best_reachability": best_reach,
                        "baseline_score": baseline_reachability,  # align baseline to reachability
                        "baseline_reachability": baseline_reachability,
                        "improvement": reach_improvement,
                        "current_iteration": current_full_state["current_iteration"],
                        "iterations_completed": current_full_state.get("iterations_completed", 0),
                        "target_reached": current_full_state.get("target_reached", False),
                        "feedback": current_full_state.get("last_feedback", ""),
                    }