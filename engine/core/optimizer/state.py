'''
Shared state for the LangGraph optimization loop.
'''
from typing import TypedDict


class PromptState(TypedDict):
    run_id: str

    # Task definition, never changes across iterations
    task: str
    input_example: str
    expected_output: str
    backend: str           # always a string (ModelBackend.value), never the enum
    use_judge: bool
    base_prompt: str       # immutable anchor, generator always reads this
    use_rpe: bool          # true: RPE / false: Bayesian search

    # Control parameters
    target_score: float
    max_iterations: int
    n_trials: int
    n_variants: int

    # Current state
    current_prompt: str        # decorated candidate for this cycle
    current_iteration: int
    current_candidate_ssc: float

    # RPE feedback verbal explanation of why last iteration's best prompt won.
    last_feedback: str

    # Best by PRIMARY CONTROL SIGNAL, used for promotion and termination.

    best_prompt: str
    best_reachability: float
    best_ssc: float            # SSC of the current best; fallback when no logprobs
    best_score: float
    logprobs_available: bool   # set by evaluator on first run, read by all nodes

    # Global best by combined score, returned to caller for UI display
    global_best_prompt: str
    global_best_score: float
    global_best_reachability: float

    # Baseline, set once, never changes
    baseline_score: float
    baseline_reachability: float

    # Full history across all cycles
    history: list

    # Terminal flags
    target_reached: bool
    iterations_completed: int