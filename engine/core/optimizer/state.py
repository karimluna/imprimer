'''
Shared state for the LangGraph optimization loop.
'''
from typing import TypedDict


class PromptState(TypedDict):
    # Task definition, never changes across iterations
    task: str
    input_example: str
    expected_output: str
    backend: str           # always a string (ModelBackend.value), never the enum
    use_judge: bool
    base_prompt: str       # immutable anchor, generator always reads this
    use_rpe: bool # true: use reflective prompt optimization / false use bayesian search

    # Control parameters
    target_score: float
    max_iterations: int
    n_trials: int
    n_variants: int

    # Current state
    current_prompt: str        # decorated candidate for this cycle
    current_iteration: int

    # RPE feedback, verbal explanation of why last iteration's best prompt won.
    # Passed to the generator each cycle so Optuna's persona slot can use it.
    # Starts as a neutral seed, updated by evaluator_node after each cycle.
    last_feedback: str

    # Best by reachability, used by controller for termination
    best_prompt: str
    best_reachability: float
    best_score: float

    # Global best by combined score, returned to caller
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