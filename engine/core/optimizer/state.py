'''
Shared state for the LangGraph optimization loop. 
'''

from typing import TypedDict

class PromptState(TypedDict):
    '''
    State for the prompt optimization loop.
    '''
    task: str # never changes across iterations
    input_example: str
    expected_output: str
    backend: str
    use_judge: bool

    # Control parameters
    target_reachability: float  # stop when?
    max_iterations: int         # hard cap on graph cycles
    n_trials: int

    # Current state
    current_prompt: str         # prompt being refined this cycle
    current_iteration: int

    # Best found so far
    best_prompt: str
    best_reachability: float
    best_score: float

    # Baseline
    baseline_score: float
    baseline_reachability: float

    # Full history across all cycles
    history: list

    # Terminal flags
    target_reached: bool
    iterations_comleted: int