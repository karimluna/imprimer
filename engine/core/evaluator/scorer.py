import math
from dataclasses import dataclass
from typing import Optional
import json
import hashlib


from core.chains.prompt_chain import VariantResult, ModelBackend
from core.evaluator.embedder import similarity as _similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)
_SCORE_CACHE = {}

OPEN_ENDED_TASKS = {
    "summarize",
    "creative_writing", 
    "roleplay", 
    "reasoning", 
    "code_generation", 
    "rewrite"
}


@dataclass
class Score:
    reachability: float
    latency_score: float
    combined: float
    quality_score: Optional[float] = None
    similarity: Optional[float] = None

LATENCY_BUDGET_MS = 1000.0
STEEP = 2.0
REACHABLE_THRESHOLD = math.log(0.40)

def _compute_reachability(logprobs: list, baseline_logprobs: Optional[list] = None) -> float:
    if not logprobs:
        return 0.5

    def get_avg_logprob(lps: list) -> float:
        valid_lps = [t.get("logprob", -10.0) for t in lps if t.get("logprob") is not None]
        if not valid_lps:
            return -10.0
        return sum(valid_lps) / len(valid_lps)

    variant_conf = get_avg_logprob(logprobs)
    
    if baseline_logprobs and len(baseline_logprobs) > 0:
        baseline_conf = get_avg_logprob(baseline_logprobs)
        improvement = variant_conf - baseline_conf
        score = 1.0 / (1.0 + math.exp(-STEEP * improvement))
    else:
        score = 1.0 / (1.0 + math.exp(-STEEP * (variant_conf - REACHABLE_THRESHOLD)))
        
    return round(score, 4)


def _creative_quality_heuristic(text: str) -> float:
    """
    Heuristic quality score for creative tasks when no logprobs
    or judge are available.
    
    Combines two signals:
      - Lexical diversity
      - Length adequacy.
    
    returns: float in [0.0, 1.0].
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    
    # lexical diversity, type-token ratio, capped at 1.0
    diversity = len(set(tokens)) / len(tokens)
    
    # length adequacy, sigmoid centered at 50 tokens
    # < 10 tokens scores near 0, > 100 tokens scores near 1
    import math
    length_score = 1.0 / (1.0 + math.exp(-0.1 * (len(tokens) - 50)))
    
    return round(0.6 * diversity + 0.4 * length_score, 4)


def score(
        result: VariantResult,
        baseline_result: Optional[VariantResult] = None,
        task: str = "",
        input_text: str = "",
        expected_output: str = "",
        use_judge: bool = False,
        backend: ModelBackend = ModelBackend.OLLAMA,
        weights: Optional[dict] = None
    ) -> Score:
    """
    Scores a variant result with consistent, flexible dimension weighting.

    FIX (Problem 2 — Similarity dead weight):
      Scenario D (classify/extract without expected_output) now returns
      similarity=0.5 (neutral) instead of 0.0. Returning 0.0 dragged every
      combined score toward zero regardless of reachability, making the
      optimizer appear far worse than it really was and causing sim=0.000 in
      all RPE variant logs.
    """

    cache_state = json.dumps({
        "result": result.text,
        "task": task,
        "expected_output": expected_output
    })

    key = hashlib.sha256(cache_state.encode('utf-8')).hexdigest()

    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    

    # Default weights guarantee stability across different configurations
    if weights is None:
        weights = {"quality": 0.20, "reachability": 0.60, "latency": 0.20}

    if baseline_result and baseline_result.logprobs:
        reachability = _compute_reachability(result.logprobs, baseline_logprobs=baseline_result.logprobs)
    elif result.logprobs:
        reachability = _compute_reachability(result.logprobs)
    else:
        reachability = _similarity(result.text, expected_output)

    # Latency
    latency_score = max(0.0, 1.0 - (result.latency_ms / LATENCY_BUDGET_MS))
    
    quality_score = 0.5
    similarity_score = 0.0

    if use_judge and task and input_text:
        # Scenario A: judge handles it intelligently (Best for creative/complex tasks)
        from core.evaluator.judge import judge
        quality_score = judge(task=task, input_text=input_text, output=result.text, backend=backend)
        
    elif task in OPEN_ENDED_TASKS:
        if result.logprobs:
            # Scenario B: creative task with logprobs. 
            # Max out quality and let the optimizer focus entirely on reachability (confidence).
            quality_score = 1.0 
        else:
            # Scenario C: creative task, no log probs, no judge
            similarity_score = 0.0
            quality_score = _creative_quality_heuristic(result.text)
            
    else:
        # Scenario D: Standard strict tasks (classify, extract, etc.)
        # FIX: when expected_output is absent, use 0.5 (neutral) not 0.0.
        if expected_output:
            similarity_score = _similarity(result.text, expected_output)
        else:
            similarity_score = 0.5  # neutral: no reference, no penalty
        quality_score = similarity_score

    # Consistent metric application
    combined = (
        weights["quality"] * quality_score + 
        weights["reachability"] * reachability + 
        weights["latency"] * latency_score
    )

    s = Score(
        reachability=reachability,
        latency_score=round(latency_score, 3),
        combined=round(combined, 3),
        quality_score=round(quality_score, 3),
        similarity=round(similarity_score, 3),
    )

    _SCORE_CACHE[key] = s
    
    return s