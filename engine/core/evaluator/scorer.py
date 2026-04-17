import math
from dataclasses import dataclass
# from unittest import result # not needed yet

from sentence_transformers import SentenceTransformer, util as st_util
from core.chains.prompt_chain import VariantResult, ModelBackend


@dataclass
class Score:
    reachability: float
    latency_score: float
    combined: float
    quality_score: float | None = None
    similarity: float | None = None

# Latency budget: anything under this scores 1.0, degrades linearly after
LATENCY_BUDGET_MS = 1000.0


# all-MiniLM-L6-v2: 80MB, CPU-friendly, ~5ms per pair.
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _similarity(output: str, expected: str) -> float:
    """
    Cosine similarity between sentence embeddings.
    """
    if not output.strip() or not expected.strip():
        return 0.0
    emb_out = _embedder.encode(output,   convert_to_tensor=True)
    emb_exp = _embedder.encode(expected, convert_to_tensor=True)
    score   = st_util.cos_sim(emb_out, emb_exp).item()
    return round(max(0.0, score), 4)


def _compute_reachability(
        logprobs: list, 
        baseline_logprobs: list | None = None,  
        target_tokens: list[str] | None = None # in real life target_tokens are not something we can reliably identify
    ) -> float:
    """
    If baseline_logprobs provided, measure control (improvement over baseline).
    If not, fall back to comfort (absolute probability threshold).
    """
    if not logprobs:
        return 0.5

    REACHABLE_THRESHOLD = math.log(0.10)
    STEEP = 5.0

    token_scores = []
    
    # NEW: Use baseline if available
    use_baseline = baseline_logprobs and len(baseline_logprobs) == len(logprobs)
    
    for i, token_data in enumerate(logprobs):
        chosen_logprob = token_data.get("logprob", -10.0)
        
        if use_baseline:
            # CONTROL: How much did we improve over baseline?
            baseline_logprob = baseline_logprobs[i].get("logprob", -10.0)
            improvement = chosen_logprob - baseline_logprob
            # Sigmoid centered at 0 (no improvement)
            score = 1.0 / (1.0 + math.exp(-STEEP * improvement))
        else:
            # COMFORT: Absolute probability
            score = 1.0 / (1.0 + math.exp(-STEEP * (chosen_logprob - REACHABLE_THRESHOLD)))
        
        token_scores.append(score)

    return round(sum(token_scores) / len(token_scores), 4)


def score(
        result: VariantResult,
        baseline_result: VariantResult | None = None,
        task: str = "",
        input_text: str = "",
        expected_output: str = "",
        use_judge: bool = False,
        backend: ModelBackend = ModelBackend.OLLAMA,
        judge_threshold: float = 0.60,
    ) -> Score:
    """
    Scores a variant result on three dimensions:

    1. Reachability - did the prompt strongly control the output distribution?

    2. Latency - did the model respond within the budget?

    3. Length - did the response match the expected scope?

    The combined score weights reachability most heavily because that is
    Imprimer's core thesis - prompt control, not just prompt speed.
    """
    if baseline_result and baseline_result.logprobs:
        reachability = _compute_reachability(
            result.logprobs, 
            baseline_logprobs=baseline_result.logprobs
        )
    else:
        reachability = _compute_reachability(result.logprobs)

    latency_score = max(0.0, 1.0 - (result.latency_ms / LATENCY_BUDGET_MS))
    similarity_score = _similarity(result.text, expected_output)

    if use_judge and task and input_text:
        from core.evaluator.judge import judge
        quality_score = judge(task=task, input_text=input_text, output=result.text, backend=backend)
        # Also compute similarity anyway
        # Use quality_score in combined, similarity just for logging
        combined = (0.50 * quality_score + 0.30 * reachability + 0.20 * latency_score)
        
    else:
        # No judge, so quality_score = similarity 
        quality_score = similarity_score  
        combined = (0.50 * reachability + 0.30 * latency_score + 0.20 * similarity_score)
    
    return Score(
        reachability=reachability,
        latency_score=round(latency_score, 3),
        combined=round(combined, 3),
        quality_score=round(quality_score, 3),
        similarity=round(similarity_score, 3),
    )