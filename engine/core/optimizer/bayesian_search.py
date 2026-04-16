'''
Bayesian search via Optuna.

Optuna's Tree-structured Parzen Estimator (TPE) is a Bayesian optimization
algorithm that models the distribution of high-scoring parameters directly.

TPE maintains two models:
   l(x): probability density of x given score > threshold (good region)
   g(x): probability density of x given score < threshold (bad region)

It samples candidates that maximize l(x)/g(x), the ratio of good in bad re-
gion. Like searching for the control input (prompt) that maximizes steerage
of the output distribution toward target behavior, measured by reachability
and similarity.
'''
import optuna
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from core.chains.prompt_chain import run_variant, ModelBackend
from core.evaluator.scorer import score as compute_score
from core.registry.prompt_store import (
    OptimizationTrialRecord,
    save_optimization_trial,
    mark_best_optimization_trial,
)
from utils.create_logger import get_logger

logger = get_logger(__name__)

# supress optuna default verbose
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Categorical search space
MUTATIONS = {
    "concise":      lambda p: p + "\nBe concise.",                          # Zero-shot learning 
    "precise":      lambda p: p + "\nBe precise and factual.",              # One-shot "
    "structured":   lambda p: p + "\nReturn structured output.",
    "stepbystep":   lambda p: p + "\nThink step by step before answering.", # Few-shot "
    "expert":       lambda p: "You are an expert. " + p,
    "no_fluff":     lambda p: p + "\nAvoid unnecessary words.",
    "rewrite_sum":  lambda p: p.replace("Summarize", "Concisely summarize"),
    "rewrite_exp":  lambda p: p.replace("Explain", "Clearly explain"),
}

MUTATION_KEYS = list(MUTATIONS.keys())


@dataclass 
class OptimizationResult:
    best_prompt: str
    best_score: float
    best_reachability: float
    baseline_score: float
    baseline_reachability: float
    trials_run: int
    improvement: float
    history: list = field(default_factory=list)


def _similarity(output: str, expected: str) -> float:
    """
    SequenceMatcher ratio: 1.0 identical, 0.0 like fire to ice.
    """
    return SequenceMatcher(None, output, expected).ratio()


def optimize(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    n_trials: int = 20,
    backend: ModelBackend = ModelBackend.OLLAMA,
    storage: str | None = None,
    study_name: str | None = None,
) -> OptimizationResult:
    """
    Runs Bayesian optimization (TPE) over the mutation space.

    storage: optional SQLite path for persisting the study.
             Pass "sqlite:///data/optimizer.db" to resume across calls.
             None means in-memory study — faster but not persistent.

    study_name: name for the Optuna study. If storage is set and a study
                with this name already exists, Optuna resumes it
                meaning previous trials inform new ones automatically.
                This is the feedback loop at the optimizer level.

    n_trials: LLM calls to make. TPE needs ~5 random trials to bootstrap
              its model, then each subsequent trial is informed by history.
              Recommended minimum: 10. Sweet spot: 20-30.
    """
    history = []
    run_id = uuid.uuid4().hex

    # Baseline. Score the original prompt before any mutation.
    # This is the control: if optimization finds nothing better,
    # we return the base prompt unchanged.
    baseline_result = run_variant(
        template=base_prompt,
        input_text=input_example,
        task=task,
        backend=backend,
    )
    baseline_score_obj = compute_score(baseline_result)
    baseline_sim = _similarity(baseline_result.text, expected_output)
    baseline_score = 0.6 * baseline_sim + 0.4 * baseline_score_obj.reachability

    logger.info(
        f"task={task} "
        f"baseline_score={baseline_score:.4f} "
        f"baseline_reachability={baseline_score_obj.reachability:.4f} "
        f"n_trials={n_trials}"
    )

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna calls this on each trial.
        trial.suggest_categorical picks a mutation key —
        TPE chooses based on which keys scored highest in past trials.
        """
        mutation_key = trial.suggest_categorical("mutation", MUTATION_KEYS)
        mutation_fn = MUTATIONS[mutation_key]
        candidate = mutation_fn(base_prompt)

        result = run_variant(
            template=candidate,
            input_text=input_example,
            task=task,
            backend=backend,
        )

        s = compute_score(result)
        sim = _similarity(result.text, expected_output)

        # Objective: 60% similarity, 40% reachability
        # Similarity tells us if the content is right.
        # Reachability tells us if the prompt is in control.
        # A prompt that produces correct output by luck (low reachability)
        # is worse than one that reliably produces correct output (high reachability).
        combined = 0.6 * sim + 0.4 * s.reachability

        # Store extra data on the trial for later analysis
        trial.set_user_attr("prompt", candidate)
        trial.set_user_attr("reachability", s.reachability)
        trial.set_user_attr("similarity", sim)
        trial.set_user_attr("latency_ms", result.latency_ms)

        history.append({
            "trial": trial.number,
            "mutation": mutation_key,
            "prompt": candidate,
            "score": combined,
            "reachability": s.reachability,
            "similarity": sim,
            "latency_ms": result.latency_ms,
        })

        save_optimization_trial(
            record=OptimizationTrialRecord(
                run_id=run_id,
                task=task,
                backend=backend.value,
                base_prompt=base_prompt,
                candidate_prompt=candidate,
                mutation=mutation_key,
                trial_number=trial.number,
                score=combined,
                reachability=s.reachability,
                similarity=sim,
                latency_ms=result.latency_ms,
                is_best=False,
            )
        )

        logger.info(
            f"trial={trial.number} "
            f"mutation={mutation_key} "
            f"score={combined:.4f} "
            f"reachability={s.reachability:.4f} "
            f"similarity={sim:.4f} "
            f"latency={result.latency_ms:.0f}ms"
        )

        return combined

    # Create or resume study
    # direction="maximize" 
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=max(3, n_trials // 4),
            seed=42,
        ),
        storage=storage,
        study_name=study_name or f"imprimer_{task}",
        load_if_exists=True,  # resume if study already exists in storage
    )

    study.optimize(objective, n_trials=n_trials)

    # Extract best trial
    best_trial = study.best_trial
    best_prompt = best_trial.user_attrs.get("prompt", base_prompt)
    best_reachability = best_trial.user_attrs.get("reachability", 0.0)

    mark_best_optimization_trial(run_id=run_id, trial_number=best_trial.number)

    improvement = round(best_trial.value - baseline_score, 4)

    logger.info(
        f"optimization complete "
        f"task={task} "
        f"best_score={best_trial.value:.4f} "
        f"best_reachability={best_reachability:.4f} "
        f"improvement={improvement:+.4f} "
        f"best_mutation={best_trial.params['mutation']}"
    )

    return OptimizationResult(
        best_prompt=best_prompt,
        best_score=best_trial.value,
        best_reachability=best_reachability,
        baseline_score=baseline_score,
        baseline_reachability=baseline_score_obj.reachability,
        trials_run=len(history),
        improvement=improvement,
        history=history,
    )