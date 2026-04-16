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
import re
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

# --> Search space
# Who the model thinks it is. Affects prior over response style.
PERSONAS = [
    "",
    "You are an expert in this domain.",
    "You are a precise, no-nonsense assistant.",
    "You are a scientific writing assistant.",
    "You are a senior researcher.",
]

# Cognitive priming: how the model should approach the task before writing.
PRIMING = [
    "",
    "Think carefully before responding.",
    "First identify the key concept, then respond.",
    "Strip all filler. Only keep what matters.",
    "Read the input twice before answering.",
]

# Hard contracts on output shape.
OUTPUT_CONTRACTS = [
    "",
    "Return exactly one sentence.",
    "Output only the answer, no preamble.",
    "Start your response directly with the answer.",
    "No bullet points. Plain prose only.",
]

# Suppress hedging/verbosity that inflates output without adding signal.
HEDGING_SUPPRESSION = [
    "",
    "Do not hedge or qualify your answer.",
    "Avoid phrases like 'it depends' or 'generally speaking'.",
    "Be definitive.",
]

# Verb rewrites: change how the task action is phrased.
# Keys are registered as a categorical so TPE can learn which framing works.
TASK_VERB_REWRITES = {
    "none":        lambda p: p,
    "imperative":  lambda p: re.sub(r'\bSummarize\b', 'Extract the core idea from', p),
    "distill":     lambda p: re.sub(r'\bSummarize\b', 'Distill the main point of', p),
    "breakdown":   lambda p: re.sub(r'\bExplain\b', 'Break down', p),
    "clarify":     lambda p: re.sub(r'\bExplain\b', 'Clearly explain', p),
}

# Structural skeletons: vary WHERE in the prompt the task instruction sits
# relative to persona and constraints.
# {persona}, {task}, {constraints} are filled at build time.
SKELETONS = [
    # Default: persona → task → constraints
    "{persona}{task}\n{constraints}",
    # Constraints first, then task (useful for rule-heavy prompts)
    "{persona}{constraints}\n{task}",
    # Task wrapped in an explicit label
    "{persona}Task: {task}\n{constraints}",
    # XML-style wrapping (some models respond well to this)
    "{persona}<task>{task}</task>\n{constraints}",
]


def build_prompt(
    base_prompt: str,
    persona: str,
    priming: str,
    output_contract: str,
    hedging: str,
    verb_rewrite_key: str,
    skeleton: str,
) -> str:
    # Apply verb rewrite to the base task instruction
    task = TASK_VERB_REWRITES[verb_rewrite_key](base_prompt)

    # Collect non-empty constraint lines
    constraint_parts = [c for c in (priming, output_contract, hedging) if c]
    constraints = "\n".join(constraint_parts)

    # Build persona prefix (with trailing newline if present)
    persona_block = (persona + "\n") if persona else ""

    prompt = skeleton.format(
        persona=persona_block,
        task=task,
        constraints=constraints,
    ).strip()

    return prompt


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
             None means in-memory study, faster but not persistent.

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
        trial.suggest_categorical picks a value per dimension 
        TPE chooses based on which values scored highest in past trials.
        """
        persona          = trial.suggest_categorical("persona", PERSONAS)
        priming          = trial.suggest_categorical("priming", PRIMING)
        output_contract  = trial.suggest_categorical("output_contract", OUTPUT_CONTRACTS)
        hedging          = trial.suggest_categorical("hedging", HEDGING_SUPPRESSION)
        verb_rewrite_key = trial.suggest_categorical("verb_rewrite", list(TASK_VERB_REWRITES.keys()))
        skeleton         = trial.suggest_categorical("skeleton", SKELETONS)

        candidate = build_prompt(
            base_prompt=base_prompt,
            persona=persona,
            priming=priming,
            output_contract=output_contract,
            hedging=hedging,
            verb_rewrite_key=verb_rewrite_key,
            skeleton=skeleton,
        )

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

        # Human-readable key summarising this trial's configuration
        mutation_key = (
            f"verb={verb_rewrite_key}"
            f"|skel={SKELETONS.index(skeleton)}"
            f"|persona={persona[:12].strip()}"
            f"|priming={priming[:12].strip()}"
        )

        # Store extra data on the trial for later analysis
        trial.set_user_attr("prompt", candidate)
        trial.set_user_attr("reachability", s.reachability)
        trial.set_user_attr("similarity", sim)
        trial.set_user_attr("latency_ms", result.latency_ms)
        trial.set_user_attr("mutation_key", mutation_key)

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
    best_mutation_key = best_trial.user_attrs.get("mutation_key", "unknown")

    mark_best_optimization_trial(run_id=run_id, trial_number=best_trial.number)

    improvement = round(best_trial.value - baseline_score, 4)

    logger.info(
        f"optimization complete "
        f"task={task} "
        f"best_score={best_trial.value:.4f} "
        f"best_reachability={best_reachability:.4f} "
        f"improvement={improvement:+.4f} "
        f"best_mutation={best_mutation_key}"
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