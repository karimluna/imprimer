"""
Bayesian search via Optuna + spaCy structured mutation engine.

Optuna's Tree-structured Parzen Estimator (TPE) is a Bayesian optimization
algorithm that models the distribution of high-scoring parameters directly.

TPE maintains two models:
   l(x): probability density of x given score > threshold (good region)
   g(x): probability density of x given score < threshold (bad region)

It samples candidates that maximize l(x)/g(x), the ratio of good to bad
region. Like searching for the control input (prompt) that maximizes steerage
of the output distribution toward target behavior, measured by reachability
and similarity.

Mutation engine:
   spaCy parses the base prompt's dependency tree at startup and generates
   mutations that operate on its actual linguistic structure.

   Three mutators, each targeting a different axis of the prompt:
     - VerbMutator     : rewrites the root verb (what the model is asked to do)
     - NounMutator     : rewrites the primary object noun chunk (what it acts on)
     - ModalityMutator : shifts surface mood (imperative / directive / interrogative)
"""

import re
import optuna
import uuid
import spacy
from dataclasses import dataclass, field
from typing import Literal
import hashlib


from core.chains.prompt_chain import run_variant, ModelBackend
from core.evaluator.scorer import _similarity
from core.evaluator.scorer import score as compute_score
from core.registry.prompt_store import (
    OptimizationTrialRecord,
    save_optimization_trial,
    mark_best_optimization_trial,
)
from utils.create_logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    # declared one time globally for each mutator class
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy 'en_core_web_sm' model...")
    from spacy.cli import download

    download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")


# Literal type for the dimension parameter
Dimension = Literal["verb", "noun", "modality"]


# Structured Mutation Engine
class VerbMutator:
    """
    Finds the root verb of the base prompt and rewrites it using registered
    strategies matched against the root verb's lemma.

    Candidates are pre-computed at parse time. Optuna samples keys, not
    strings, keeping the search space discrete and stable across trials.
    """

    # (match_lemmas, replacement_text)
    # Empty match_lemmas means the strategy applies to any root verb.
    _STRATEGIES: dict[str, tuple[set[str], str]] = {
        "none": (set(), ""),
        "extract": ({"summarize"}, "Extract the core idea from"),
        "distill": ({"summarize"}, "Distill the main point of"),
        "condense": ({"summarize"}, "Condense"),
        "breakdown": ({"explain"}, "Break down"),
        "clarify": ({"explain"}, "Clearly explain"),
        "articulate": ({"describe"}, "Articulate"),
        "outline": ({"describe", "explain"}, "Outline"),
    }

    # If mutation produces a prompt shorter than this fraction of the
    # original, treat it as a broken rewrite and skip it.
    _MIN_LENGTH_RATIO = 0.5

    def __init__(self, base_prompt: str):
        self._base = base_prompt
        self._doc = _nlp(base_prompt)
        self._roots = [tok for tok in self._doc if tok.dep_ == "ROOT"]
        self._applicable = self._resolve_applicable()

    def _resolve_applicable(self) -> dict[str, str]:
        applicable = {"none": self._base}
        root_lemmas = {r.lemma_.lower() for r in self._roots}

        for key, (match_lemmas, replacement) in self._STRATEGIES.items():
            if key == "none":
                continue
            if not match_lemmas or root_lemmas & match_lemmas:
                mutated = self._apply(replacement)
                if (
                    mutated != self._base
                    and len(mutated) >= len(self._base) * self._MIN_LENGTH_RATIO
                ):
                    applicable[key] = mutated

        return applicable

    def _apply(self, replacement: str) -> str:
        if not self._roots or not replacement:
            return self._base

        root = self._roots[0]
        result = []
        replaced = False

        for tok in root.sent:
            if tok.i == root.i and not replaced:
                result.append(replacement)
                replaced = True
            else:
                result.append(tok.text_with_ws)

        first_sent = "".join(result).strip()
        suffix = str(self._doc)[root.sent.end_char :]
        return (first_sent + suffix).strip()

    @property
    def keys(self) -> list[str]:
        return list(self._applicable.keys())

    def apply(self, key: str) -> str:
        return self._applicable.get(key, self._base)


class NounMutator:
    """
    Rewrites the primary object noun chunk (dobj / pobj / xcomp) to shift
    what the model understands it is acting on.
    """

    _REWRITES: dict[str, dict[str, str]] = {
        "determiner": {
            "the": "this",
            "a": "the given",
            "an": "the following",
        },
        "noun": {
            "document": "text",
            "article": "passage",
            "paragraph": "excerpt",
            "text": "content",
            "concept": "idea",
            "topic": "subject",
        },
    }

    def __init__(self, base_prompt: str):
        self._base = base_prompt
        self._doc = _nlp(base_prompt)
        self._candidates = self._build_candidates()

    def _build_candidates(self) -> dict[str, str]:
        candidates = {"none": self._base}

        for chunk in self._doc.noun_chunks:
            if chunk.root.dep_ not in {"dobj", "xcomp", "pobj"}:
                continue
            for tok in chunk:
                if tok.dep_ == "det" and tok.lower_ in self._REWRITES["determiner"]:
                    new = self._REWRITES["determiner"][tok.lower_]
                    candidates[f"det:{tok.lower_}->{new}"] = self._replace_token(
                        tok, new
                    )
                if tok.pos_ == "NOUN" and tok.lower_ in self._REWRITES["noun"]:
                    new = self._REWRITES["noun"][tok.lower_]
                    candidates[f"noun:{tok.lower_}->{new}"] = self._replace_token(
                        tok, new
                    )

        return candidates

    def _replace_token(self, token: spacy.tokens.Token, replacement: str) -> str:
        s, e = token.idx, token.idx + len(token.text)
        return (self._base[:s] + replacement + self._base[e:]).strip()

    @property
    def keys(self) -> list[str]:
        return list(self._candidates.keys())

    def apply(self, key: str) -> str:
        return self._candidates.get(key, self._base)


class ModalityMutator:
    """
    Shifts the surface mood of the instruction without changing its intent.
    Example:
    imperative   : "Summarize the document."
    directive    : "Your task is to summarize the document."
    interrogative: "Could you summarize the document."
    """

    def __init__(self, base_prompt: str):
        self._base = base_prompt
        self._candidates = self._build_candidates()

    def _build_candidates(self) -> dict[str, str]:
        candidates = {"none": self._base}

        doc = _nlp(self._base)
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        if not roots:
            return candidates

        verb = roots[0].text

        # imperative: strip leading "Please" if present
        stripped = re.sub(r"^Please\s+", "", self._base).strip()
        if stripped != self._base:
            candidates["imperative"] = stripped

        # directive and interrogative: reframe the opening
        for key, prefix in (
            ("directive", f"Your task is to {verb.lower()}"),
            ("interrogative", f"Could you {verb.lower()}"),
        ):
            rewritten = re.sub(
                r"^(?:Please\s+)?" + re.escape(verb),
                prefix,
                self._base,
                count=1,
            ).strip()
            if rewritten != self._base:
                candidates[key] = rewritten

        return candidates

    @property
    def keys(self) -> list[str]:
        return list(self._candidates.keys())

    def apply(self, key: str) -> str:
        return self._candidates.get(key, self._base)


# Static combinatorial search space
# Semantic signal that spaCy cannot generate
PERSONAS = [
    "",
    "You are an expert in this domain.",
    "You are a precise, no-nonsense assistant.",
    "You are a scientific writing assistant.",
    "You are a senior researcher.",
]

DIMENSION_SEQUENCE = ["verb", "noun", "modality"]

PRIMING = [
    "",
    "Think carefully before responding.",
    "First identify the key concept, then respond.",
    "Strip all filler. Only keep what matters.",
    "Read the input twice before answering.",
]

OUTPUT_CONTRACTS = [
    "",
    "Return exactly one sentence.",
    "Output only the answer, no preamble.",
    "Start your response directly with the answer.",
    "No bullet points. Plain prose only.",
]

HEDGING_SUPPRESSION = [
    "",
    "Do not hedge or qualify your answer.",
    "Avoid phrases like 'it depends' or 'generally speaking'.",
    "Be definitive.",
]

SKELETONS = [
    "{persona}{task}\n{constraints}",
    "{persona}{constraints}\n{task}",
    "{persona}Task: {task}\n{constraints}",
    "{persona}<task>{task}</task>\n{constraints}",
]


def build_prompt(
    task_text: str,
    persona: str,
    priming: str,
    output_contract: str,
    hedging: str,
    skeleton: str,
) -> str:
    """
    Assembles the final prompt from a pre-mutated task string and constraint
    slots. Mutation is fully resolved before this call assembly is pure
    formatting with no hidden branching logic.
    """
    constraint_parts = [c for c in (priming, output_contract, hedging) if c]
    constraints = "\n".join(constraint_parts)
    persona_block = (persona + "\n") if persona else ""

    return skeleton.format(
        persona=persona_block,
        task=task_text,
        constraints=constraints,
    ).strip()


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


def optimize(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str | None = None,
    n_trials: int = 20,
    backend: ModelBackend = ModelBackend.OLLAMA,
    storage: str | None = None,
    study_name: str | None = None,
    dimension: Dimension = "verb",
    reflection_hints: dict | None = None,
) -> OptimizationResult:
    """
    Runs Bayesian optimization (TPE) over one mutation dimension at a time.
    """
    if dimension not in ("verb", "noun", "modality"):
        raise ValueError(
            f"dimension must be 'verb', 'noun', or 'modality', got '{dimension}'"
        )

    history = []
    run_id = uuid.uuid4().hex
    local_cache = {}  # Initialize cache for this optimization run

    # Build all three mutators once
    verb_mutator = VerbMutator(base_prompt)
    noun_mutator = NounMutator(base_prompt)
    modality_mutator = ModalityMutator(base_prompt)

    # Select the one mutator that is active for this dimension
    active_mutator: VerbMutator | NounMutator | ModalityMutator = {
        "verb": verb_mutator,
        "noun": noun_mutator,
        "modality": modality_mutator,
    }[dimension]

    logger.info(
        f"dimension={dimension} active_keys={active_mutator.keys} n_trials={n_trials}"
    )

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
        expected_output=expected_output or "",
    )

    if expected_output:
        baseline_sim = _similarity(baseline_result.text, expected_output)
        baseline_score = 0.6 * baseline_sim + 0.4 * baseline_score_obj.reachability
    else:
        baseline_score = baseline_score_obj.combined

    def objective(trial: optuna.Trial) -> float:
        # Static slots
        persona = trial.suggest_categorical("persona", PERSONAS)
        priming = trial.suggest_categorical("priming", PRIMING)
        output_contract = trial.suggest_categorical("output_contract", OUTPUT_CONTRACTS)
        hedging = trial.suggest_categorical("hedging", HEDGING_SUPPRESSION)
        skeleton = trial.suggest_categorical("skeleton", SKELETONS)

        # Active mutation dimension only.
        mutation_key = trial.suggest_categorical(
            f"{dimension}_mutation", active_mutator.keys
        )
        task_text = active_mutator.apply(mutation_key)

        candidate = build_prompt(
            task_text=task_text,
            persona=persona,
            priming=priming,
            output_contract=output_contract,
            hedging=hedging,
            skeleton=skeleton,
        )

        mut_label = (
            f"dim={dimension}"
            f"|key={mutation_key}"
            f"|skel={SKELETONS.index(skeleton)}"
            f"|persona={persona[:12].strip()}"
        )

        cache_key_string = f"{backend.value}|{candidate}"
        key = hashlib.sha256(cache_key_string.encode("utf-8")).hexdigest()

        if key in local_cache:
            # Cache Hit
            cached_data = local_cache[key]
            combined = cached_data["combined"]
            sim = cached_data["similarity"]
            reachability = cached_data["reachability"]
            latency_ms = cached_data["latency_ms"]
        else:
            # Cache Miss: Run evaluation
            result = run_variant(
                template=candidate,
                input_text=input_example,
                task=task,
                backend=backend,
            )
            s = compute_score(
                result=result,
                baseline_result=baseline_result,
                task=task,
                input_text=input_example,
                expected_output=expected_output,
            )

            sim = s.similarity
            combined = s.combined
            reachability = s.reachability
            latency_ms = result.latency_ms

            # Save to cache
            local_cache[key] = {
                "combined": combined,
                "similarity": sim,
                "reachability": reachability,
                "latency_ms": latency_ms,
            }

        # optuna and db recording
        trial.set_user_attr("prompt", candidate)
        trial.set_user_attr("reachability", reachability)
        trial.set_user_attr("similarity", sim)
        trial.set_user_attr("latency_ms", latency_ms)
        trial.set_user_attr("mutation_key", mut_label)

        history.append(
            {
                "trial": trial.number,
                "mutation": mut_label,
                "prompt": candidate,
                "score": combined,
                "reachability": reachability,
                "similarity": sim,
                "latency_ms": latency_ms,
            }
        )

        save_optimization_trial(
            record=OptimizationTrialRecord(
                run_id=run_id,
                task=task,
                backend=backend.value,
                base_prompt=base_prompt,
                candidate_prompt=candidate,
                mutation=mut_label,
                trial_number=trial.number,
                score=combined,
                reachability=reachability,
                similarity=sim,
                latency_ms=latency_ms,
                is_best=False,
            )
        )

        logger.info(
            f"trial={trial.number} "
            f"mutation={mut_label} "
            f"score={combined:.4f} "
            f"reachability={reachability:.4f} "
            f"similarity={sim:.4f} "
            f"latency={latency_ms:.0f}ms"
        )

        return combined

    # Study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=12,
            multivariate=True,
            gamma=0.25,  # 25% of trials will explore wider search spaces to prevent plateaus
            seed=17,
        ),
        # Removed Pruner here since you evaluate in single-shot (no batches/steps)
        storage=storage,
        study_name=study_name or f"imprimer_{task}_{dimension}",
        load_if_exists=True,
    )

    # If the parent graph node passed in reflection hints, queue them as the very first trial
    if reflection_hints:
        logger.info(f"Enqueueing reflection hints into study: {reflection_hints}")
        study.enqueue_trial(reflection_hints)

    study.optimize(objective, n_trials=n_trials)

    # Result
    best_trial = study.best_trial
    best_prompt = best_trial.user_attrs.get("prompt", base_prompt)
    best_reachability = best_trial.user_attrs.get("reachability", 0.0)
    best_mutation_key = best_trial.user_attrs.get("mutation_key", "unknown")

    mark_best_optimization_trial(run_id=run_id, trial_number=best_trial.number)

    improvement = round(best_trial.value - baseline_score, 4)

    logger.info(
        f"optimization complete "
        f"task={task} "
        f"dimension={dimension} "
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
