"""
Prompt registry.

Persists every evaluation result so prompt quality can be tracked
over time. A prompt that consistently scores high reachability
is a well-controlled prompt - the registry makes that visible.

Storage: SQLite for the MVP, zero infrastructure, file-based,
inspectable with any SQLite viewer. We may swap for Postgres in 
production by changing _get_conn() only. Nothing else changes.
"""
import sqlite3
import time
import json
from pathlib import Path
from dataclasses import dataclass
from utils.create_logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/prompt_registry.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates the registry table if it does not exist.
    Called once at engine startup.
    Safe to call multiple times - idempotent.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id      TEXT NOT NULL,
                task          TEXT NOT NULL,
                backend       TEXT NOT NULL,
                variant_a     TEXT NOT NULL,
                variant_b     TEXT NOT NULL,
                winner        TEXT NOT NULL,
                reachability_a REAL NOT NULL,
                reachability_b REAL NOT NULL,
                score_a       REAL NOT NULL,
                score_b       REAL NOT NULL,
                latency_a_ms  REAL NOT NULL,
                latency_b_ms  REAL NOT NULL,
                gap_report    TEXT,
                created_at    TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task
            ON evaluations(task)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace
            ON evaluations(trace_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_trials (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                task            TEXT NOT NULL,
                backend         TEXT NOT NULL,
                base_prompt     TEXT NOT NULL,
                candidate_prompt TEXT NOT NULL,
                mutation        TEXT NOT NULL,
                trial_number    INTEGER NOT NULL,
                score           REAL NOT NULL,
                reachability    REAL NOT NULL,
                similarity      REAL NOT NULL,
                latency_ms      REAL NOT NULL,
                is_best         INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_optimization_run
            ON optimization_trials(run_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_optimization_task
            ON optimization_trials(task)
        """)
    logger.info("Prompt registry initialized")


@dataclass
class EvalRecord:
    trace_id: str
    task: str
    backend: str
    variant_a: str
    variant_b: str
    winner: str
    reachability_a: float
    reachability_b: float
    score_a: float
    score_b: float
    latency_a_ms: float
    latency_b_ms: float
    gap_report: str = ""


def save(record: EvalRecord) -> int:
    """
    Persists one evaluation result to the registry.
    Returns the row ID for reference.
    """
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO evaluations (
                trace_id, task, backend,
                variant_a, variant_b, winner,
                reachability_a, reachability_b,
                score_a, score_b,
                latency_a_ms, latency_b_ms,
                gap_report, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.trace_id, record.task, record.backend,
            record.variant_a, record.variant_b, record.winner,
            record.reachability_a, record.reachability_b,
            record.score_a, record.score_b,
            record.latency_a_ms, record.latency_b_ms,
            record.gap_report, created_at,
        ))
        row_id = cursor.lastrowid

    logger.info(
        f"trace={record.trace_id} "
        f"registry_id={row_id} "
        f"winner={record.winner} "
        f"reachability_a={record.reachability_a} "
        f"reachability_b={record.reachability_b}"
    )
    return row_id


@dataclass
class OptimizationTrialRecord:
    run_id: str
    task: str
    backend: str
    base_prompt: str
    candidate_prompt: str
    mutation: str
    trial_number: int
    score: float
    reachability: float
    similarity: float
    latency_ms: float
    is_best: bool = False


def save_optimization_trial(record: OptimizationTrialRecord) -> int:
    """
    Persists one trial from the optimizer run.
    Returns the row ID for reference.
    """
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO optimization_trials (
                run_id, task, backend,
                base_prompt, candidate_prompt, mutation,
                trial_number, score, reachability,
                similarity, latency_ms, is_best,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_id, record.task, record.backend,
            record.base_prompt, record.candidate_prompt, record.mutation,
            record.trial_number, record.score, record.reachability,
            record.similarity, record.latency_ms,
            int(record.is_best), created_at,
        ))
        return cursor.lastrowid


def mark_best_optimization_trial(run_id: str, trial_number: int) -> None:
    """Marks the chosen best trial in the optimizer run."""
    with _get_conn() as conn:
        conn.execute("""
            UPDATE optimization_trials
            SET is_best = 1
            WHERE run_id = ? AND trial_number = ?
        """, (run_id, trial_number))


def best_variant_for_task(task: str, limit: int = 10) -> dict:
    """
    Returns the highest-scoring variant for a given task
    based on average reachability across recent evaluations.

    This is how Imprimer learns over time - the registry accumulates
    evidence about which prompts control the model most effectively
    for each task type, and this query surfaces that knowledge.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                winner,
                CASE WHEN winner = 'a' THEN variant_a ELSE variant_b END AS winning_template,
                AVG(CASE WHEN winner = 'a' THEN reachability_a ELSE reachability_b END) AS avg_reachability,
                AVG(CASE WHEN winner = 'a' THEN score_a ELSE score_b END) AS avg_score,
                COUNT(*) as evaluations
            FROM evaluations
            WHERE task = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (task, limit)).fetchall()

    if not rows:
        return {}

    return {
        "task": task,
        "best_template": rows[0]["winning_template"],
        "avg_reachability": round(rows[0]["avg_reachability"], 4),
        "avg_score": round(rows[0]["avg_score"], 4),
        "evaluations_sampled": rows[0]["evaluations"],
    }