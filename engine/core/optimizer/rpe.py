"""
Reflective Prompt Engineering helpers
"""

import json
import re
from typing import Optional

from core.chains.prompt_chain import ModelBackend, call_llm
from utils.create_logger import get_logger

logger = get_logger(__name__)

_MIN_PROMPT_LEN = 20

_LABEL_PATTERN = re.compile(r"^\s*[\w'-]{1,20}(\s+[\w'-]{1,15}){0,2}\s*\d*\s*[.:]?\s*$")


def _is_valid_prompt(text: str, anchor: str) -> bool:
    s = text.strip()
    return (
        len(s) >= _MIN_PROMPT_LEN
        and s != anchor.strip()
        and not s.endswith(":")          # preamble header in any language
        and not _LABEL_PATTERN.match(s)  # "version 3", "versión 3", "Version 3"
    )

def _is_constraint_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if ":" in s and s.index(":") < 40:  # "Output: label" / "Salida: etiqueta"
        return True
    words = s.split()
    if 3 <= len(words) <= 15 and s[-1] in ".!":  # short complete sentence
        return True
    return False



def extract_residual_content(prompt: str) -> str:
    """
    RiOT residual extractor.
    """
    stripped = prompt.strip()
    if not stripped:
        return ""

    lines = [l.strip() for l in stripped.splitlines() if l.strip()]

    residual = [l for l in lines if _is_constraint_line(l)]
    logger.debug(f"riot residual: extracted {len(residual)}/{len(lines)} constraint lines")
    return "\n".join(residual)


def _generate_variants_with_residual(
    base_prompt: str,
    feedback: str,
    n_variants: int,
    backend: ModelBackend,
    task: str,
    current_best_prompt: Optional[str] = None,
    residual_content: str = "",
) -> list[str]:
    """
    Calls the GENERATOR model to produce N improved prompt variants.

    RiOT injection: residual_content (proven constraints) is presented as
    lines the model must preserve, preventing semantic drift across cycles.

    Parsing pipeline (robust for models that may not output clean JSON):
      1. Strict JSON array
      2. Quoted-string extraction
      3. Non-empty line fallback
      4. [anchor] on total failure, loop never crashes
    """
    anchor = current_best_prompt or base_prompt

    feedback_block = f"\nPrevious feedback:\n{feedback}\n" if feedback.strip() else ""
    residual_block = (
        f"\nPreserve these constraints in every version:\n{residual_content}\n"
    ) if residual_content.strip() else ""

    generation_prompt = (
        f"You are improving an AI prompt for the task: {task}\n\n"
        f"Current best prompt:\n{anchor}\n"
        f"{feedback_block}"
        f"{residual_block}"
        f"Write {n_variants} improved versions. Each must:\n"
        f"- Keep {{input}} exactly as written\n"
        f"- Change only wording, tone, or instruction style (one change per version)\n"
        f"- Be a complete, usable instruction\n\n"
        f"Respond with ONLY a JSON array, no other text. Example format:\n"
        f'["improved prompt 1", "improved prompt 2"]'
    )

    raw = ""
    try:
        raw = call_llm(
            prompt_text=generation_prompt,
            backend=backend,
            temperature=0.7,
            max_tokens=600,
        )

        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if match:
            try:
                variants = json.loads(match.group())
                valid = [v for v in variants if isinstance(v, str) and _is_valid_prompt(v, anchor)]
                if valid:
                    logger.info(f"rpe: {len(valid)} variants (JSON)")
                    return valid[:n_variants]
            except json.JSONDecodeError:
                pass

        # quoted fallback: handles models that write "prompt" instead of JSON
        quoted = re.findall(r'"([^"]{20,})"', cleaned)
        valid  = [v.strip() for v in quoted if _is_valid_prompt(v, anchor)]
        if valid:
            logger.info(f"rpe: {len(valid)} variants (quoted fallback)")
            return valid[:n_variants]

    except Exception as exc:
        logger.warning(f"variant generation failed: {exc} — returning anchor")

    if raw:
        logger.warning(f"all parsers failed. Raw: {raw[:200]!r}")

    return [anchor]