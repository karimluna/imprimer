"""
Reflective Prompt Optimization 

Replaces the Optuna TPE inner loop with LLM-driven candidate generation.
Instead of searching over predefined mutation keys, the LLM generates its own
variant prompts based on the current best and verbal feedback from prior rounds.
---
Semantic Self-Consistency (SSC): Run the same prompt K times at temperature > 0.
                                 Average pairwise semantic similarity of the K outputs.
                                 High SSC -> prompt reliably steers the model to similar outputs.
                                 Low SSC -> model is uncertain, prompt leaves too much to chance.
---                                 
Here reachability is an optional metric. When the backend supports logprobs 
(e.g., ollama and openai). Mostly logprobs are unavailable, so SSC is more stable. 

FIX (Problem 3 — Generator base drift):
  run_rpe now accepts `current_best_prompt` in addition to `base_prompt`.
  Variant generation always builds on top of the CURRENT best prompt from the
  graph state, not the frozen original base_prompt. This closes the feedback
  loop that was causing every iteration to restart from scratch.
"""

import json
import re
import requests
from dataclasses import dataclass, field
import os
from typing import Optional

from core.chains.prompt_chain import ModelBackend, run_variant, _run_ollama
from core.evaluator.scorer import (
    _compute_reachability, 
    OPEN_ENDED_TASKS, 
    _creative_quality_heuristic
)
from core.evaluator.embedder import pairwise_similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)

SSC_RUNS = 2
SSC_TEMPERATURE = 0.7
N_VARIANTS = 5


@dataclass
class RPEResult:
    best_prompt: str
    best_score: float
    best_reachability: float  # 0.5 neutral when logprobs unavailable
    history: list = field(default_factory=list)


def _generate_variants(
    base_prompt: str,
    feedback: str,
    n_variants: int,
    backend: ModelBackend,
    task: str,
    # FIX: current_best_prompt is the evolving anchor, distinct from the frozen base_prompt
    current_best_prompt: Optional[str] = None,
) -> list[str]:
    """
    Verbalized sampling — asks the LLM to generate N improved prompt variants.

    The LLM receives the CURRENT BEST prompt (not the frozen original) and
    verbal feedback from the previous iteration. This ensures each RPE cycle
    genuinely builds on prior gains rather than restarting from scratch.

    FIX: uses `current_best_prompt` as the generation anchor when provided.

    returns: list of variant strings 
    """
    # Use the evolving best prompt as the anchor, fall back to base_prompt
    anchor_prompt = current_best_prompt if current_best_prompt else base_prompt

    feedback_section = (
        f"\nCRITICAL FEEDBACK FROM PREVIOUS ROUND:\n{feedback}\n\n"
        f"You MUST fix the issues mentioned in the feedback above."
        if feedback else ""
    )
    example_array_str = "[" + ", ".join([f'"variant {i+1}"' for i in range(n_variants)]) + "]"
    generation_prompt = f"""You are a prompt engineering expert. Your task is to improve the following instruction prompt.
    
    Current best prompt:
    {anchor_prompt}
    {feedback_section}
    
    Generate exactly {n_variants} improved variants of this prompt for the task: {task}.

    Rules:
    - Each variant must be a complete, standalone instruction.
    - Vary the structure and framing, but explicitly address the feedback provided.
    - Use {{{{input}}}} as the placeholder for user input (keep it exactly as-is).
    - Do not add explanations, just the variants.

    Respond with ONLY a JSON array of strings containing exactly {n_variants} variants, no markdown:
    {example_array_str}"""
    
    raw = ""
    try:
        if backend == ModelBackend.OLLAMA:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": generation_prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")

        elif backend == ModelBackend.OPENAI:
            from core.chains.prompt_chain import _build_openai_llm
            llm = _build_openai_llm()
            response = llm.invoke(generation_prompt)
            raw = response.content

        elif backend == ModelBackend.HUGGINGFACE:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                model=os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
                token=os.getenv("HF_TOKEN"),
            )
            response = client.chat_completion(
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            raw = response.choices[0].message.content

        # parses JSON array from response
        cleaned = re.sub(r'```json\s*', '', raw)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            variants = json.loads(match.group())
            # Validate: must be non-empty strings
            valid = [
                v for v in variants
                if isinstance(v, str) and v.strip() 
            ]
            if valid:
                logger.info(f"generated {len(valid)} valid variants")
                return valid[:n_variants]
        else:
            logger.warning(
                f"variant generation failed to output an array. Raw output: {raw[:60]}... "
                f"using current best prompt"
            )

    except Exception as e:
        logger.warning(f"variant generation failed: {e}, using current best prompt")

    return [anchor_prompt]

def _compute_ssc(
        prompt: str,
        input_example: str,
        task: str, 
        backend: ModelBackend,
        k: int = SSC_RUNS,
        temperature: float = SSC_TEMPERATURE,
) -> tuple[float, float, str]:
    """
    Semantic Self-Consistency score for one prompt. Runs the prompt K times and 
    computes average pairwise semantic similarity. Also returns the average 
    reachability if logprobs are available.

    returns (ssc_score, avg_reachability, sample_output)
    avg_reachability is 0.5 (neutral) when logprobs unavailable.
    """
    from langchain_core.prompts import PromptTemplate

    prompt_template = PromptTemplate(
        template=prompt, 
        input_variables=["task", "input"]
    )
    rendered = prompt_template.format(task=task, input=input_example)

    outputs = []
    reachabilities = []
    for _ in range(k):
        try:
            if backend == ModelBackend.OLLAMA:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
                resp = requests.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": rendered}],
                        "stream": False,
                        "logprobs": True,
                        "top_logprobs": 5,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.95,
                        },
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                raw_lp = data.get("logprobs") or []
                logprobs = [
                    {
                        "token": e.get("token", ""),
                        "logprob": e.get("logprob", -10.0),
                        "top": [
                            {"token": t["token"], "logprob": t["logprob"]}
                            for t in e.get("top_logprobs", [])
                        ],
                    }
                    for e in raw_lp
                ]
                reachabilities.append(_compute_reachability(logprobs))

            elif backend == ModelBackend.OPENAI:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=5,
                )
                response = llm.invoke(rendered)
                text = response.content
                try:
                    lp = response.response_metadata.get("logprobs", {})
                    lp_list = [
                        {
                            "token": td["token"],
                            "logprob": td["logprob"],
                            "top": [
                                {"token": t["token"], "logprob": t["logprob"]}
                                for t in td.get("top_logprobs", [])
                            ],
                        }
                        for td in lp.get("content", [])
                    ]
                    reachabilities.append(_compute_reachability(lp_list))
                except Exception:
                    reachabilities.append(0.5)

            else:
                # HuggingFace — no logprobs
                result = run_variant(
                    template=prompt,
                    input_text=input_example,
                    task=task,
                    backend=backend,
                )
                text = result.text
                reachabilities.append(0.5)  # neutral fallback

            outputs.append(text)

        except Exception as e:
            logger.warning(f"SSC run failed: {e}")

    if not outputs:
        return 0.0, 0.5, ""

    sample_output = outputs[0] if outputs else "" 

    ssc = pairwise_similarity(outputs) if len(outputs) > 1 else 0.5
    avg_reach = sum(reachabilities) / len(reachabilities) if reachabilities else 0.5

    return round(ssc, 4), round(avg_reach, 4), sample_output


def run_rpe(
    task: str,
    base_prompt: str,
    input_example: str,
    expected_output: str,
    backend: ModelBackend,
    feedback: str = "",
    n_variants: int = N_VARIANTS,
    ssc_runs: int = SSC_RUNS,
    weights: Optional[dict] = None,
    current_best_prompt: Optional[str] = None,
) -> RPEResult:
    
    if weights is None:
            if task in OPEN_ENDED_TASKS:
                # CREATIVE TASKS: No exact right answer. 
                # Prioritize consistency (SSC) and control (Reachability).
                weights = {"ssc": 0.5, "reach": 0.3, "sim": 0.2}
                logger.info("Using creative weights (prioritizing SSC)")
            else:
                # DETERMINISTIC TASKS (extract, classify, summarize): 
                # Prioritize getting the right answer (Sim) against the expected_output.
                if expected_output:
                    weights = {"ssc": 0.2, "reach": 0.2, "sim": 0.6}
                    logger.info("Using deterministic weights (prioritizing Similarity)")
                else:
                    weights = {"ssc": 0.4, "reach": 0.4, "sim": 0.2}
                    logger.info("Using deterministic weights without reference (SSC+Reachability)")
            
    from core.evaluator.embedder import similarity as semantic_sim

    logger.info(
        f"rpe task={task} "
        f"n_variants={n_variants} "
        f"ssc_runs={ssc_runs} "
        f"backend={backend.value}"
    )

    # 1 call generates all N variants — from the CURRENT BEST, not frozen base
    variants = _generate_variants(
        base_prompt=base_prompt,
        feedback=feedback,
        n_variants=n_variants,
        backend=backend,
        task=task,
        current_best_prompt=current_best_prompt,  # FIX: evolving anchor
    )

    history = []
    best_prompt = current_best_prompt if current_best_prompt else base_prompt
    best_score = -1.0
    best_reachability = 0.5

    for i, variant in enumerate(variants):
        # K calls per variant: SSC scoring
        # sample_output reused for similarity: no extra call needed
        ssc, reach, sample_output = _compute_ssc(
            prompt=variant,
            input_example=input_example,
            task=task,
            backend=backend,
            k=ssc_runs,
        )

        # reuse sample_output from SSC runs 
        if task in OPEN_ENDED_TASKS:
            sim = _creative_quality_heuristic(sample_output)
        elif task and expected_output:
            sim = semantic_sim(output=sample_output, expected=expected_output)
        else:
            # FIX (Problem 2): no expected_output → neutral 0.5, not 0.0
            # This prevents the similarity dimension from dragging down every
            # score to near-zero when the user leaves the reference field blank.
            sim = 0.5

        combined = (
            weights["ssc"] * ssc + 
            weights["reach"] * reach + 
            weights["sim"] * sim
        )
        
        combined = round(combined, 4)

        logger.info(
            f"variant={i} ssc={ssc:.4f} reach={reach:.4f} "
            f"sim={sim:.4f} combined={combined:.4f}"
        )

        history.append({
            "variant": variant,
            "ssc": ssc,
            "reachability": reach,
            "similarity": sim,
            "score": combined,
        })

        if combined > best_score:
            best_score = combined
            best_prompt = variant
            best_reachability = reach

    return RPEResult(
        best_prompt=best_prompt,
        best_score=best_score,
        best_reachability=best_reachability,
        history=history,
    )