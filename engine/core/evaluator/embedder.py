import os
from difflib import SequenceMatcher
from typing import Any

from utils.create_logger import get_logger
from huggingface_hub import login 

logger = get_logger(__name__)

_embedder: Any = None
_st_util: Any = None
_embedder_load_failed = False


def _ensure_embedder() -> None:
    global _embedder, _st_util, _embedder_load_failed
    if _embedder is not None or _embedder_load_failed:
        return

    try:
        from sentence_transformers import SentenceTransformer, util  

        model_name = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        hf_token = os.getenv("HF_TOKEN") 
        
        if hf_token:
            try:
                login(hf_token)
            except Exception as auth_exc:
                logger.warning("HF login failed, attempting to load model anyway: %s", auth_exc)

        logger.info("Loading sentence-transformers embedder: %s", model_name)
        _embedder = SentenceTransformer(model_name_or_path=model_name)
        _st_util = util
    except Exception as exc:
        _embedder_load_failed = True
        logger.warning(
            "Unable to load sentence-transformers embedder; similarity scoring will fall back to a lightweight comparator. Error: %s",
            exc,
        )


def _simple_text_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def similarity(output: str, expected: str) -> float:
    if not output.strip() or not expected.strip():
        return 0.0

    _ensure_embedder()
    if _embedder is None or _st_util is None:
        return _simple_text_similarity(output, expected)

    emb_out = _embedder.encode(output, convert_to_tensor=True)
    emb_exp = _embedder.encode(expected, convert_to_tensor=True)
    score = _st_util.cos_sim(emb_out, emb_exp).item()
    return round(max(0.0, score), 4)


def pairwise_similarity(outputs: list[str]) -> float:
    if len(outputs) < 2:
        return 1.0

    _ensure_embedder()
    if _embedder is None or _st_util is None:
        scores = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                scores.append(_simple_text_similarity(outputs[i], outputs[j]))
        return round(sum(scores) / len(scores), 4)

    embeddings = _embedder.encode(outputs, convert_to_tensor=True)
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = _st_util.cos_sim(embeddings[i], embeddings[j]).item()
            scores.append((sim + 1) / 2)

    return round(sum(scores) / len(scores), 4)
