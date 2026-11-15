"""Dense Semantic Retrieval (`sentence-transformers/all-MiniLM-L6-v2`) with in-memory caching (Sections 7.1, 10.1)."""

from __future__ import annotations

import numpy as np

from src.retrieval.bm25 import build_event_search_text
from src.schemas import Event

_MODEL_INSTANCE = None
_MODEL_NAME_LOADED = None
_TEXT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def get_dense_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the SentenceTransformer model once per process to keep memory footprint low."""
    global _MODEL_INSTANCE, _MODEL_NAME_LOADED
    if _MODEL_INSTANCE is None or _MODEL_NAME_LOADED != model_name:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        try:
            _MODEL_INSTANCE = SentenceTransformer(
                model_name, device="cpu", local_files_only=True
            )
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _MODEL_INSTANCE = SentenceTransformer(model_name, device="cpu")
        _MODEL_NAME_LOADED = model_name
    return _MODEL_INSTANCE


def encode_texts_cached(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode texts into L2-normalized embeddings, caching unique strings in memory."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    missing = [t for t in set(texts) if t not in _TEXT_EMBEDDING_CACHE]
    if missing:
        model = get_dense_model(model_name)
        vecs = model.encode(
            missing,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for t, v in zip(missing, vecs):
            _TEXT_EMBEDDING_CACHE[t] = v.astype(np.float32)

    return np.stack([_TEXT_EMBEDDING_CACHE[t] for t in texts], axis=0)


def dense_scores(
    candidate_events: list[Event],
    query_text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Compute cosine similarity scores between `query_text` and `candidate_events`."""
    if not candidate_events:
        return {}
    event_texts = [build_event_search_text(ev) for ev in candidate_events]
    event_vecs = encode_texts_cached(event_texts, model_name=model_name)
    query_vec = encode_texts_cached([query_text], model_name=model_name)[0]
    sims = np.dot(event_vecs, query_vec)
    return {
        ev.event_id: float(sim) for ev, sim in zip(candidate_events, sims)
    }
