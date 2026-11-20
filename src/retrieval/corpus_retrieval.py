"""Demo Clinical Corpus Retrieval (`src/retrieval/corpus_retrieval.py`) (Sections 6.4, 9.5 & Phase 5 Plan Part 1).

Rules enforced:
  1. Separate index over `data/corpus/demo_corpus.jsonl` (`CorpusChunk` contracts).
  2. Never mixed into patient event `[E#]` scores and off in all benchmark evaluation configs.
  3. Uses BM25 + MiniLM (`all-MiniLM-L6-v2`) + RRF (`k_rrf = 60`) with query = question + relevant selected event concepts.
  4. Relevance threshold: if no passage clears the threshold (or if `insufficient_evidence` is True / question is off-topic),
     returns `[]` so the UI renders no background box rather than weak/irrelevant text.
  5. Returns at most `top_k = 2` passages (`[C1]`, `[C2]`).
  6. Provides `assert_no_corpus_citation_in_claims(answer)` to verify that no `[C#]` or `C_###` ID ever supports
     a patient-specific claim.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.bm25 import tokenize_medical_text
from src.retrieval.dense import encode_texts_cached
from src.schemas import AnswerJSON, CorpusChunk, Event, load_jsonl
from src.temporal.history_relevance import extract_question_anchors, load_relation_table

GENERIC_NON_TOPIC_CONCEPTS = {
    "vital_signs",
    "outpatient_visit",
    "ed_visit",
    "visit",
    "triage_observation",
}

_CORPUS_CACHE: dict[str, dict[str, Any]] = {}


def load_demo_corpus(
    corpus_path: str | Path = "data/corpus/demo_corpus.jsonl",
) -> list[CorpusChunk]:
    """Load and validate all `CorpusChunk` entries from `demo_corpus.jsonl`."""
    return load_jsonl(CorpusChunk, corpus_path)


def _get_or_build_corpus_index(
    corpus_path: str | Path = "data/corpus/demo_corpus.jsonl",
) -> dict[str, Any]:
    key = str(Path(corpus_path).resolve())
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]

    chunks = load_demo_corpus(corpus_path)
    doc_texts = [
        f"{c.title}. {' '.join(c.concepts)}. {c.text}" for c in chunks
    ]
    tokenized = [tokenize_medical_text(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized)

    emb = encode_texts_cached(doc_texts)

    entry = {
        "chunks": chunks,
        "doc_texts": doc_texts,
        "bm25": bm25,
        "embeddings": emb,
    }
    _CORPUS_CACHE[key] = entry
    return entry


def retrieve_corpus_passages(
    question_text: str,
    selected_events: Optional[list[Event]] = None,
    insufficient_evidence: bool = False,
    top_k: int = 2,
    min_relevance_threshold: float = 0.38,
    corpus_path: str | Path = "data/corpus/demo_corpus.jsonl",
    relation_table_path: str | Path = "configs/relation_table.json",
) -> list[dict[str, Any]]:
    """Retrieve up to `top_k = 2` background corpus passages (`[C1]`, `[C2]`) that clear the relevance threshold.

    Returns a list of dicts:
      `{"citation_tag": "[C1]", "chunk": CorpusChunk, "score": float, "dense_sim": float, "bm25_score": float, "matched_concepts": list[str]}`
    If `insufficient_evidence` is True or no passage clears `min_relevance_threshold`, returns `[]`.
    """
    if insufficient_evidence or not question_text or not question_text.strip():
        return []

    idx_data = _get_or_build_corpus_index(corpus_path)
    chunks: list[CorpusChunk] = idx_data["chunks"]
    if not chunks:
        return []

    rel_table = load_relation_table(relation_table_path)
    q_anchors = set(extract_question_anchors(question_text, rel_table))

    event_concepts: list[str] = []
    if selected_events:
        allowed_from_anchors: set[str] = set(q_anchors)
        for a in q_anchors:
            entry = rel_table.get("concepts", {}).get(a, {})
            for r in entry.get("related_concepts", []):
                allowed_from_anchors.add(r.strip().lower())

        for ev in selected_events:
            c = ev.concept.strip().lower()
            if c in GENERIC_NON_TOPIC_CONCEPTS:
                continue
            if allowed_from_anchors and c not in allowed_from_anchors:
                continue
            if c not in event_concepts:
                event_concepts.append(c)

    q_tokens = set(tokenize_medical_text(question_text))
    combined_query = question_text.strip()
    if event_concepts and (q_anchors or any(w in q_tokens for w in {"medication", "medications", "taking", "prescribed", "active", "current", "history", "condition", "conditions", "problem", "problems", "lab", "labs", "test", "results", "allergy", "allergies", "treatment", "therapy", "dose", "changed", "trend"})):
        combined_query = f"{question_text.strip()} {' '.join(event_concepts[:4])}"

    query_tokens = tokenize_medical_text(combined_query)
    bm25_raw = (
        np.array(idx_data["bm25"].get_scores(query_tokens), dtype=float)
        if query_tokens
        else np.zeros(len(chunks), dtype=float)
    )
    q_only_bm25 = (
        np.array(idx_data["bm25"].get_scores(list(q_tokens)), dtype=float)
        if q_tokens
        else np.zeros(len(chunks), dtype=float)
    )

    q_emb = encode_texts_cached([combined_query, question_text.strip()])
    dense_sims = np.dot(idx_data["embeddings"], q_emb[0])
    q_only_dense = np.dot(idx_data["embeddings"], q_emb[1])

    if not q_anchors and float(np.max(q_only_dense)) < 0.30 and float(np.max(q_only_bm25)) < 2.5:
        return []

    bm25_order = np.argsort(-bm25_raw, kind="mergesort")
    dense_order = np.argsort(-dense_sims, kind="mergesort")
    bm25_ranks = {int(idx): r + 1 for r, idx in enumerate(bm25_order)}
    dense_ranks = {int(idx): r + 1 for r, idx in enumerate(dense_order)}

    k_rrf = 60
    max_rrf = 2.0 / (k_rrf + 1.0)
    active_concepts = q_anchors | set(event_concepts[:4])

    candidates: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        ch_concepts = {c.strip().lower() for c in ch.concepts}
        matched = sorted(ch_concepts & active_concepts)

        q_lower = question_text.lower()
        for c in ch_concepts:
            if c in q_lower and c not in matched:
                matched.append(c)

        rrf_val = (1.0 / (k_rrf + bm25_ranks[i])) + (1.0 / (k_rrf + dense_ranks[i]))
        rrf_norm = rrf_val / max_rrf
        d_sim = float(dense_sims[i])
        b_val = float(bm25_raw[i])

        has_concept_match = len(matched) > 0
        has_strong_text_match = (float(q_only_dense[i]) >= 0.36 and float(q_only_bm25[i]) >= 2.0)
        if not (has_concept_match or has_strong_text_match):
            continue

        concept_bonus = min(0.25, 0.15 * len(matched)) if has_concept_match else 0.0
        combined_score = 0.45 * rrf_norm + 0.35 * max(0.0, d_sim) + concept_bonus

        if combined_score >= min_relevance_threshold:
            candidates.append(
                {
                    "chunk": ch,
                    "score": round(float(combined_score), 4),
                    "dense_sim": round(d_sim, 4),
                    "bm25_score": round(b_val, 4),
                    "matched_concepts": matched,
                }
            )

    candidates.sort(
        key=lambda x: (-len(x["matched_concepts"]), -x["score"], x["chunk"].chunk_id)
    )
    selected = candidates[:top_k]
    for rank_idx, item in enumerate(selected, start=1):
        item["citation_tag"] = f"[C{rank_idx}]"
    return selected


def assert_no_corpus_citation_in_claims(answer: AnswerJSON) -> bool:
    """Verify that `AnswerJSON` never uses a `[C#]` or `C_###` corpus ID to support a patient-specific claim."""
    for cid in answer.cited_event_ids:
        if re.match(r"^(?:\[?C\d+\]?|C_\d+)$", cid.strip(), re.IGNORECASE):
            raise ValueError(
                f"Separation violation: Corpus chunk ID '{cid}' found in patient cited_event_ids."
            )
    if re.search(r"\[C\d+\]", answer.answer_text):
        raise ValueError(
            "Separation violation: Corpus citation tag [C#] found inside patient-specific answer_text."
        )
    return True
