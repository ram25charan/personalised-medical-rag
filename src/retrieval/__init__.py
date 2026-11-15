"""Package exports for `src.retrieval` (Section 12.2)."""

from src.retrieval.bm25 import bm25_scores
from src.retrieval.corpus_retrieval import (
    assert_no_corpus_citation_in_claims,
    load_demo_corpus,
    retrieve_corpus_passages,
)
from src.retrieval.dense import dense_scores
from src.retrieval.rrf import min_max_normalise, reciprocal_rank_fusion, rq_scores
from src.retrieval.validate_corpus import validate_demo_corpus

__all__ = [
    "assert_no_corpus_citation_in_claims",
    "bm25_scores",
    "dense_scores",
    "load_demo_corpus",
    "min_max_normalise",
    "reciprocal_rank_fusion",
    "retrieve_corpus_passages",
    "rq_scores",
    "validate_demo_corpus",
]
