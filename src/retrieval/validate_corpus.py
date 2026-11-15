"""Demo Clinical Corpus Validator (`src/retrieval/validate_corpus.py`) (Phase 5 Plan Part 1).

Validates `data/corpus/demo_corpus.jsonl` against:
  1. `CorpusChunk` Pydantic schema (`src/schemas.py`)
  2. Concept key alignment: every key in `chunk.concepts` must exist in `configs/relation_table.json`
     or in `data/processed/events.jsonl` / `data/synthetic/events.jsonl`
  3. Provenance & licensing: non-empty `source_name`, valid `https://` `url`, `accessed` date, and `license_note`;
     no MedlinePlus/ASHP drug monographs (`medlineplus.gov/druginfo` forbidden)
  4. Advice & dosing filter: flags words such as `mg`, `mcg`, `dose`, `take`, `twice daily`, `you should`
  5. Uniqueness & length bounds: no duplicate `chunk_id` or `text`, word count between 60 and 220 words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.schemas import CorpusChunk, load_jsonl

FORBIDDEN_ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mg", re.compile(r"\b\d*\s*mg\b", re.IGNORECASE)),
    ("mcg", re.compile(r"\b\d*\s*mcg\b", re.IGNORECASE)),
    ("dose", re.compile(r"\bdoses?\b|\bdosing\b", re.IGNORECASE)),
    ("take", re.compile(r"\btake\b", re.IGNORECASE)),
    ("twice daily", re.compile(r"\btwice\s+daily\b|\bonce\s+daily\b", re.IGNORECASE)),
    ("you should", re.compile(r"\byou\s+should\b|\byour\s+doctor\b", re.IGNORECASE)),
]


def load_known_pipeline_concepts(
    relation_table_path: str | Path = "configs/relation_table.json",
    real_events_path: str | Path = "data/processed/events.jsonl",
    synth_events_path: str | Path = "data/synthetic/events.jsonl",
) -> set[str]:
    """Return the union of normalised concept keys across relation_table.json, Track R, and Track S."""
    known: set[str] = set()
    rt_path = Path(relation_table_path)
    if rt_path.exists():
        data = json.loads(rt_path.read_text(encoding="utf-8"))
        for k, v in data.get("concepts", {}).items():
            known.add(k.strip().lower())
            for rel in v.get("related_concepts", []):
                known.add(rel.strip().lower())

    for ev_path_str in (real_events_path, synth_events_path):
        ev_path = Path(ev_path_str)
        if ev_path.exists():
            with ev_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        c = str(row.get("concept", "")).strip().lower()
                        if c:
                            known.add(c)
    return known


def validate_demo_corpus(
    corpus_path: str | Path = "data/corpus/demo_corpus.jsonl",
    relation_table_path: str | Path = "configs/relation_table.json",
) -> dict[str, Any]:
    """Validate `demo_corpus.jsonl` and return a structured audit report. Raises ValueError on violations."""
    chunks: list[CorpusChunk] = load_jsonl(CorpusChunk, corpus_path)
    if not chunks:
        raise ValueError(f"Corpus at {corpus_path} is empty.")

    known_concepts = load_known_pipeline_concepts(relation_table_path=relation_table_path)

    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    advice_flags: list[dict[str, str]] = []
    unknown_concepts: list[tuple[str, str]] = []
    provenance_errors: list[str] = []
    word_counts: list[int] = []

    for ch in chunks:
        if ch.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {ch.chunk_id}")
        seen_ids.add(ch.chunk_id)

        norm_text = re.sub(r"\s+", " ", ch.text.strip().lower())
        if norm_text in seen_texts:
            raise ValueError(f"Duplicate passage text in {ch.chunk_id}")
        seen_texts.add(norm_text)

        words = ch.text.split()
        word_counts.append(len(words))
        if not (100 <= len(words) <= 200):
            provenance_errors.append(
                f"{ch.chunk_id}: word count {len(words)} outside [100, 200]"
            )

        if not ch.url.startswith("https://"):
            provenance_errors.append(f"{ch.chunk_id}: invalid URL {ch.url}")
        if "druginfo" in ch.url.lower():
            provenance_errors.append(
                f"{ch.chunk_id}: MedlinePlus/ASHP drug monograph URLs are forbidden ({ch.url})"
            )
        if not ch.source_name.strip() or not ch.license_note.strip() or not ch.accessed.strip():
            provenance_errors.append(f"{ch.chunk_id}: missing source/license/accessed metadata")

        if not ch.concepts:
            provenance_errors.append(f"{ch.chunk_id}: empty concepts list")
        for c in ch.concepts:
            if c.strip().lower() not in known_concepts:
                unknown_concepts.append((ch.chunk_id, c))

        for label, pat in FORBIDDEN_ADVICE_PATTERNS:
            m = pat.search(ch.text)
            if m:
                advice_flags.append(
                    {"chunk_id": ch.chunk_id, "rule": label, "match": m.group(0)}
                )

    if provenance_errors:
        raise ValueError(f"Provenance errors: {provenance_errors}")
    if unknown_concepts:
        raise ValueError(f"Unknown concept keys not in pipeline/relation table: {unknown_concepts}")
    if advice_flags:
        raise ValueError(f"Forbidden advice/dosing terms found in corpus: {advice_flags}")

    return {
        "num_chunks": len(chunks),
        "by_type": {
            t: sum(1 for c in chunks if c.chunk_type == t)
            for t in ("condition_overview", "test_explainer", "drug_class_note")
        },
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "advice_flags": advice_flags,
        "unknown_concepts": unknown_concepts,
        "valid": True,
    }


if __name__ == "__main__":
    rep = validate_demo_corpus()
    print(json.dumps(rep, indent=2))
