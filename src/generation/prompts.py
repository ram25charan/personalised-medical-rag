"""Prompt Construction for Evidence-Grounded Generation (Sections 7.7, 12.1, 13.2).

Prompt version: 1.0 (frozen before test evaluation).
Uses the identical prompt template across all systems (B1, B1+, B2-, B2, B4, ORACLE-G),
changing only the assembled context passed to the template.
"""

from __future__ import annotations

import hashlib
from typing import Optional

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """You are a clinical record research assistant evaluating a longitudinal patient history.
Research prototype. Not medical advice.

STRICT GROUNDING AND CITATION RULES:
1. Base all patient-specific claims ONLY on the provided patient events `[E#]`.
2. In `cited_event_ids`, list ONLY the minimal decisive `event_id` string(s) (e.g. `"S_001_0012"` or `"R10000032_0005"`, never `"E1"`) that directly answer the question:
   - For `current_state` / discontinuation / active therapy questions: cite ONLY the single latest decisive record for the active or discontinued therapy (do NOT list prior `SUPERSEDED` refills).
   - For `first_occurrence` questions: cite ONLY the earliest diagnosis/start record for that concept.
   - For `change_over_time` / `ordered_values` questions: cite ONLY the earliest and latest boundary records showing the change or sequence.
3. Interpret temporal status strictly relative to `as_of_date`:
   - Interval-basis records: `ACTIVE`, `ENDED`, `SUPERSEDED`, `PERSISTENT`, or `UNKNOWN`.
   - Track R snapshot reconciliation lists (`medrecon`): use `LISTED` (if present on the latest visit list at/before `as_of_date`) or `NOT_LISTED_LATEST` (if present on an earlier list but absent from the latest list).
4. If the records cannot answer the question or if `[INSUFFICIENT EVIDENCE WARNING (U5)]` is present, set `"abstain": true`, `"status_claims": []`, `"cited_event_ids": []`, and `"uncertainty_noted": true`.
5. If records disagree or if `[DETECTED CONFLICTS]` is present, set `"conflict_flagged": true`, `"uncertainty_noted": true`, include both conflicting statuses (e.g. `ACTIVE` and `ENDED`) in `status_claims`, and cite both conflicting `event_id`s.
6. Set `"uncertainty_noted": true` ONLY when `[UNCERTAINTY FLAGS]`, `[DETECTED CONFLICTS]`, or `[INSUFFICIENT EVIDENCE WARNING (U5)]` is present in the context, or when the decisive cited record has a missing date (`UNKNOWN_DATE`), month/year-only date, open medication end date (`OPEN`), or snapshot reconciliation list (`medrecon`). Otherwise set `"uncertainty_noted": false`.
7. Never invent medications, doses, dates, or diagnoses. Never provide clinical advice.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this exact schema:
{
  "answer_text": "Concise grounded answer citing decisive event_id(s) and stating exact values/dates/statuses.",
  "status_claims": [
    {"concept": "lowercase concept name", "status": "ACTIVE | ENDED | SUPERSEDED | PERSISTENT | UNKNOWN | LISTED | NOT_LISTED_LATEST"}
  ],
  "cited_event_ids": ["decisive_event_id"],
  "abstain": false,
  "conflict_flagged": false,
  "uncertainty_noted": false
}"""


def build_user_prompt(
    patient_id: str,
    question_text: str,
    as_of_date: str,
    context_text: str,
    window_start: Optional[str] = None,
    background_text: Optional[str] = None,
) -> str:
    """Build the user message for the LLM generator."""
    win_line = f"Reference Window Start: {window_start}\n" if window_start else ""
    bg_block = (
        f"\n=== GENERAL BACKGROUND INFORMATION (NON-PERSONAL, DEMO ONLY) ===\n{background_text}\n"
        if background_text
        else ""
    )
    return (
        f"Patient ID: {patient_id}\n"
        f"Reference Date (as_of_date): {as_of_date}\n"
        f"{win_line}"
        f"Question: {question_text}\n\n"
        f"=== PROVIDED PATIENT EVENTS (ORDERED CHRONOLOGICALLY UP TO {as_of_date}) ===\n"
        f"{context_text if context_text.strip() else '(No candidate events available)'}\n"
        f"{bg_block}\n"
        f"Respond with the required JSON object only."
    )


def compute_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Compute a deterministic SHA-256 hash of `(system_prompt, user_prompt)` for response caching."""
    payload = f"v={PROMPT_VERSION}\nSYS:\n{system_prompt}\nUSER:\n{user_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
