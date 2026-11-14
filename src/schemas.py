"""Shared data contracts and schema validators (Section 12.1 of Master Specification).

Schema Version: 1.0
All modules exchange only these contracts. Retrieval functions never receive Gold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"

TrackType = Literal["R", "S", "P"]
EventType = Literal[
    "reported_home_medication",
    "ed_administered_medication",
    "medication_order",
    "diagnosis_at_visit",
    "condition",
    "allergy",
    "lab_result",
    "symptom",
    "vital_sign",
    "triage_observation",
    "visit",
]
DatePrecision = Literal["day", "month", "year", "unknown"]
StatusBasis = Literal["interval", "snapshot"]
SourceType = Literal["real", "synthetic", "perturbed"]
SplitType = Literal["dev", "test"]
IntentType = Literal[
    "current_state",
    "historical",
    "change_over_time",
    "first_occurrence",
    "persistent_fact",
]
CategoryType = Literal[
    "temporal_ordering",
    "current_vs_historical",
    "longitudinal_change",
    "irrelevant_history",
    "missing_or_ambiguous_dates",
    "conflicting_information",
    "insufficient_evidence",
]
ExpectedAnswerType = Literal["status", "set", "value", "ordered_values", "none"]
ExpectedBehaviorType = Literal["answer", "abstain", "flag_conflict", "flag_uncertainty"]
TemporalStateType = Literal[
    "ACTIVE",
    "ENDED",
    "SUPERSEDED",
    "PERSISTENT",
    "UNKNOWN",
    "LISTED",
    "NOT_LISTED_LATEST",
]
ChunkType = Literal["condition_overview", "test_explainer", "drug_class_note"]


class Event(BaseModel):
    """Unified Patient Event Representation (Sections 6.6 and 12.1)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    patient_id: str
    track: TrackType
    event_type: EventType
    concept: str
    concept_class: Optional[str] = None
    text: str
    observed_time: Optional[str] = None
    valid_start: Optional[str] = None
    valid_end: Optional[str] = None
    date_precision: DatePrecision
    time_is_shifted: bool
    status_basis: StatusBasis
    source_type: SourceType
    source_dataset: str
    source_table: Optional[str] = None
    source_rows: list[int] = Field(default_factory=list)
    encounter_id: Optional[str] = None
    confidence: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def check_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class Question(BaseModel):
    """Benchmark Question Contract (Section 12.1)."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    patient_id: str
    track: TrackType
    split: SplitType
    text: str
    as_of_date: str
    window_start: Optional[str] = None


class ExpectedAnswer(BaseModel):
    """Structured expected answer inside Gold."""

    model_config = ConfigDict(extra="forbid")

    type: ExpectedAnswerType
    value: Any = None


class Gold(BaseModel):
    """Programmatic Gold Contract (Section 12.1). Never read by retrieval or generation."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    intent: IntentType
    category: CategoryType
    anchor_concepts: list[str]
    gold_events: list[str]
    distractor_events: list[str] = Field(default_factory=list)
    expected_answer: ExpectedAnswer
    expected_behavior: ExpectedBehaviorType
    expected_conflict: bool = False
    expected_flags: list[str] = Field(default_factory=list)


class CorpusChunk(BaseModel):
    """Clinical Evidence Corpus Passage for demo only (Sections 6.4 and 12.1)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    title: str
    text: str
    chunk_type: ChunkType
    source_name: str
    url: str
    accessed: str
    license_note: str
    concepts: list[str]


class CandidateScore(BaseModel):
    """Per-candidate scoring breakdown inside PipelineOutput."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    state: TemporalStateType
    rq: float
    rh: float
    rt: float
    score: float
    selected: bool


class ConflictFlag(BaseModel):
    """Conflict detection flag (C1-C3)."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["C1", "C2", "C3"]
    event_ids: list[str]
    description: Optional[str] = None


class UncertaintyFlag(BaseModel):
    """Uncertainty flag (U1-U4)."""

    model_config = ConfigDict(extra="forbid")

    flag: Literal["U1", "U2", "U3", "U4"]
    event_ids: list[str]
    description: Optional[str] = None


class ProvenanceNote(BaseModel):
    """Informational provenance note (e.g. Track R medrecon vs pyxis mismatch)."""

    model_config = ConfigDict(extra="forbid")

    note_type: str
    event_ids: list[str]
    description: str


class FlagsContainer(BaseModel):
    """Combined conflict and uncertainty flags in PipelineOutput."""

    model_config = ConfigDict(extra="forbid")

    conflicts: list[ConflictFlag] = Field(default_factory=list)
    uncertainty: list[UncertaintyFlag] = Field(default_factory=list)
    insufficient_evidence: bool = False
    provenance_notes: list[ProvenanceNote] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    """Output of retrieval, scoring, fusion, and flag detection stages (Section 12.1)."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    as_of_date: str
    system_id: str
    intent: IntentType
    intent_source: Literal["rules", "oracle"]
    candidates: list[CandidateScore]
    selected_event_ids: list[str]
    flags: FlagsContainer
    context_text: str
    background_chunk_ids: list[str] = Field(default_factory=list)


class StatusClaim(BaseModel):
    """Structured status claim made by the generator."""

    model_config = ConfigDict(extra="forbid")

    concept: str
    status: str


class AnswerJSON(BaseModel):
    """Structured LLM Output Contract (Section 12.1)."""

    model_config = ConfigDict(extra="forbid")

    answer_text: str
    status_claims: list[StatusClaim] = Field(default_factory=list)
    cited_event_ids: list[str] = Field(default_factory=list)
    abstain: bool = False
    conflict_flagged: bool = False
    uncertainty_noted: bool = False


class ResultRecord(BaseModel):
    """Experiment Run Result Record (Section 12.1)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    git_commit: str
    config_hash: str
    system_id: str
    track: TrackType
    question_id: str
    model_id: Optional[str] = None
    run_date: str
    temperature: float = 0.0
    k: int = 10
    weights: dict[str, float] = Field(default_factory=dict)
    intent_source: Literal["rules", "oracle"] = "rules"
    prompt_hash: Optional[str] = None
    raw_output: Optional[str] = None
    parsed_answer: Optional[AnswerJSON] = None
    metrics: dict[str, Any] = Field(default_factory=dict)


def save_jsonl(items: list[BaseModel], path: str | Path) -> None:
    """Serialize a list of Pydantic models to UTF-8 JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_jsonl(model_cls: type[BaseModel], path: str | Path) -> list[Any]:
    """Load and validate a UTF-8 JSONL file against a Pydantic model class."""
    in_path = Path(path)
    results = []
    with in_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            results.append(model_cls.model_validate(data))
    return results
