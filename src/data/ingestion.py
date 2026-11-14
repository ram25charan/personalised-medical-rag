"""Ingestion and data-quality assessment for MIMIC-IV-ED Demo v2.2 (Sections 6.2, 6.5, 12.2).

Loads the six raw tables read-only, verifies expected row and patient counts,
performs relational and timestamp quality checks, and flags outliers without deleting rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_TABLE_ROWS: dict[str, int] = {
    "edstays": 222,
    "diagnosis": 545,
    "medrecon": 2764,
    "pyxis": 1082,
    "triage": 222,
    "vitalsign": 1038,
}

EXPECTED_UNIQUE_PATIENTS = 64


def resolve_table_path(base_path: str | Path, table_name: str) -> Path:
    """Locate `<table_name>.csv.gz` or `<table_name>.csv` under `base_path` or `base_path/ed`."""
    root = Path(base_path)
    candidates = [
        root / "ed" / f"{table_name}.csv.gz",
        root / f"{table_name}.csv.gz",
        root / "ed" / f"{table_name}.csv",
        root / f"{table_name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find raw table '{table_name}' under {root.resolve()}"
    )


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a raw file for provenance and run manifests."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path = "data/raw/mimic_iv_ed_demo") -> dict[str, pd.DataFrame]:
    """Load the six MIMIC-IV-ED demo tables read-only without mutating source files.

    Returns a dictionary mapping table name to its raw pandas DataFrame, preserving original row indices.
    """
    raw_tables: dict[str, pd.DataFrame] = {}
    for table_name in EXPECTED_TABLE_ROWS:
        table_path = resolve_table_path(path, table_name)
        df = pd.read_csv(table_path, low_memory=False)
        raw_tables[table_name] = df
    return raw_tables


def validate_raw_counts(raw: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate that loaded tables match the exact row and patient counts in Section 6.2."""
    row_counts = {name: len(df) for name, df in raw.items()}
    mismatches = {
        name: {"actual": row_counts.get(name), "expected": expected}
        for name, expected in EXPECTED_TABLE_ROWS.items()
        if row_counts.get(name) != expected
    }
    unique_patients = int(raw["edstays"]["subject_id"].nunique())
    if unique_patients != EXPECTED_UNIQUE_PATIENTS:
        mismatches["unique_patients"] = {
            "actual": unique_patients,
            "expected": EXPECTED_UNIQUE_PATIENTS,
        }

    return {
        "valid": len(mismatches) == 0,
        "row_counts": row_counts,
        "unique_patients": unique_patients,
        "mismatches": mismatches,
    }


def assess_data_quality(
    raw: dict[str, pd.DataFrame],
    raw_dir: str | Path = "data/raw/mimic_iv_ed_demo",
) -> dict[str, Any]:
    """Perform a comprehensive quality, longitudinal, and outlier audit (Section 6.5).

    Flags physiological outliers and missing values without deleting any rows.
    """
    validation = validate_raw_counts(raw)
    edstays = raw["edstays"]
    diagnosis = raw["diagnosis"]
    medrecon = raw["medrecon"]
    pyxis = raw["pyxis"]
    triage = raw["triage"]
    vitalsign = raw["vitalsign"]

    # File hashes
    file_hashes = {}
    for name in EXPECTED_TABLE_ROWS:
        try:
            file_hashes[name] = compute_file_sha256(resolve_table_path(raw_dir, name))
        except FileNotFoundError:
            file_hashes[name] = None

    # Missing value counts per table
    missing_values: dict[str, dict[str, int]] = {}
    for name, df in raw.items():
        na_counts = df.isna().sum().to_dict()
        missing_values[name] = {k: int(v) for k, v in na_counts.items() if v > 0}

    # Exact duplicate row counts
    duplicate_rows = {name: int(df.duplicated().sum()) for name, df in raw.items()}

    # Relational integrity check (all stay_ids in child tables exist in edstays)
    valid_stays = set(edstays["stay_id"].unique())
    valid_subjects = set(edstays["subject_id"].unique())
    orphan_stays: dict[str, int] = {}
    orphan_subjects: dict[str, int] = {}
    for name, df in raw.items():
        if "stay_id" in df.columns:
            orphan_stays[name] = int((~df["stay_id"].isin(valid_stays)).sum())
        if "subject_id" in df.columns:
            orphan_subjects[name] = int((~df["subject_id"].isin(valid_subjects)).sum())

    # Timestamp checks on edstays
    intime_dt = pd.to_datetime(edstays["intime"], errors="coerce")
    outtime_dt = pd.to_datetime(edstays["outtime"], errors="coerce")
    invalid_intime = int(intime_dt.isna().sum())
    invalid_outtime = int(outtime_dt.isna().sum())
    inverted_stay_times = int((outtime_dt < intime_dt).sum())

    # Longitudinal structure (Section 6.2)
    stays_per_patient = edstays.groupby("subject_id")["stay_id"].nunique()
    patients_2plus_stays = int((stays_per_patient >= 2).sum())
    patients_3plus_stays = int((stays_per_patient >= 3).sum())
    max_stays_per_patient = int(stays_per_patient.max())

    edstays_time = edstays.assign(intime_dt=intime_dt, outtime_dt=outtime_dt)
    multi_visit_ids = stays_per_patient[stays_per_patient >= 2].index
    spans_days: list[float] = []
    for pid in multi_visit_ids:
        p_rows = edstays_time[edstays_time["subject_id"] == pid]
        span = (p_rows["outtime_dt"].max() - p_rows["intime_dt"].min()).total_seconds() / 86400.0
        spans_days.append(span)
    median_span_days = float(pd.Series(spans_days).median()) if spans_days else 0.0

    stays_with_medrecon = int(medrecon["stay_id"].nunique())
    medrecon_stays_per_patient = medrecon.groupby("subject_id")["stay_id"].nunique()
    patients_2plus_medrecon_visits = int((medrecon_stays_per_patient >= 2).sum())

    # Multi-visit stays missing a medication list
    multi_visit_stays = edstays[edstays["subject_id"].isin(multi_visit_ids)]
    multi_visit_stays_without_medrecon = int(
        (~multi_visit_stays["stay_id"].isin(set(medrecon["stay_id"].unique()))).sum()
    )

    # Patients with both ICD-9 and ICD-10 diagnosis codes
    icd_versions_per_patient = diagnosis.groupby("subject_id")["icd_version"].nunique()
    patients_both_icd9_and_icd10 = int((icd_versions_per_patient > 1).sum())

    # Physiological outlier flags (vitalsign & triage) - flagged, never deleted
    def count_vital_outliers(df: pd.DataFrame) -> dict[str, int]:
        flags: dict[str, int] = {}
        if "temperature" in df.columns:
            temp = pd.to_numeric(df["temperature"], errors="coerce")
            flags["temperature_outside_85_115F"] = int(((temp < 85.0) | (temp > 115.0)).sum())
        if "heartrate" in df.columns:
            hr = pd.to_numeric(df["heartrate"], errors="coerce")
            flags["heartrate_outside_20_250"] = int(((hr < 20.0) | (hr > 250.0)).sum())
        if "resprate" in df.columns:
            rr = pd.to_numeric(df["resprate"], errors="coerce")
            flags["resprate_outside_4_60"] = int(((rr < 4.0) | (rr > 60.0)).sum())
        if "o2sat" in df.columns:
            o2 = pd.to_numeric(df["o2sat"], errors="coerce")
            flags["o2sat_outside_50_100"] = int(((o2 < 50.0) | (o2 > 100.0)).sum())
        if "sbp" in df.columns:
            sbp = pd.to_numeric(df["sbp"], errors="coerce")
            flags["sbp_outside_40_300"] = int(((sbp < 40.0) | (sbp > 300.0)).sum())
        if "dbp" in df.columns:
            dbp = pd.to_numeric(df["dbp"], errors="coerce")
            flags["dbp_outside_20_200"] = int(((dbp < 20.0) | (dbp > 200.0)).sum())
        return flags

    outlier_flags = {
        "triage": count_vital_outliers(triage),
        "vitalsign": count_vital_outliers(vitalsign),
        "medrecon_gsn_zero": int((medrecon["gsn"] == 0).sum()),
        "pyxis_missing_gsn": int(pyxis["gsn"].isna().sum()),
    }

    return {
        "schema_version": "1.0",
        "validation": validation,
        "file_sha256": file_hashes,
        "missing_values": missing_values,
        "exact_duplicate_rows": duplicate_rows,
        "relational_integrity": {
            "orphan_stays": orphan_stays,
            "orphan_subjects": orphan_subjects,
        },
        "timestamp_checks": {
            "invalid_intime": invalid_intime,
            "invalid_outtime": invalid_outtime,
            "inverted_stay_times": inverted_stay_times,
            "min_year": int(intime_dt.dt.year.min()),
            "max_year": int(outtime_dt.dt.year.max()),
        },
        "longitudinal_summary": {
            "total_patients": validation["unique_patients"],
            "total_stays": len(edstays),
            "patients_with_2plus_stays": patients_2plus_stays,
            "patients_with_3plus_stays": patients_3plus_stays,
            "max_stays_per_patient": max_stays_per_patient,
            "median_multi_visit_span_days": round(median_span_days, 2),
            "stays_with_medrecon": stays_with_medrecon,
            "patients_with_2plus_medrecon_visits": patients_2plus_medrecon_visits,
            "multi_visit_stays_without_medrecon": multi_visit_stays_without_medrecon,
            "patients_with_both_icd9_and_icd10": patients_both_icd9_and_icd10,
        },
        "outlier_flags": outlier_flags,
    }
