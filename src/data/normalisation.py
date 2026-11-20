"""Versioned normalisation rules for medications and diagnoses (Sections 6.2, 6.5, 8.10).

Normalises medication strings using cleaned base strings and GSN / ETC therapeutic classes,
keeps original raw strings, and supports deterministic canonical concept resolution across
`medrecon` and `pyxis` records.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd

NORMALISATION_VERSION = "1.0"

_SALT_SUFFIXES = (
    " hcl",
    " hydrochloride",
    " maleate",
    " tartrate",
    " succinate",
    " sulfate",
    " sodium",
    " potassium",
    " calcium",
    " besylate",
    " acetate",
    " citrate",
    " fumarate",
    " mesylate",
    " nitrate",
    " mononitrate",
    " dinitrate",
    " bromide",
    " chloride",
)

_FORMULATION_SUFFIXES = (
    " er",
    " dr",
    " xr",
    " sr",
    " cr",
    " xl",
    " hfa",
    " ec",
    " odt",
    " oral",
    " liquid",
    " tablet",
    " capsule",
    " injection",
    " solution",
    " suspension",
)


def clean_medication_string(raw_name: str) -> str:
    """Clean a raw medication string into a lowercase base concept name.

    1. Strips leading asterisks (`*`) and surrounding whitespace.
    2. Removes bracketed brand suffixes (`[Digitek]`, `[Vitamin D3]`).
    3. If a parenthetical generic name is present (e.g. `cholecalciferol (vitamin D3)`),
       keeps the primary base name before the parentheses unless the primary token is numeric.
    4. Removes trailing dosage numbers/units (e.g. `Klor-Con 10`, `aspirin 81 mg`).
    5. Collapses whitespace and lowercases.
    """
    if not raw_name or pd.isna(raw_name):
        return "unknown_medication"

    s = str(raw_name).strip().lstrip("*").strip()
    s = re.sub(r"\[.*?\]", " ", s)
    s_no_paren = re.sub(r"\(.*?\)", " ", s).strip()
    if s_no_paren:
        s = s_no_paren

    s = s.lower()
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|units|meq|%)?\b\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" -/,")
    return s if s else str(raw_name).strip().lower()


def build_gsn_canonical_map(
    medrecon_df: pd.DataFrame,
    pyxis_df: Optional[pd.DataFrame] = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Build deterministic GSN -> canonical concept name and GSN -> therapeutic class maps.

    For each non-zero GSN, chooses the most frequent cleaned generic-style name
    (preferring names that originally started lowercase in `medrecon`, which indicates generic
    entries over Title-Case brand entries like `Vasotec` or `Keflex`; ties broken alphabetically).
    """
    gsn_names: dict[int, list[tuple[str, bool]]] = {}
    gsn_classes: dict[int, Counter[str]] = {}

    for _, row in medrecon_df.iterrows():
        gsn_val = row.get("gsn")
        if pd.isna(gsn_val):
            continue
        gsn_int = int(gsn_val)
        if gsn_int <= 0:
            continue

        raw_name = str(row.get("name", "")).strip()
        cleaned = clean_medication_string(raw_name)
        is_Originally_lower = raw_name[:1].islower() if raw_name else False
        gsn_names.setdefault(gsn_int, []).append((cleaned, is_Originally_lower))

        etc_desc = row.get("etcdescription")
        if pd.notna(etc_desc) and str(etc_desc).strip():
            gsn_classes.setdefault(gsn_int, Counter())[str(etc_desc).strip()] += 1

    if pyxis_df is not None:
        for _, row in pyxis_df.iterrows():
            gsn_val = row.get("gsn")
            if pd.isna(gsn_val):
                continue
            gsn_int = int(float(gsn_val))
            if gsn_int <= 0:
                continue
            raw_name = str(row.get("name", "")).strip()
            cleaned = clean_medication_string(raw_name)
            is_lower = raw_name[:1].islower() if raw_name else False
            gsn_names.setdefault(gsn_int, []).append((cleaned, is_lower))

    gsn_to_concept: dict[int, str] = {}
    for gsn_int, entries in gsn_names.items():
        lower_entries = [name for name, is_low in entries if is_low]
        pool = lower_entries if lower_entries else [name for name, _ in entries]
        counts = Counter(pool)
        best_name = sorted(counts.keys(), key=lambda n: (-counts[n], len(n), n))[0]
        gsn_to_concept[gsn_int] = best_name

    gsn_to_class: dict[int, str] = {}
    for gsn_int, class_counter in gsn_classes.items():
        best_class = sorted(
            class_counter.keys(), key=lambda c: (-class_counter[c], c)
        )[0]
        gsn_to_class[gsn_int] = best_class

    return gsn_to_concept, gsn_to_class


def normalise_medication(
    raw_name: str,
    gsn: Optional[int | float],
    etcdescription: Optional[str],
    gsn_to_concept: dict[int, str],
    gsn_to_class: dict[int, str],
) -> tuple[str, Optional[str]]:
    """Return `(canonical_concept, concept_class)` for a medication record."""
    gsn_int = 0
    if gsn is not None and pd.notna(gsn):
        try:
            gsn_int = int(float(gsn))
        except (ValueError, TypeError):
            gsn_int = 0

    if gsn_int > 0 and gsn_int in gsn_to_concept:
        concept = gsn_to_concept[gsn_int]
    else:
        concept = clean_medication_string(raw_name)

    concept_class: Optional[str] = None
    if etcdescription is not None and pd.notna(etcdescription) and str(etcdescription).strip():
        concept_class = str(etcdescription).strip()
    elif gsn_int > 0 and gsn_int in gsn_to_class:
        concept_class = gsn_to_class[gsn_int]

    return concept, concept_class


def normalise_diagnosis(icd_title: str, icd_code: str, icd_version: int) -> tuple[str, str]:
    """Normalise an ICD diagnosis title into a clean lowercase concept and ICD chapter/version class."""
    if not icd_title or pd.isna(icd_title):
        concept = f"icd{icd_version}_{str(icd_code).strip().lower()}"
    else:
        s = str(icd_title).strip().lower()
        s = re.sub(r"\s+", " ", s)
        concept = s
    concept_class = f"ICD-{icd_version}"
    return concept, concept_class
