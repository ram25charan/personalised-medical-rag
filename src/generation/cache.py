"""Persistent Disk Cache for LLM Generation Responses (Sections 4.5, 7.7, 8.8).

Caches raw and parsed LLM outputs keyed by `(model_id, prompt_hash)` under `results/cache/`
so interrupted runs resume seamlessly without re-spending quota.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip())


def get_cache_file_path(
    model_id: str,
    prompt_hash: str,
    cache_dir: str | Path = "results/cache",
) -> Path:
    model_folder = Path(cache_dir) / _sanitize_model_id(model_id)
    model_folder.mkdir(parents=True, exist_ok=True)
    return model_folder / f"{prompt_hash}.json"


def load_cached_response(
    model_id: str,
    prompt_hash: str,
    cache_dir: str | Path = "results/cache",
) -> Optional[dict[str, Any]]:
    """Return cached entry dict if present on disk, else None."""
    path = get_cache_file_path(model_id, prompt_hash, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_response(
    model_id: str,
    prompt_hash: str,
    raw_output: str,
    parsed_answer: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
    cache_dir: str | Path = "results/cache",
) -> Path:
    """Persist an LLM response to the disk cache."""
    path = get_cache_file_path(model_id, prompt_hash, cache_dir=cache_dir)
    payload = {
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "raw_output": raw_output,
        "parsed_answer": parsed_answer,
        "metadata": metadata or {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
