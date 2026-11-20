"""Multi-Key, Rate-Limit-Tolerant LLM Client for Groq and Gemini Flash (Sections 7.7, 10.5, 12.2, TD1).

Features:
  - Supports 4 API keys from `.env` (`GROQ_API_KEY_1`, `GROQ_API_KEY_2`, `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`)
    plus legacy fallback `GROQ_API_KEY` / `GEMINI_API_KEY`.
  - Automatic intra-provider key rotation (`GROQ_API_KEY_1 <-> GROQ_API_KEY_2`, `GEMINI_API_KEY_1 <-> GEMINI_API_KEY_2`)
    whenever an HTTP 429 rate-limit or transient API error occurs.
  - Strictly pins experimental runs to a single `model_id` (never mixing Groq and Gemini inside one experiment table).
  - Optional cross-provider failover (`allow_cross_provider_fallback=True`) for the interactive Streamlit demo only.
  - Persistent disk response cache keyed by `(model_id, prompt_hash)`.
  - Robust JSON extraction and validation into `AnswerJSON`.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import dotenv_values

from src.generation.cache import load_cached_response, save_cached_response
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, compute_prompt_hash
from src.schemas import AnswerJSON, StatusClaim

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"


def load_api_keys(env_path: str | Path = ".env") -> dict[str, list[tuple[str, str]]]:
    """Load Groq and Gemini API keys from `.env` and environment variables.

    Returns `{"groq": [("groq-1", key1), ("groq-2", key2)], "gemini": [("gemini-1", key1), ("gemini-2", key2)]}`.
    """
    file_vars = dotenv_values(env_path) if Path(env_path).exists() else {}

    def _get(name: str) -> str:
        val = os.environ.get(name) or file_vars.get(name) or ""
        return str(val).strip()

    groq_keys: list[tuple[str, str]] = []
    for kname in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY"]:
        val = _get(kname)
        if val and not val.startswith("your_") and val not in [k for _, k in groq_keys]:
            label = f"groq-{len(groq_keys) + 1}"
            groq_keys.append((label, val))

    gemini_keys: list[tuple[str, str]] = []
    for kname in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY"]:
        val = _get(kname)
        if val and not val.startswith("your_") and val not in [k for _, k in gemini_keys]:
            label = f"gemini-{len(gemini_keys) + 1}"
            gemini_keys.append((label, val))

    return {"groq": groq_keys, "gemini": gemini_keys}


def parse_and_validate_answer_json(
    raw_text: str,
    context_prompt: Optional[str] = None,
) -> AnswerJSON:
    """Parse an LLM response string into a validated `AnswerJSON` contract object (Section 7.7 & 10.6).

    Handles markdown code fences (` ```json ... ``` `), leading/trailing prose, maps `[E#]`
    shorthand citations to their `event_id` when `context_prompt` is provided, and normalises
    status claims and booleans. Raises `ValueError` if no valid JSON object can be recovered.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM output cannot be parsed into AnswerJSON.")

    e_map: dict[str, str] = {}
    id_to_line: dict[str, str] = {}
    if context_prompt:
        for m in re.finditer(r"\[(E\d+)\]\s+id=([^\s|]+)\s*\|([^\n]*)", context_prompt):
            e_key = m.group(1).upper()
            ev_id = m.group(2).strip()
            e_map[e_key] = ev_id
            id_to_line[ev_id] = m.group(3).lower()

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode JSON from LLM output: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Parsed LLM JSON root must be an object/dict.")

    ans_text = str(data.get("answer_text", "")).strip()
    if not ans_text:
        ans_text = "No answer text provided."

    raw_claims = data.get("status_claims") or []
    claims: list[StatusClaim] = []
    if isinstance(raw_claims, list):
        for item in raw_claims:
            if isinstance(item, dict) and item.get("concept") and item.get("status"):
                claims.append(
                    StatusClaim(
                        concept=str(item["concept"]).strip().lower(),
                        status=str(item["status"]).strip().upper(),
                    )
                )

    raw_cited = data.get("cited_event_ids") or []
    cited_ids: list[str] = []
    if isinstance(raw_cited, list):
        for cid in raw_cited:
            if cid is not None and str(cid).strip():
                clean_cid = str(cid).strip().strip("[]")
                if clean_cid.upper().startswith("ID="):
                    clean_cid = clean_cid[3:].strip()
                if clean_cid.upper() in e_map:
                    clean_cid = e_map[clean_cid.upper()]
                if clean_cid and clean_cid not in cited_ids:
                    cited_ids.append(clean_cid)

    if e_map and ans_text and not bool(data.get("abstain", False)):
        explicit_e_refs = [
            f"E{num}" for num in re.findall(r"\bE(\d{1,2})\b", ans_text) if f"E{num}" in e_map
        ]
        if explicit_e_refs:
            mapped_from_text = [e_map[ek] for ek in explicit_e_refs]
            claimed_concepts = [c.concept for c in claims]
            if claimed_concepts and cited_ids:
                valid_existing = [
                    cid
                    for cid in cited_ids
                    if any(cc in id_to_line.get(cid, "") for cc in claimed_concepts)
                ]
                if not valid_existing:
                    cited_ids = list(dict.fromkeys(mapped_from_text))

    return AnswerJSON(
        answer_text=ans_text,
        status_claims=claims,
        cited_event_ids=cited_ids,
        abstain=bool(data.get("abstain", False)),
        conflict_flagged=bool(data.get("conflict_flagged", False)),
        uncertainty_noted=bool(data.get("uncertainty_noted", False)),
    )


def _call_groq_http(
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout: int = 45,
) -> tuple[str, dict[str, Any]]:
    """Call Groq OpenAI-compatible Chat Completions endpoint."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in model_id:
        payload["reasoning_effort"] = "low"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return content, {"provider": "groq", "usage": usage}


def _call_gemini_http(
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout: int = 45,
) -> tuple[str, dict[str, Any]]:
    """Call Google Gemini `generateContent` REST endpoint with JSON response mime type."""
    clean_model = model_id.replace("models/", "").strip()
    if clean_model in {"gemini-2.5-flash", "gemini-2.5-flash-lite"}:
        clean_model = DEFAULT_GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    content = body["candidates"][0]["content"]["parts"][0]["text"]
    usage = body.get("usageMetadata", {})
    return content, {"provider": "gemini", "usage": usage}


class MultiKeyLLMClient:
    """Rate-limit-tolerant LLM client with automatic key rotation and response caching."""

    def __init__(
        self,
        provider: str = "groq",
        model_id: Optional[str] = None,
        temperature: float = 0.0,
        cache_dir: str | Path = "results/cache",
        env_path: str | Path = ".env",
        preferred_key_label: Optional[str] = None,
        allow_cross_provider_fallback: bool = False,
        min_request_interval_sec: float = 0.6,
    ) -> None:
        self.provider = provider.lower().strip()
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = (
                DEFAULT_GROQ_MODEL if self.provider == "groq" else DEFAULT_GEMINI_MODEL
            )
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.keys_by_provider = load_api_keys(env_path)
        self.preferred_key_label = preferred_key_label
        self.allow_cross_provider_fallback = allow_cross_provider_fallback
        self.min_request_interval_sec = min_request_interval_sec
        self._key_cursor: dict[str, int] = {"groq": 0, "gemini": 0}
        self._last_call_ts: float = 0.0

    def _get_ordered_keys(self, provider: str) -> list[tuple[str, str]]:
        pool = list(self.keys_by_provider.get(provider, []))
        if not pool:
            return []
        if self.preferred_key_label:
            preferred = [p for p in pool if p[0] == self.preferred_key_label]
            others = [p for p in pool if p[0] != self.preferred_key_label]
            if preferred:
                return preferred + others
        idx = self._key_cursor.get(provider, 0) % len(pool)
        return pool[idx:] + pool[:idx]

    def generate_from_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
        use_cache: bool = True,
        max_retries: int = 12,
    ) -> tuple[AnswerJSON, str, str, dict[str, Any]]:
        """Return `(parsed_answer, raw_output, prompt_hash, metadata)`."""
        prompt_hash = compute_prompt_hash(system_prompt, user_prompt)

        if use_cache:
            cached = load_cached_response(self.model_id, prompt_hash, cache_dir=self.cache_dir)
            if cached is not None:
                parsed = parse_and_validate_answer_json(
                    cached["raw_output"], context_prompt=user_prompt
                )
                meta = dict(cached.get("metadata", {}))
                meta["cached"] = True
                return parsed, cached["raw_output"], prompt_hash, meta

        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)

        providers_to_try = [self.provider]
        if self.allow_cross_provider_fallback:
            other = "gemini" if self.provider == "groq" else "groq"
            providers_to_try.append(other)

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            for prov in providers_to_try:
                ordered_keys = self._get_ordered_keys(prov)
                if not ordered_keys:
                    continue
                active_model = (
                    self.model_id
                    if prov == self.provider
                    else (DEFAULT_GROQ_MODEL if prov == "groq" else DEFAULT_GEMINI_MODEL)
                )

                for key_label, api_key in ordered_keys:
                    try:
                        self._last_call_ts = time.time()
                        if prov == "groq":
                            raw_out, call_meta = _call_groq_http(
                                api_key,
                                active_model,
                                system_prompt,
                                user_prompt,
                                temperature=self.temperature,
                            )
                        else:
                            raw_out, call_meta = _call_gemini_http(
                                api_key,
                                active_model,
                                system_prompt,
                                user_prompt,
                                temperature=self.temperature,
                            )

                        parsed = parse_and_validate_answer_json(raw_out, context_prompt=user_prompt)
                        call_meta["key_label"] = key_label
                        call_meta["model_id"] = active_model
                        call_meta["cached"] = False
                        self._key_cursor[prov] = (self._key_cursor.get(prov, 0) + 1) % max(
                            1, len(ordered_keys)
                        )

                        if use_cache:
                            save_cached_response(
                                active_model,
                                prompt_hash,
                                raw_out,
                                parsed.model_dump(mode="json"),
                                metadata=call_meta,
                                cache_dir=self.cache_dir,
                            )
                        return parsed, raw_out, prompt_hash, call_meta

                    except Exception as exc:
                        last_err = exc
                        self._key_cursor[prov] = (self._key_cursor.get(prov, 0) + 1) % max(
                            1, len(ordered_keys)
                        )
                        err_str = str(exc)
                        if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                            m_wait = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                            if m_wait:
                                time.sleep(min(20.0, float(m_wait.group(1)) + 0.5))
                            else:
                                time.sleep(min(12.0, 2.0 * attempt))
                        else:
                            time.sleep(0.5)

            time.sleep(min(15.0, 2.0 * attempt))

        raise RuntimeError(
            f"All API attempts failed after {max_retries} retries. Last error: {last_err}"
        )


def generate(
    context: str,
    config: dict[str, Any],
) -> AnswerJSON:
    """Contract function `generate(context, config) -> AnswerJSON` (cached) (Section 12.2)."""
    provider = str(config.get("provider", "groq"))
    model_id = config.get("model_id")
    patient_id = str(config.get("patient_id", "UNKNOWN"))
    question_text = str(config.get("question_text", ""))
    as_of_date = str(config.get("as_of_date", ""))
    window_start = config.get("window_start")
    background_text = config.get("background_text")
    key_label = config.get("preferred_key_label")
    allow_fallback = bool(config.get("allow_cross_provider_fallback", False))

    client = MultiKeyLLMClient(
        provider=provider,
        model_id=model_id,
        temperature=float(config.get("temperature", 0.0)),
        preferred_key_label=key_label,
        allow_cross_provider_fallback=allow_fallback,
    )
    user_prompt = build_user_prompt(
        patient_id=patient_id,
        question_text=question_text,
        as_of_date=as_of_date,
        context_text=context,
        window_start=window_start,
        background_text=background_text,
    )
    parsed, _, _, _ = client.generate_from_prompts(SYSTEM_PROMPT, user_prompt)
    return parsed
