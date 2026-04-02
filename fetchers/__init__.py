"""
fetchers/__init__.py

Unified fetch interface. Routes to the correct fetcher based on ats_type,
then applies rule-based filters from preferences.yaml.
"""

import json
import re
import yaml
from pathlib import Path

from fetchers import greenhouse, lever, ashby, workable, workday, symbotic

PREFS_PATH = Path(__file__).parent.parent / "config" / "preferences.yaml"


def _load_prefs() -> dict:
    return yaml.safe_load(PREFS_PATH.read_text())


def fetch_company(company_row) -> list[dict]:
    """
    Fetch all current job postings for a company and apply rule-based filters.

    Parameters
    ----------
    company_row : sqlite3.Row
        A row from the companies table (must have ats_type, ats_id, id, name).

    Returns
    -------
    list[dict]
        Normalized job dicts with filter flags set:
        external_id, title, url, description, location, date_posted,
        exclude_flag, geo_match, title_excluded
    Raises on fetch failure so the caller can decide whether to mark jobs disappeared.
    """
    ats_type = company_row["ats_type"]
    ats_id   = company_row["ats_id"]

    if ats_type == "greenhouse":
        raw_jobs = greenhouse.fetch(ats_id)
    elif ats_type == "lever":
        raw_jobs = lever.fetch(ats_id)
    elif ats_type == "ashby":
        raw_jobs = ashby.fetch(ats_id)
    elif ats_type == "workable":
        raw_jobs = workable.fetch(ats_id)
    elif ats_type == "workday":
        raw_jobs = workday.fetch(ats_id)
    elif ats_type == "symbotic":
        raw_jobs = symbotic.fetch(ats_id)
    else:
        raise NotImplementedError(f"Fetcher not implemented for ats_type='{ats_type}'")

    prefs = _load_prefs()
    title_filter = _parse_title_filter(company_row)
    return [_apply_filters(job, prefs, title_filter) for job in raw_jobs]


def _parse_title_filter(company_row) -> list[str] | None:
    """Extract the company's title_filter include list, or None if not set."""
    raw = company_row["title_filter"] if "title_filter" in company_row.keys() else None
    if not raw:
        return None
    try:
        patterns = json.loads(raw)
        return [p.lower() for p in patterns if p]
    except Exception:
        return None


def _apply_filters(job: dict, prefs: dict, title_filter: list[str] | None = None) -> dict:
    """Annotate a job dict with geo_match, title_excluded, exclude_flag."""
    job["geo_match"]       = _check_geo(job.get("location", ""), prefs)
    job["title_excluded"]  = _check_title_excluded(job.get("title", ""), prefs)
    job["exclude_flag"]    = _check_exclude(
        job.get("title", "") + " " + (job.get("description", "") or ""), prefs
    )
    # Per-company include filter: if set, job must match at least one keyword
    # in title or description, otherwise mark as excluded.
    if title_filter and not job["title_excluded"]:
        title = (job.get("title") or "").lower()
        desc  = (job.get("description") or "").lower()
        if not any(kw in title or kw in desc for kw in title_filter):
            job["title_excluded"] = 1
    return job


def _check_geo(location: str, prefs: dict) -> int:
    """Return 1 if location matches allowed geography, 0 if not, None if unknown."""
    if not location:
        return None
    allowed = prefs.get("geography", {}).get("allowed", [])
    loc_lower = location.lower()

    # Flexible locations should not be treated as mismatches.
    if any(term in loc_lower for term in ["remote", "any location", "anywhere", "multiple locations"]):
        return 1

    allowed_lower = [p.lower() for p in allowed]

    # If preferences include Boston/Cambridge, treat MA locations as acceptable.
    if ("boston" in allowed_lower or "cambridge" in allowed_lower) and (
        ", ma" in loc_lower or "massachusetts" in loc_lower
    ):
        return 1

    # If preferences include NYC/New York, treat NJ (NYC metro) as acceptable.
    if ("new york" in allowed_lower or "nyc" in allowed_lower) and (
        ", nj" in loc_lower or "new jersey" in loc_lower
    ):
        return 1

    for place in allowed:
        if place.lower() in loc_lower:
            return 1
    return 0


def _check_title_excluded(title: str, prefs: dict) -> int:
    """Return 1 if the job title matches an exclude keyword or contract signal."""
    role_types = prefs.get("role_types", {})
    exclude_list = role_types.get("exclude_titles", [])
    contract_list = role_types.get("contract_signals", [])
    title_lower = title.lower()
    for kw in exclude_list + contract_list:
        if kw.lower() in title_lower:
            return 1
    return 0


def _check_exclude(text: str, prefs: dict) -> int:
    """Return 1 if the job text contains configured exclusion signals."""
    text_lower = text.lower()
    strong_terms, context_terms, generic_terms = _exclude_terms_from_prefs(prefs)

    for term in strong_terms:
        if _contains_term(text_lower, term):
            return 1

    # Generic signals only trigger when contextualized.
    if generic_terms and context_terms and any(_contains_term(text_lower, g) for g in generic_terms) and any(
        _contains_term(text_lower, c) for c in context_terms
    ):
        return 1

    return 0


def _contains_term(text: str, term: str) -> bool:
    """Case-insensitive token-aware term matching."""
    t = (term or "").strip().lower()
    if not t:
        return False
    pattern = r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def _exclude_terms_from_prefs(prefs: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Build exclusion matching terms from preferences.
    Preferred keys:
      - hard_excludes.exclude_strong_signals
      - hard_excludes.exclude_context_terms
      - hard_excludes.exclude_generic_terms
    Backward-compatible fallback uses hard_excludes.exclude_signals.
    """
    hard = prefs.get("hard_excludes", {})
    all_signals = [str(s).strip().lower() for s in hard.get("exclude_signals", []) if str(s).strip()]

    strong = [str(s).strip().lower() for s in hard.get("exclude_strong_signals", []) if str(s).strip()]
    context = [str(s).strip().lower() for s in hard.get("exclude_context_terms", []) if str(s).strip()]
    generic = [str(s).strip().lower() for s in hard.get("exclude_generic_terms", []) if str(s).strip()]

    if not strong:
        strong = [s for s in all_signals if s not in set(generic)] if generic else list(all_signals)
    if generic and not context:
        # If generic terms were configured without context terms, treat them as strong.
        strong.extend(generic)
        generic = []

    return list(dict.fromkeys(strong)), list(dict.fromkeys(context)), list(dict.fromkeys(generic))
