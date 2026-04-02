"""
agents/tools.py

Tool functions and provider-agnostic tool schema dicts.

Three tools:
  check_new_jobs(company_name)  — fetch + store new postings for a company
  score_all_new_jobs()          — score all unscored jobs against resume + preferences
  get_briefing_data()           — return high-scoring jobs as a markdown table

Each tool function is paired with a metadata dict (tool_*) that describes
it in OpenAI-compatible tool schema format for function calling.
"""

import json
import os
import re
import yaml
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

from db import database as db
from fetchers import fetch_company
from utils.resume_parser import get_resume_text
from agents.functions import get_ollama_client

CONFIG_DIR  = Path(__file__).parent.parent / "config"
PREFS_PATH  = CONFIG_DIR / "preferences.yaml"
PREFS_MD    = CONFIG_DIR / "preferences.md"

PROMPT_VERSION = "v3"


def _load_prefs() -> dict:
    return yaml.safe_load(PREFS_PATH.read_text())


def _get_model() -> str:
    cfg = _load_prefs()
    return cfg.get("model", "gpt-4o")


def _get_provider() -> str:
    cfg = _load_prefs()
    return str(cfg.get("provider", "openai")).strip().lower()


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o")


def _resolve_provider(model: str, provider: str | None) -> str:
    if provider:
        return provider.strip().lower()
    # Infer from model name when provider is omitted.
    return "openai" if _is_openai_model(model) else "ollama"


def _component_weighted_score(parsed: dict) -> float | None:
    """Compute weighted score from component fields when available."""
    needed = ["industry_fit", "role_fit", "location_fit", "level_fit", "exclude_risk"]
    if not all(k in parsed for k in needed):
        return None
    industry = float(parsed["industry_fit"])
    role = float(parsed["role_fit"])
    location = float(parsed["location_fit"])
    level = float(parsed["level_fit"])
    risk = float(parsed["exclude_risk"])  # 10 means low risk, 0 means high risk
    score = (
        0.10 * industry +
        0.55 * role +
        0.15 * location +
        0.15 * level +
        0.05 * risk
    )
    return score


def _contains_term(text: str, term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    pattern = r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def _exclude_terms_from_prefs(prefs: dict) -> tuple[list[str], list[str], list[str]]:
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


def _has_strong_exclude_signal(text: str, prefs: dict) -> bool:
    strong, context, generic = _exclude_terms_from_prefs(prefs)
    if any(_contains_term(text, s) for s in strong):
        return True
    if any(_contains_term(text, g) for g in generic) and any(_contains_term(text, c) for c in context):
        return True
    return False


def _geo_score_eligible(location: str | None) -> bool:
    """
    Return True if the job's location is in-scope for AI scoring.
    Blank/unknown locations pass through (could be remote or multi-location).
    Out-of-geo locations (e.g. Austin TX, Seattle WA) are skipped to avoid
    wasting API calls.
    """
    if not location or not location.strip():
        return True  # unknown — pass through

    loc = location.lower()

    # Remote / anywhere signals
    if any(term in loc for term in ["remote", "any location", "anywhere", "multiple locations"]):
        return True

    # New York state / city
    if any(term in loc for term in ["new york", "nyc", ", ny", "ny,", "brooklyn", "manhattan"]):
        return True

    # Massachusetts
    if any(term in loc for term in ["boston", "cambridge", ", ma", "ma,", "massachusetts"]):
        return True

    # New Jersey (close enough to NYC)
    if any(term in loc for term in [", nj", "nj,", "new jersey"]):
        return True

    return False


def _apply_score_guardrails(job, score: float) -> tuple[float, list[str]]:
    """Apply deterministic caps/adjustments for obvious mismatch signals."""
    notes = []
    title = (job["title"] or "").lower()

    if job["title_excluded"] == 1:
        score = min(score, 3.5)
        notes.append("excluded_title_cap")

    # Leadership/non-target roles should not score like core IC systems roles.
    if re.search(r"\b(vice president|vp|director|chief|counsel)\b", title):
        score = min(score, 5.0)
        notes.append("leadership_cap")
    elif re.search(r"\b(program manager|project manager|technical project manager|technical program manager)\b", title):
        score = min(score, 7.2)
        notes.append("pm_cap")

    # Positive signal for directly aligned systems/integration requirements work.
    if re.search(r"\b(systems engineer|requirements|integration)\b", title):
        score += 0.5
        notes.append("systems_bonus")

    return round(min(max(score, 0), 10), 1), notes


# ---------------------------------------------------------------------------
# Tool 1: check_new_jobs
# ---------------------------------------------------------------------------

def check_new_jobs(company_name: str) -> str:
    """
    Fetch current job postings for a company, store new ones in the DB,
    and mark any that have disappeared.

    Returns a plain-text summary of what was found.
    """
    company = db.get_company_by_name(company_name)
    if company is None:
        return f"Company '{company_name}' not found in database."

    if company["ats_type"] == "scrape":
        return f"Skipped {company_name}: no supported fetcher (scrape)."

    try:
        jobs = fetch_company(company)
    except NotImplementedError as e:
        return f"Fetcher not available for {company_name}: {e}"
    except Exception as e:
        return f"Fetch failed for {company_name}: {e}"

    new_count = 0
    seen_ids  = set()

    for job in jobs:
        external_id = job.get("external_id")
        if external_id:
            seen_ids.add(external_id)

        job_id, is_new = db.upsert_job(
            company_id    = company["id"],
            external_id   = external_id,
            title         = job["title"],
            url           = job.get("url"),
            description   = job.get("description"),
            location      = job.get("location"),
            date_posted   = job.get("date_posted"),
            exclude_flag  = job.get("exclude_flag", 0),
            geo_match     = job.get("geo_match"),
            title_excluded= job.get("title_excluded", 0),
        )
        if is_new:
            new_count += 1

    # Fetch succeeded, so reconcile missing ATS-backed jobs even when
    # the board is now empty (seen_ids == empty set).
    db.mark_missing_jobs(company["id"], seen_ids)

    total = len(jobs)
    return (
        f"{company_name}: fetched {total} postings, {new_count} new. "
        f"({total - new_count} already seen)"
    )


tool_check_new_jobs = {
    "type": "function",
    "function": {
        "name": "check_new_jobs",
        "description": "Fetch current job postings for a company and store any new ones in the database.",
        "parameters": {
            "type": "object",
            "required": ["company_name"],
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "The exact company name as stored in the database.",
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 2: score_all_new_jobs
# ---------------------------------------------------------------------------

def score_all_new_jobs(show_progress: bool = True) -> str:
    """
    Score all unscored open jobs against the resume and preferences.
    Makes one provider model call per job. Returns a summary of scores assigned.
    """
    jobs = db.get_new_unscored_jobs()
    if not jobs:
        return "No new unscored jobs to evaluate."

    resume_text = get_resume_text()
    prefs_text  = PREFS_MD.read_text(encoding="utf-8") if PREFS_MD.exists() else ""
    model       = _get_model()
    provider    = _get_provider()

    # Apply pre-filters before starting the progress bar so the count reflects
    # jobs that will actually get an API call.
    eligible = []
    for job in jobs:
        if not _geo_score_eligible(job["location"]):
            db.mark_job_unscored(job["id"], "Skipped by pre-filter: out_of_geo", model, PROMPT_VERSION)
        else:
            eligible.append(job)

    skipped = len(jobs) - len(eligible)
    if skipped and show_progress:
        print(f"  {skipped} job(s) skipped by geo pre-filter, {len(eligible)} to score.")

    if not eligible:
        return "No new unscored jobs to evaluate (all skipped by pre-filter)."

    results = []
    iterable = eligible
    if show_progress and len(eligible) > 0:
        iterable = tqdm(eligible, total=len(eligible), desc="  Scoring", unit="job")

    for job in iterable:

        result = _score_one_job(job, resume_text, prefs_text, model, provider=provider, return_components=True)
        if isinstance(result, tuple) and len(result) == 3:
            score, rationale, components = result
        else:
            score, rationale = result
            components = None
        db.update_job_score(
            job_id        = job["id"],
            score         = score,
            rationale     = rationale,
            model_name    = model,
            prompt_version= PROMPT_VERSION,
            components    = components,
        )
        results.append({
            "company": job["company_name"],
            "title":   job["title"],
            "score":   score,
            "rationale": rationale,
        })

    lines = [f"Scored {len(results)} job(s):"]
    for r in sorted(results, key=lambda x: -x["score"]):
        lines.append(f"  [{r['score']:.0f}/10] {r['company']} — {r['title']}: {r['rationale']}")
    return "\n".join(lines)


def _score_one_job(
    job,
    resume_text: str,
    prefs_text: str,
    model: str,
    provider: str | None = None,
    return_components: bool = False,
):
    """Call provider model to score a single job."""
    system_prompt = f"""You are a strict job-fit evaluator. Score job postings against a candidate's resume and preferences.

Return ONLY valid JSON in this exact format:
{{
  "industry_fit": <0-10>,
  "role_fit": <0-10>,
  "location_fit": <0-10>,
  "level_fit": <0-10>,
  "exclude_risk": <0-10>,
  "score": <0-10>,
  "rationale": "<1-2 concise sentences>"
}}

## Component definitions

**industry_fit** — how well the company's domain matches target industries (fusion/fission energy, robotics, medtech, advanced manufacturing). Score the company and domain, not the role.

**role_fit** — score the *function of the work*, completely independent of the company. Ask: does this role involve systems-level thinking — requirements management, system integration, verification & validation, interface control, cross-subsystem ownership?
- 8–10: Direct match — systems engineer, integration engineer, test/V&V engineer, TPM/PM with deep technical scope, solutions/sales engineer requiring systems knowledge
- 5–7: Adjacent — hardware product engineer with some integration scope, technical lead touching multiple subsystems
- 2–4: Specialist discipline — controls, mechanical design, software, neutronics, operations, manufacturing, process engineering — even at a top-tier company
- 0–1: Non-technical — finance, legal, HR, marketing, recruiting

**CRITICAL**: industry_fit and role_fit are fully independent dimensions. A compelling company (high industry_fit) does NOT raise role_fit. Score role_fit on what the role actually does.

A role that *coordinates with* or *supports* systems teams is not the same as a systems engineering role.
- ConOps authoring, operational program development, and procedure writing are operations discipline work (role_fit 5–6), not direct systems engineering (8–10), even if the description mentions systems integration.
- Shift operations, plant operations, and facility operations (24/7 operation of equipment, commissioning, maintenance) are specialist/technician-adjacent work (role_fit 2–3).

**location_fit** — 10 if NYC/Boston area or remote; 5 if adjacent metro (NJ, CT); 0 if elsewhere.

**level_fit** — read the job description's stated experience requirements and compare to the candidate's background (see Seniority in candidate preferences above).
- 8–10: Description requires 3–7 years, or up to 8 years if an advanced degree is accepted in lieu of experience
- 5–7: Description requires 7–10 years; candidate is junior but not a dealbreaker
- 2–4: Description requires 10+ years, or the role is clearly entry-level/intern (wrong in either direction)
- 0–1: Explicitly VP/Director/C-suite seniority

If the description does not state years explicitly, use the title seniority prefix as a fallback: no prefix or "Associate" → 8–10; "Senior" or "Lead" → 5–7; "Principal", "Staff", or "Distinguished" → 2–4; "VP", "Director", "Chief" → 0–1.

**exclude_risk** — 10 = no exclusion risk; 0 = hard exclude based on configured exclusion terms.

**score** — holistic 0–10 reflecting the full picture. Must use the full range — avoid clustering in the 7–8 band. A great company with a mismatched role should score 4–6, not 7–8.

CANDIDATE RESUME:
{resume_text}

CANDIDATE PREFERENCES:
{prefs_text}"""

    user_message = f"""Please score this job posting:

Company: {job['company_name']}
Title: {job['title']}
Location: {job['location'] or 'Not specified'}
Description:
{(job['description'] or '')[:3000]}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    try:
        provider = _resolve_provider(model, provider)
        if provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
        elif provider == "ollama":
            client = get_ollama_client()
            response = client.chat(model=model, messages=messages, stream=False)
            result = response if isinstance(response, dict) else response.model_dump()
            content = result["message"]["content"].strip()
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Use 'openai' or 'ollama'.")

        # Strip markdown code fences if the model wraps JSON in them
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: recover first JSON object if model wrapped output.
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                raise
            parsed = json.loads(m.group(0))

        component_score = _component_weighted_score(parsed)
        raw_score = float(parsed["score"]) if "score" in parsed else None
        rationale = str(parsed.get("rationale", "")).strip() or "No rationale provided."

        if component_score is None and raw_score is None:
            raise ValueError("Model response missing both score and component fields")

        if component_score is not None and raw_score is not None:
            # Favor component-based score for better differentiation.
            score = 0.75 * component_score + 0.25 * raw_score
        else:
            score = component_score if component_score is not None else raw_score

        # Cap score when role_fit is clearly mismatched, regardless of industry attractiveness.
        # This prevents a great company from rescuing a fundamentally wrong role type.
        role_fit_val = float(parsed.get("role_fit", 10))
        role_cap_note = None
        if role_fit_val <= 3:
            score = min(score, 4.5)
            role_cap_note = "low_role_fit_cap"
        elif role_fit_val <= 4:
            score = min(score, 5.5)
            role_cap_note = "low_role_fit_cap"

        pre_guardrail = round(min(max(float(score), 0), 10), 1)
        score, notes = _apply_score_guardrails(job, score)
        if role_cap_note:
            notes = [role_cap_note] + notes
        if notes:
            rationale = f"{rationale} [adjustments: {', '.join(notes)}]"

        components = {
            "industry_fit": float(parsed["industry_fit"]) if "industry_fit" in parsed else None,
            "role_fit": float(parsed["role_fit"]) if "role_fit" in parsed else None,
            "location_fit": float(parsed["location_fit"]) if "location_fit" in parsed else None,
            "level_fit": float(parsed["level_fit"]) if "level_fit" in parsed else None,
            "exclude_risk": float(parsed["exclude_risk"]) if "exclude_risk" in parsed else None,
            "raw_score": float(raw_score) if raw_score is not None else None,
            "component_score": round(float(component_score), 2) if component_score is not None else None,
            "pre_guardrail_score": pre_guardrail,
            "final_score": float(score),
            "adjustments": notes,
        }
        if return_components:
            return score, rationale, components
        return score, rationale

    except Exception as e:
        if return_components:
            return 0.0, f"Scoring failed: {e}", None
        return 0.0, f"Scoring failed: {e}"


tool_score_all_new_jobs = {
    "type": "function",
    "function": {
        "name": "score_all_new_jobs",
        "description": (
            "Score all unscored job postings against the candidate's resume and preferences. "
            "Returns a summary of scores assigned."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 3: get_briefing_data
# ---------------------------------------------------------------------------

def get_briefing_data() -> str:
    """
    Return high-scoring open jobs as a markdown table for the Reporter agent.
    """
    jobs = db.get_briefing_jobs(min_score=6.0)
    if not jobs:
        return "No high-scoring new jobs to report."

    rows = [
        {
            "ID":        job["id"],
            "Score":     f"{job['ai_score']:.1f}",
            "Company":   job["company_name"],
            "Title":     job["title"],
            "Location":  job["location"] or "",
            "Found":     (job["date_found"] or "")[:10],
            "Rationale": job["ai_rationale"] or "",
            "URL":       job["url"] or "",
        }
        for job in jobs
    ]
    return tabulate(rows, headers="keys", tablefmt="github")


tool_get_briefing_data = {
    "type": "function",
    "function": {
        "name": "get_briefing_data",
        "description": (
            "Retrieve high-scoring open job postings (score >= 6) as a table. "
            "Use this to generate the final briefing."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 4: get_pipeline_data
# ---------------------------------------------------------------------------

def get_pipeline_data() -> str:
    """
    Return a summary of jobs currently in the application pipeline
    (applied, interviewing, rejected, offer), grouped by company.
    """
    conn = db.get_connection()
    rows = conn.execute(
        """
        SELECT j.title, j.application_state, c.name AS company_name
        FROM jobs j
        JOIN companies c ON j.company_id = c.id
        WHERE j.application_state IN ('applied', 'interviewing', 'rejected', 'offer')
        ORDER BY c.name, j.application_state
        """
    ).fetchall()

    if not rows:
        return "No jobs currently in the application pipeline."

    table = [
        {
            "Company": r["company_name"],
            "Title":   r["title"],
            "Status":  r["application_state"],
        }
        for r in rows
    ]
    return tabulate(table, headers="keys", tablefmt="github")


tool_get_pipeline_data = {
    "type": "function",
    "function": {
        "name": "get_pipeline_data",
        "description": (
            "Retrieve jobs currently in the application pipeline "
            "(applied, interviewing, rejected, offer), grouped by company. "
            "Use this to avoid recommending more jobs from companies already being pursued."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 5: search_jobs
# ---------------------------------------------------------------------------

def search_jobs(query: str, limit: int = 10) -> str:
    """
    Search jobs using FTS keywords and return a markdown table.
    This is intended for interactive Q&A retrieval.
    """
    q = (query or "").strip()
    if not q:
        return "No query provided."

    try:
        rows = db.search_jobs(q)
    except Exception as e:
        return f"Search failed: {e}"

    if not rows:
        return "No matching jobs found."

    top = rows[: max(1, min(int(limit), 50))]
    data = [
        {
            "ID": r["id"],
            "Score": f"{r['ai_score']:.1f}" if r["ai_score"] is not None else "",
            "Company": r["company_name"],
            "Title": r["title"],
            "Location": r["location"] or "",
            "State": f"{r['job_state']}/{r['application_state']}",
            "URL": r["url"] or "",
        }
        for r in top
    ]
    return tabulate(data, headers="keys", tablefmt="github")


tool_search_jobs = {
    "type": "function",
    "function": {
        "name": "search_jobs",
        "description": (
            "Keyword search over stored job titles/descriptions. "
            "Use concise keywords like 'systems integration fusion Boston'."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "FTS keyword query to find relevant jobs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (1-50).",
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 5: get_job_details
# ---------------------------------------------------------------------------

def get_job_details(job_id: int) -> str:
    """Return detailed context for one job ID."""
    conn = db.get_connection()
    row = conn.execute(
        """
        SELECT j.*, c.name as company_name
        FROM jobs j
        JOIN companies c ON c.id = j.company_id
        WHERE j.id = ?
        """,
        (job_id,),
    ).fetchone()
    if not row:
        return f"No job found with id={job_id}."

    comps = ""
    if row["ai_components"]:
        try:
            parsed = json.loads(row["ai_components"])
            parts = []
            for k in ["industry_fit", "role_fit", "location_fit", "level_fit", "exclude_risk", "final_score"]:
                if parsed.get(k) is not None:
                    parts.append(f"{k}={parsed[k]}")
            if parts:
                comps = ", ".join(parts)
        except Exception:
            comps = "unparseable"

    desc = (row["description"] or "")[:1800]
    lines = [
        f"ID: {row['id']}",
        f"Company: {row['company_name']}",
        f"Title: {row['title']}",
        f"Location: {row['location'] or ''}",
        f"Score: {row['ai_score'] if row['ai_score'] is not None else 'unscored'}",
        f"Rationale: {row['ai_rationale'] or ''}",
        f"Components: {comps}",
        f"Application state: {row['application_state']}",
        f"Job state: {row['job_state']}",
        f"URL: {row['url'] or ''}",
        "Description:",
        desc,
    ]
    return "\n".join(lines)


tool_get_job_details = {
    "type": "function",
    "function": {
        "name": "get_job_details",
        "description": "Get full details for a job ID: title, company, location, score, rationale, application state, URL, and the full job description text. Use this to read a job description before assessing fit.",
        "parameters": {
            "type": "object",
            "required": ["job_id"],
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "Internal job ID from the tracker database.",
                },
            },
        },
    },
}
