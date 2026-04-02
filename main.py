"""
main.py — Job Tracker CLI

Usage:
  python main.py add <url> [--ats-type TYPE] [--ats-id SLUG] [--title-filter "kw1,kw2"]
  python main.py set-url <company_id> <url>   Update the careers URL for a company
  python main.py set-title-filter <company_id> ["kw1,kw2"]  Set or clear per-company title filter
  python main.py add-file <path>     Bulk add companies from text file
  python main.py export-companies <path>  Export monitored companies to text file
  python main.py list                List all monitored companies
  python main.py refresh [--no-brief]  Run full pipeline: fetch → score → report (skip briefing with --no-brief)
  python main.py status              Print job + application pipeline summary
  python main.py score <job_id>      Manually score a specific job
  python main.py rescore all [--model MODEL]
  python main.py rescore <job_id> [<job_id> ...] [--model MODEL]
  python main.py apply <job_id|range> ...      Mark as applied
  python main.py skip <job_id|range> ...       Mark as skipped
  python main.py review <job_id|range> ...     Mark as reviewed
  python main.py interview <job_id|range> ...  Mark as interviewing
  python main.py reject <job_id|range> ...     Mark as rejected
  python main.py offer <job_id|range> ...      Mark as offer received
  python main.py reset <job_id|range> ...      Reset to new (unreviewed)
  python main.py jobs                      List high-scoring new jobs
  python main.py jobs-all                  List all jobs sorted by score
  python main.py jobs-new                  List jobs in new state
  python main.py jobs-reviewed             List reviewed jobs
  python main.py jobs-pipeline             List all jobs in the application pipeline (applied/interviewing/rejected/offer)
  python main.py jobs-applied              List applied jobs
  python main.py jobs-skipped              List skipped jobs
  python main.py jobs-interviewing         List jobs in interviewing
  python main.py jobs-rejected             List rejected jobs
  python main.py jobs-offer                List jobs with offers
  python main.py jobs-disappeared          List jobs that disappeared from job boards
  python main.py job <job_id> [--full]
  python main.py ask <question>      Ask a question and retrieve relevant jobs via tools
  python main.py doctor [--model MODEL]
"""

import sys
import re
from tqdm import tqdm
import shutil
import os
import textwrap
import json
from urllib.parse import urlparse
from db.database import (
    init_db,
    upsert_company,
    get_company_by_name,
    list_companies,
    get_briefing_jobs,
    update_job_score,
    mark_job_unscored,
    update_job_states,
    get_connection,
    set_company_title_filter,
    set_company_ats,
    set_company_url,
)
from fetchers.detector import detect
from agents.tools import (
    check_new_jobs,
    get_briefing_data,
    get_pipeline_data,
    search_jobs,
    get_job_details,
    tool_search_jobs,
    tool_get_briefing_data,
    tool_get_pipeline_data,
    tool_get_job_details,
    _score_one_job,
    _geo_score_eligible,
)
from agents.pipeline import run_pipeline
from agents.functions import agent_run, agent_loop
from utils.resume_parser import get_resume_text, DOCX_PATH, PDF_PATH, CACHE_PATH
import yaml
from pathlib import Path
from agents.tools import PROMPT_VERSION
from tabulate import tabulate
from tqdm import tqdm

PREFS_PATH = Path(__file__).parent / "config" / "preferences.yaml"


def _get_model() -> str:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    return cfg.get("model", "gpt-4o")


def _get_provider() -> str:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    return str(cfg.get("provider", "openai")).strip().lower()


def _get_ask_model() -> str:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    return cfg.get("ask_model", _get_model())


def _get_ask_provider() -> str:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    return str(cfg.get("ask_provider", _get_provider())).strip().lower()


def _term_width(default: int = 120) -> int:
    return shutil.get_terminal_size(fallback=(default, 24)).columns


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o")


def _resolve_provider(model: str, model_override: bool = False) -> str:
    # For ad-hoc --model overrides without explicit provider, infer from model name.
    if model_override:
        return "openai" if _is_openai_model(model) else "ollama"
    return _get_provider()


def _extract_adjustments(rationale: str | None) -> tuple[str, str]:
    """
    Split rationale into plain text and adjustment suffix.
    Returns (base_rationale, adjustments_text).
    """
    text = (rationale or "").strip()
    marker = " [adjustments:"
    if marker in text and text.endswith("]"):
        idx = text.rfind(marker)
        base = text[:idx].strip()
        adj = text[idx + 2 : -1]  # drop leading '[' and trailing ']'
        return base, adj
    return text, ""


def _prefilter_skip_reasons(job) -> list[str]:
    """Return pre-filter reasons for skipping scoring."""
    reasons = []
    if job["job_state"] != "open":
        reasons.append(f"job_state={job['job_state']}")
    if job["title_excluded"] == 1:
        reasons.append("excluded_title")
    if not _geo_score_eligible(job["location"]):
        reasons.append("out_of_geo")
    return reasons


def cmd_add(url: str, ats_type_override: str | None = None, ats_id_override: str | None = None, title_filter: list[str] | None = None):
    """Detect ATS from URL, prompt for company name, add to DB."""
    if ats_type_override:
        ats_type = ats_type_override.strip().lower()
        ats_id = ats_id_override
        print(f"Using manual ATS override: {ats_type}" + (f" (slug: {ats_id})" if ats_id else ""))
    else:
        print(f"Detecting ATS for: {url}")
        ats_type, ats_id = detect(url)
        print(f"  Detected: {ats_type}" + (f" (slug: {ats_id})" if ats_id else ""))

    if ats_type == "scrape":
        print("  Warning: no supported ATS detected. Company will be added but fetching is not yet supported.")

    if ats_type == "ashby" and not ats_id:
        ats_id = input("Ashby slug not auto-detected. Enter slug (e.g. 'formenergy'): ").strip() or None

    name = input("Company name: ").strip()
    if not name:
        print("Aborted — no name provided.")
        return

    existing = get_company_by_name(name)
    if existing:
        print(f"'{name}' is already in the database (id={existing['id']}).")
        return

    company_id = upsert_company(name, url, ats_type, ats_id, title_filter=title_filter)
    filter_note = f", title_filter={title_filter}" if title_filter else ""
    print(f"Added '{name}' (id={company_id}, ats={ats_type}, slug={ats_id}{filter_note})")


def _company_exists_by_url(url: str) -> bool:
    conn = get_connection()
    row = conn.execute("SELECT id FROM companies WHERE careers_url=?", (url,)).fetchone()
    return row is not None


def _derive_company_name(url: str, ats_id: str | None) -> str:
    """Derive a readable company name when input line only contains a URL."""
    if ats_id:
        return ats_id.replace("-", " ").replace("_", " ").title()

    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return "Unknown Company"
    stem = host.split(".")[0]
    return stem.replace("-", " ").replace("_", " ").title()


def _parse_company_line(line: str) -> tuple[str | None, str]:
    """
    Parse one import line.
    Supported:
      - name | url
      - name, url
      - url
    Returns (name_or_none, url).
    """
    raw = line.strip()
    if "|" in raw:
        name, url = [x.strip() for x in raw.split("|", 1)]
    elif "," in raw and not raw.lower().startswith(("http://", "https://")):
        name, url = [x.strip() for x in raw.split(",", 1)]
    else:
        name, url = None, raw

    if not url or not url.lower().startswith(("http://", "https://")):
        raise ValueError("expected URL beginning with http:// or https://")
    if name is not None and not name:
        raise ValueError("empty company name")
    return name, url


def cmd_add_file(path_str: str):
    """Bulk add companies from a text file."""
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    added = 0
    skipped = 0
    errored = 0

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        try:
            name, url = _parse_company_line(stripped)
            ats_type, ats_id = detect(url)
            final_name = name or _derive_company_name(url, ats_id)

            if get_company_by_name(final_name):
                skipped += 1
                print(f"[line {line_no}] SKIP name exists: {final_name}")
                continue
            if _company_exists_by_url(url):
                skipped += 1
                print(f"[line {line_no}] SKIP url exists: {url}")
                continue

            upsert_company(final_name, url, ats_type, ats_id)
            added += 1
            print(f"[line {line_no}] ADDED {final_name} ({ats_type})")
        except Exception as e:
            errored += 1
            print(f"[line {line_no}] ERROR {e}")

    print(f"Import complete: {added} added, {skipped} skipped, {errored} errors.")


def cmd_export_companies(path_str: str):
    """
    Export monitored companies to a text file compatible with `add-file`.
    Output format: `Company Name | URL`
    """
    companies = list_companies()
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for c in companies:
        lines.append(f"{c['name']} | {c['careers_url']}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Exported {len(lines)} companies to {path}")


def cmd_list():
    """List all companies with their monitoring status."""
    companies = list_companies()
    if not companies:
        print("No companies in database. Add one with: python main.py add <url>")
        return

    width = _term_width()
    rows = []
    for c in companies:
        try:
            tf = json.loads(c["title_filter"]) if c["title_filter"] else []
        except Exception:
            tf = []
        rows.append(
            {
                "ID": c["id"],
                "Active": "yes" if c["active"] else "no",
                "ATS": c["ats_type"],
                "Slug": c["ats_id"] or "",
                "Name": c["name"],
                "TitleFilter": ", ".join(tf) if tf else "",
            }
        )

    slug_w = max(12, min(24, int(width * 0.18)))
    name_w = max(16, min(36, int(width * 0.28)))
    filter_w = max(16, min(40, int(width * 0.3)))
    print(tabulate(rows, headers="keys", tablefmt="simple", maxcolwidths=[4, 6, 12, slug_w, name_w, filter_w]))


def cmd_set_ats(company_id: int, ats_type: str, ats_id: str | None):
    """Update the ATS type and slug for an existing company."""
    conn = get_connection()
    company = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    if not company:
        print(f"No company found with id={company_id}")
        return
    set_company_ats(company_id, ats_type, ats_id)
    print(f"Updated '{company['name']}': ats_type={ats_type}, ats_id={ats_id}")


def cmd_set_url(company_id: int, url: str):
    """Update the careers URL for an existing company."""
    conn = get_connection()
    company = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    if not company:
        print(f"No company found with id={company_id}")
        return
    set_company_url(company_id, url)
    print(f"Updated '{company['name']}': url={url}")


def cmd_set_title_filter(company_id: int, patterns: list[str] | None):
    """Set or clear the per-company title/description include filter."""
    conn = get_connection()
    company = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    if not company:
        print(f"No company found with id={company_id}")
        return
    set_company_title_filter(company_id, patterns)
    if patterns:
        print(f"Title filter set for '{company['name']}': {patterns}")
    else:
        print(f"Title filter cleared for '{company['name']}'")


def _parse_job_ids(tokens: list[str]) -> list[int] | None:
    """
    Parse a list of ID tokens into a flat list of ints.
    Each token may be a single ID ("42") or a range ("42-50").
    Returns None if any token is invalid.
    """
    ids = []
    for token in tokens:
        if re.fullmatch(r'\d+', token):
            ids.append(int(token))
        elif re.fullmatch(r'(\d+)-(\d+)', token):
            start, end = token.split('-', 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            return None
    return ids


def cmd_refresh(run_brief: bool = True):
    """Run the full fetch → score → report pipeline."""
    print("Ensuring resume cache is up to date...")
    get_resume_text()
    print("Running pipeline...\n")
    run_pipeline(verbose=True, run_brief=run_brief)


def cmd_status():
    """Print a summary of the job + application pipeline."""
    conn = get_connection()

    totals = conn.execute("""
        SELECT
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN job_state='open' THEN 1 ELSE 0 END), 0) as open,
            COALESCE(SUM(CASE WHEN job_state='disappeared' THEN 1 ELSE 0 END), 0) as disappeared,
            COALESCE(SUM(CASE WHEN ai_score IS NULL AND job_state='open' THEN 1 ELSE 0 END), 0) as unscored,
            COALESCE(SUM(CASE WHEN ai_score >= 7 AND application_state='new' AND job_state='open' THEN 1 ELSE 0 END), 0) as pending_review
        FROM jobs
    """).fetchone()

    app_states = conn.execute("""
        SELECT application_state, COUNT(*) as n
        FROM jobs
        WHERE job_state = 'open'
        GROUP BY application_state
        ORDER BY n DESC
    """).fetchall()

    print("=== Job Pipeline ===")
    print(f"  Total jobs seen:      {totals['total']}")
    print(f"  Currently open:       {totals['open']}")
    print(f"  Disappeared/closed:   {totals['disappeared']}")
    print(f"  Unscored (open):      {totals['unscored']}")
    print(f"  High-score, new:      {totals['pending_review']}  ← worth reviewing")

    print("\n=== Application States (open jobs) ===")
    for row in app_states:
        print(f"  {row['application_state']:<20} {row['n']}")

    print("\n=== Top Companies by Open Jobs ===")
    rows = conn.execute("""
        SELECT c.name, COUNT(*) as n,
               ROUND(AVG(j.ai_score), 1) as avg_score
        FROM jobs j
        JOIN companies c ON j.company_id = c.id
        WHERE j.job_state = 'open'
        GROUP BY c.id
        ORDER BY n DESC
        LIMIT 10
    """).fetchall()
    for row in rows:
        score_str = f"avg score {row['avg_score']}" if row['avg_score'] else "unscored"
        print(f"  {row['name']:<30} {row['n']} open  ({score_str})")


def cmd_score(job_id: int):
    """Manually trigger AI scoring on a specific job."""
    conn = get_connection()
    job = conn.execute(
        "SELECT j.*, c.name as company_name FROM jobs j JOIN companies c ON j.company_id=c.id WHERE j.id=?",
        (job_id,)
    ).fetchone()

    if not job:
        print(f"No job found with id={job_id}")
        return

    print(f"Scoring: [{job_id}] {job['company_name']} — {job['title']}")
    resume_text = get_resume_text()
    prefs_text  = (Path(__file__).parent / "config" / "preferences.md").read_text()

    model = _get_model()
    provider = _resolve_provider(model, model_override=False)
    result = _score_one_job(job, resume_text, prefs_text, model, provider=provider, return_components=True)
    if isinstance(result, tuple) and len(result) == 3:
        score, rationale, components = result
    else:
        score, rationale = result
        components = None
    if rationale.lower().startswith("scoring failed:"):
        print(f"  Scoring failed, not updating DB: {rationale}")
        return
    update_job_score(job_id, score, rationale, model, PROMPT_VERSION, components=components)

    print(f"  Score:    {score}/10")
    print(f"  Rationale: {rationale}")


def cmd_rescore(job_ids: list[int] | None = None, model_override: str | None = None):
    """
    Rescore all jobs or a selected subset.
    - job_ids=None means rescore all jobs.
    - model_override selects the model for this run.
    """
    conn = get_connection()
    model = model_override or _get_model()
    provider = _resolve_provider(model, model_override=bool(model_override))
    resume_text = get_resume_text()
    prefs_path = Path(__file__).parent / "config" / "preferences.md"
    prefs_text = prefs_path.read_text() if prefs_path.exists() else ""

    if job_ids is None:
        jobs = conn.execute(
            """
            SELECT j.*, c.name as company_name
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            ORDER BY j.date_found DESC
            """
        ).fetchall()
    else:
        placeholders = ",".join("?" for _ in job_ids)
        jobs = conn.execute(
            f"""
            SELECT j.*, c.name as company_name
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.id IN ({placeholders})
            ORDER BY j.date_found DESC
            """,
            tuple(job_ids),
        ).fetchall()

    if not jobs:
        print("No matching jobs found to rescore.")
        return

    mode = "all (prefiltered)" if job_ids is None else "selected IDs"
    print(f"Rescoring {len(jobs)} job(s) using model: {model} ({provider}) [{mode}]")
    rescored = 0
    failed = 0
    skipped_prefilter = 0
    skipped_examples = []
    failed_examples = []
    for job in tqdm(jobs, desc="Rescoring", unit="job"):
        # For "rescore all", honor pre-filter eligibility and explicitly label skips.
        if job_ids is None:
            reasons = _prefilter_skip_reasons(job)
            if reasons:
                skipped_prefilter += 1
                label = "Skipped by pre-filter: " + ", ".join(reasons)
                mark_job_unscored(job["id"], label, model, PROMPT_VERSION)
                if len(skipped_examples) < 5:
                    skipped_examples.append((job["id"], job["title"], label))
                tqdm.write(f"  [{job['id']}] {job['company_name']} — {job['title']}: SKIPPED_PREFILTER")
                continue

        result = _score_one_job(job, resume_text, prefs_text, model, provider=provider, return_components=True)
        if isinstance(result, tuple) and len(result) == 3:
            score, rationale, components = result
        else:
            score, rationale = result
            components = None
        if rationale.lower().startswith("scoring failed:"):
            failed += 1
            if len(failed_examples) < 5:
                failed_examples.append((job["id"], job["title"], rationale))
            tqdm.write(f"  [{job['id']}] {job['company_name']} — {job['title']}: FAILED")
            continue

        update_job_score(job["id"], score, rationale, model, PROMPT_VERSION, components=components)
        rescored += 1
        tqdm.write(f"  [{job['id']}] {job['company_name']} — {job['title']}: {score}/10")

    print(f"Completed rescoring {rescored} job(s).")
    if skipped_prefilter:
        print(f"{skipped_prefilter} job(s) skipped by pre-filter and labeled as intentionally unscored.")
        for job_id, title, reason in skipped_examples:
            print(f"  [{job_id}] {title}: {reason}")
    if failed:
        print(f"{failed} job(s) failed to score and were left unchanged.")
        for job_id, title, reason in failed_examples:
            print(f"  [{job_id}] {title}: {reason}")


def cmd_apply(job_ids: list[int]):
    """Mark one or more jobs as applied."""
    _set_application_state_many(job_ids, "applied")


def cmd_skip(job_ids: list[int]):
    """Mark one or more jobs as skipped (reviewed, not applying)."""
    _set_application_state_many(job_ids, "skipped")


def cmd_review(job_ids: list[int]):
    """Mark one or more jobs as reviewed (seen, not yet decided)."""
    _set_application_state_many(job_ids, "reviewed")


def cmd_interview(job_ids: list[int]):
    """Mark one or more jobs as interviewing."""
    _set_application_state_many(job_ids, "interviewing")


def cmd_reject(job_ids: list[int]):
    """Mark one or more jobs as rejected."""
    _set_application_state_many(job_ids, "rejected")


def cmd_offer(job_ids: list[int]):
    """Mark one or more jobs as offer received."""
    _set_application_state_many(job_ids, "offer")


def cmd_reset(job_ids: list[int]):
    """Reset one or more jobs back to new (unreviewed)."""
    _set_application_state_many(job_ids, "new")


def cmd_jobs():
    """List recent high-scoring open jobs."""
    jobs = get_briefing_jobs(min_score=7.0)
    if not jobs:
        print("No high-scoring new jobs. Run 'python main.py refresh' to check for new postings.")
        return

    width = _term_width()
    company_w = max(16, min(30, int(width * 0.22)))
    title_w = max(24, min(54, int(width * 0.38)))
    location_w = max(12, min(26, int(width * 0.2)))

    rows = []
    for j in jobs:
        rows.append(
            {
                "ID": j["id"],
                "Score": f"{j['ai_score']:.0f}/10" if j["ai_score"] is not None else "—",
                "Company": j["company_name"],
                "Title": j["title"],
                "Location": j["location"] or "",
            }
        )

    print(tabulate(rows, headers="keys", tablefmt="simple", maxcolwidths=[6, 7, company_w, title_w, location_w]))


def cmd_jobs_all(company_filter: str | None = None):
    """List all jobs sorted by score (highest first, unscored last)."""
    conn = get_connection()
    if company_filter:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.job_state, j.application_state
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE LOWER(c.name) LIKE ?
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """,
            (f"%{company_filter.lower()}%",),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.job_state, j.application_state
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """
        ).fetchall()

    if not rows:
        msg = f"No jobs found for company matching '{company_filter}'." if company_filter else "No jobs found. Run 'python main.py refresh' after adding companies."
        print(msg)
        return

    width = _term_width()
    company_w = max(14, min(24, int(width * 0.19)))
    title_w = max(20, min(42, int(width * 0.3)))
    location_w = max(10, min(22, int(width * 0.16)))

    table_rows = []
    for r in rows:
        table_rows.append(
            {
                "ID": r["id"],
                "Score": f"{r['ai_score']:.1f}" if r["ai_score"] is not None else "—",
                "Company": r["company_name"],
                "Title": r["title"],
                "JobState": r["job_state"],
                "AppState": r["application_state"],
                "Location": r["location"] or "",
            }
        )

    print(
        tabulate(
            table_rows,
            headers="keys",
            tablefmt="simple",
            maxcolwidths=[6, 7, company_w, title_w, 12, 12, location_w],
        )
    )


def _cmd_jobs_by_state(app_state: str, company_filter: str | None = None):
    """List jobs for a specific application state."""
    conn = get_connection()
    if company_filter:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.date_found
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.application_state = ? AND LOWER(c.name) LIKE ?
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """,
            (app_state, f"%{company_filter.lower()}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.date_found
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.application_state = ?
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """,
            (app_state,),
        ).fetchall()

    if not rows:
        msg = f"No {app_state} jobs for company matching '{company_filter}'." if company_filter else f"No jobs marked as {app_state} yet."
        print(msg)
        return

    width = _term_width()
    company_w = max(14, min(26, int(width * 0.2)))
    title_w   = max(20, min(48, int(width * 0.36)))
    location_w = max(10, min(24, int(width * 0.18)))

    table_rows = []
    for r in rows:
        table_rows.append(
            {
                "ID":       r["id"],
                "Score":    f"{r['ai_score']:.1f}" if r["ai_score"] is not None else "—",
                "Company":  r["company_name"],
                "Title":    r["title"],
                "Location": r["location"] or "",
                "Found":    r["date_found"][:10] if r["date_found"] else "",
            }
        )

    print(
        tabulate(
            table_rows,
            headers="keys",
            tablefmt="simple",
            maxcolwidths=[6, 6, company_w, title_w, location_w, 10],
        )
    )


def cmd_jobs_pipeline():
    """List all jobs in the active application pipeline (applied/interviewing/rejected/offer)."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT j.id, c.name AS company_name, j.title, j.location,
               j.ai_score, j.application_state, j.job_state, j.date_found
        FROM jobs j
        JOIN companies c ON j.company_id = c.id
        WHERE j.application_state IN ('applied', 'interviewing', 'rejected', 'offer')
        ORDER BY
            CASE j.application_state
                WHEN 'offer'        THEN 1
                WHEN 'interviewing' THEN 2
                WHEN 'applied'      THEN 3
                WHEN 'rejected'     THEN 4
            END,
            j.date_found DESC
        """
    ).fetchall()

    if not rows:
        print("No jobs in the application pipeline yet.")
        return

    width = _term_width()
    company_w = max(14, min(24, int(width * 0.19)))
    title_w   = max(20, min(42, int(width * 0.3)))
    location_w = max(10, min(22, int(width * 0.16)))

    table_rows = []
    for r in rows:
        table_rows.append({
            "ID":       r["id"],
            "Status":   r["application_state"],
            "Score":    f"{r['ai_score']:.1f}" if r["ai_score"] is not None else "—",
            "Company":  r["company_name"],
            "Title":    r["title"],
            "Location": r["location"] or "",
            "Found":    r["date_found"][:10] if r["date_found"] else "",
        })

    print(tabulate(
        table_rows,
        headers="keys",
        tablefmt="simple",
        maxcolwidths=[6, 12, 7, company_w, title_w, location_w, 10],
    ))


def cmd_jobs_applied(company_filter: str | None = None):
    _cmd_jobs_by_state("applied", company_filter)


def cmd_jobs_skipped(company_filter: str | None = None):
    _cmd_jobs_by_state("skipped", company_filter)


def cmd_jobs_reviewed(company_filter: str | None = None):
    _cmd_jobs_by_state("reviewed", company_filter)


def cmd_jobs_interviewing(company_filter: str | None = None):
    _cmd_jobs_by_state("interviewing", company_filter)


def cmd_jobs_rejected(company_filter: str | None = None):
    _cmd_jobs_by_state("rejected", company_filter)


def cmd_jobs_offer(company_filter: str | None = None):
    _cmd_jobs_by_state("offer", company_filter)


def cmd_jobs_new(company_filter: str | None = None):
    _cmd_jobs_by_state("new", company_filter)


def cmd_jobs_disappeared(company_filter: str | None = None):
    """List jobs that have disappeared from job boards."""
    conn = get_connection()
    if company_filter:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.application_state, j.date_found
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.job_state = 'disappeared' AND LOWER(c.name) LIKE ?
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """,
            (f"%{company_filter.lower()}%",),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT j.id, c.name AS company_name, j.title, j.location,
                   j.ai_score, j.application_state, j.date_found
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.job_state = 'disappeared'
            ORDER BY (j.ai_score IS NULL), j.ai_score DESC, j.date_found DESC
            """
        ).fetchall()

    if not rows:
        msg = f"No disappeared jobs for company matching '{company_filter}'." if company_filter else "No disappeared jobs."
        print(msg)
        return

    width = _term_width()
    company_w  = max(14, min(26, int(width * 0.2)))
    title_w    = max(20, min(48, int(width * 0.36)))
    location_w = max(10, min(24, int(width * 0.18)))

    table_rows = []
    for r in rows:
        table_rows.append({
            "ID":       r["id"],
            "Score":    f"{r['ai_score']:.1f}" if r["ai_score"] is not None else "—",
            "Company":  r["company_name"],
            "Title":    r["title"],
            "AppState": r["application_state"],
            "Location": r["location"] or "",
            "Found":    r["date_found"][:10] if r["date_found"] else "",
        })

    print(tabulate(
        table_rows,
        headers="keys",
        tablefmt="simple",
        maxcolwidths=[6, 6, company_w, title_w, 12, location_w, 10],
    ))


def cmd_job(job_id: int, full: bool = False):
    """Show one job posting with detailed scoring context."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT j.*, c.name AS company_name
        FROM jobs j
        JOIN companies c ON j.company_id = c.id
        WHERE j.id = ?
        """,
        (job_id,),
    ).fetchone()

    if not row:
        print(f"No job found with id={job_id}")
        return

    print(f"=== Job {row['id']} ===")
    print(f"Company:            {row['company_name']}")
    print(f"Title:              {row['title']}")
    print(f"Location:           {row['location'] or 'Not specified'}")
    print(f"URL:                {row['url'] or 'Not available'}")
    print(f"Job state:          {row['job_state']}")
    print(f"Application state:  {row['application_state']}")
    print(f"Date found:         {row['date_found']}")
    print(f"Date posted:        {row['date_posted'] or 'Not provided'}")

    print("\n--- Scoring ---")
    print(f"Score:              {row['ai_score'] if row['ai_score'] is not None else 'Not scored'}")
    print(f"Model:              {row['model_name'] or 'N/A'}")
    print(f"Prompt version:     {row['prompt_version'] or 'N/A'}")
    print(f"Scored at:          {row['scored_at'] or 'N/A'}")

    rationale, adjustments = _extract_adjustments(row["ai_rationale"])
    if rationale:
        print("Rationale:")
        print(textwrap.fill(rationale, width=max(80, _term_width() - 2)))
    else:
        print("Rationale:          N/A")
    if adjustments:
        print(f"Adjustments:        {adjustments}")

    if row["ai_components"]:
        print("Components:")
        try:
            comp = json.loads(row["ai_components"])
            fields = [
                ("industry_fit", "Industry fit"),
                ("role_fit", "Role fit"),
                ("location_fit", "Location fit"),
                ("level_fit", "Level fit"),
                ("exclude_risk", "Exclude risk"),
                ("raw_score", "Raw model score"),
                ("component_score", "Weighted component score"),
                ("pre_guardrail_score", "Pre-guardrail score"),
                ("final_score", "Final score"),
            ]
            for key, label in fields:
                if comp.get(key) is not None:
                    print(f"  {label:<26} {comp[key]}")
            if comp.get("adjustments"):
                print(f"  {'adjustments':<26} {', '.join(comp['adjustments'])}")
        except Exception:
            print("  (Could not parse component score JSON)")

    print("\n--- Filter Flags ---")
    print(f"Geo match:          {row['geo_match']}")
    print(f"Title excluded:     {row['title_excluded']}")
    print(f"Exclude flag:       {row['exclude_flag']}")

    description = row["description"] or ""
    if not description:
        print("\nDescription:        Not available")
        return

    print("\n--- Description ---")
    if full:
        print(textwrap.fill(description, width=max(80, _term_width() - 2)))
    else:
        preview = description[:1200]
        print(textwrap.fill(preview, width=max(80, _term_width() - 2)))
        if len(description) > 1200:
            print("\n(Truncated. Re-run with --full to view complete description.)")


def cmd_doctor(model_override: str | None = None):
    """Run preflight checks for model/runtime/config readiness."""
    model = model_override or _get_model()
    provider = _resolve_provider(model, model_override=bool(model_override))
    print("=== Job Tracker Doctor ===")
    print(f"Selected model: {model}")
    print(f"Provider: {provider}")
    print(f"OPENAI_API_KEY set: {'yes' if bool(os.getenv('OPENAI_API_KEY')) else 'no'}")
    print(f"OLLAMA_API_KEY set: {'yes' if bool(os.getenv('OLLAMA_API_KEY')) else 'no'}")

    issues = []

    has_resume = DOCX_PATH.exists() or PDF_PATH.exists()
    if not has_resume:
        issues.append(f"Missing resume file: add {DOCX_PATH.name} or {PDF_PATH.name} under config/.")

    prefs_md = Path(__file__).parent / "config" / "preferences.md"
    if not prefs_md.exists():
        issues.append("Missing config/preferences.md (recommended for better scoring quality).")

    if CACHE_PATH.exists():
        print(f"Resume cache: found ({CACHE_PATH.name})")
    else:
        print("Resume cache: not found yet (created on first scoring run)")

    if provider == "openai":
        try:
            import openai  # noqa: F401
        except Exception:
            issues.append("Python package 'openai' is not installed. Run: uv pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY is not set for selected OpenAI model.")
    elif provider == "ollama":
        if not os.getenv("OLLAMA_API_KEY"):
            issues.append("OLLAMA_API_KEY is not set for selected Ollama cloud model.")
        else:
            print("Using Ollama Cloud mode.")
    else:
        issues.append("Unsupported provider in config. Use 'openai' or 'ollama'.")

    if issues:
        print("\nIssues found:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo blocking issues found.")


def cmd_ask(question: str):
    """Ask a natural-language question and retrieve relevant jobs via tool-calling."""
    q = (question or "").strip()
    if not q:
        print("Please provide a question.")
        return

    resume_note = ""
    try:
        resume_text = get_resume_text()
        resume_text = " ".join((resume_text or "").split())
        resume_excerpt = resume_text[:6000] if resume_text else ""
        if resume_excerpt:
            resume_note = (
                "Candidate resume context (use this to tailor fit assessments):\n"
                f"{resume_excerpt}\n\n"
            )
        else:
            resume_note = "Candidate resume context is currently empty.\n\n"
    except Exception as e:
        resume_note = (
            "Candidate resume context is unavailable. "
            f"Reason: {e}\n\n"
        )

    role = (
        "You are a thoughtful job-search advisor helping a candidate make good decisions — not "
        "a recruiter trying to maximize applications. Use the available tools to look up relevant "
        "postings from the candidate's tracked company database. Jobs have AI-generated fit scores "
        "(0–10) based on the candidate's profile. Use the candidate's resume to give specific, "
        "personalized answers.\n\n"
        "Always check application_state — do not recommend applying to jobs already marked "
        "'applied', 'interviewing', 'offer', or 'rejected'. Apply the same judgment to roles "
        "at companies where the candidate is already pursuing something stronger.\n\n"
        "To read a job description, use get_job_details with the job ID — it returns the full "
        "description text along with score, rationale, and application state.\n\n"
        "Make tool calls immediately and directly. Do not narrate what you are about to do — "
        "just do it. You may call tools multiple times in sequence if needed.\n\n"
        "Be concise and direct. Answer the question asked. A few clear sentences is better than "
        "a formatted report. Only surface roles genuinely worth the candidate's attention — "
        "it is fine and often correct to recommend against applying to most things."
    )
    task = resume_note + f"Question: {q}"

    model = _get_ask_model()
    provider = _get_ask_provider()
    reply = agent_loop(
        role=role,
        task=task,
        model=model,
        provider=provider,
        tools=[tool_search_jobs, tool_get_job_details, tool_get_briefing_data, tool_get_pipeline_data],
        tool_funcs={
            "search_jobs": search_jobs,
            "get_job_details": get_job_details,
            "get_briefing_data": get_briefing_data,
            "get_pipeline_data": get_pipeline_data,
        },
    )
    print(reply)


def _set_application_state(job_id: int, state: str) -> bool:
    conn = get_connection()
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        print(f"No job found with id={job_id}")
        return False
    update_job_states(job_id, application_state=state)
    print(f"Job {job_id} marked as '{state}'.")
    return True


def _set_application_state_many(job_ids: list[int], state: str):
    updated = 0
    missing = 0
    for jid in job_ids:
        ok = _set_application_state(jid, state)
        if ok:
            updated += 1
        else:
            missing += 1
    if len(job_ids) > 1:
        print(f"Done: {updated} updated, {missing} missing.")


def main():
    init_db()

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "add":
        if len(args) < 2:
            print("Usage: python main.py add <url> [--ats-type TYPE] [--ats-id SLUG]")
            return
        add_args = args[1:]
        ats_type_override = None
        ats_id_override = None
        title_filter = None
        for flag, target in [("--ats-type", "ats_type"), ("--ats-id", "ats_id"), ("--title-filter", "title_filter")]:
            if flag in add_args:
                idx = add_args.index(flag)
                if idx + 1 < len(add_args):
                    val = add_args[idx + 1]
                    if target == "ats_type":
                        ats_type_override = val
                    elif target == "ats_id":
                        ats_id_override = val
                    elif target == "title_filter":
                        title_filter = [k.strip() for k in val.split(",") if k.strip()]
                    del add_args[idx:idx + 2]
        cmd_add(add_args[0], ats_type_override=ats_type_override, ats_id_override=ats_id_override, title_filter=title_filter)

    elif cmd == "add-file":
        if len(args) < 2:
            print("Usage: python main.py add-file <path>")
            return
        cmd_add_file(args[1])

    elif cmd == "export-companies":
        if len(args) < 2:
            print("Usage: python main.py export-companies <path>")
            return
        cmd_export_companies(args[1])

    elif cmd == "list":
        cmd_list()

    elif cmd == "refresh":
        cmd_refresh(run_brief="--no-brief" not in args)

    elif cmd == "status":
        cmd_status()

    elif cmd == "score":
        if len(args) < 2 or not args[1].isdigit():
            print("Usage: python main.py score <job_id>")
            return
        cmd_score(int(args[1]))

    elif cmd == "rescore":
        if len(args) < 2:
            print("Usage: python main.py rescore all [--model MODEL]")
            print("   or: python main.py rescore <job_id> [<job_id> ...] [--model MODEL]")
            return

        model_override = None
        rescore_args = args[1:]
        if "--model" in rescore_args:
            idx = rescore_args.index("--model")
            if idx + 1 >= len(rescore_args):
                print("Usage: --model requires a value")
                return
            model_override = rescore_args[idx + 1].strip()
            if not model_override:
                print("Usage: --model requires a value")
                return
            del rescore_args[idx:idx + 2]

        if len(rescore_args) == 1 and rescore_args[0].lower() == "all":
            cmd_rescore(job_ids=None, model_override=model_override)
            return

        ids = _parse_job_ids(rescore_args)
        if ids is None:
            print("Usage: python main.py rescore all [--model MODEL]")
            print("   or: python main.py rescore <job_id|range> ... [--model MODEL]")
            print("   e.g: python main.py rescore 42 43 50-60")
            return
        cmd_rescore(job_ids=ids, model_override=model_override)

    elif cmd in ("apply", "skip", "review", "interview", "reject", "offer", "reset"):
        ids = _parse_job_ids(args[1:]) if len(args) >= 2 else None
        if not ids:
            print(f"Usage: python main.py {cmd} <job_id|range> ...")
            print(f"   e.g: python main.py {cmd} 42 50-60")
            return
        {
            "apply":     cmd_apply,
            "skip":      cmd_skip,
            "review":    cmd_review,
            "interview": cmd_interview,
            "reject":    cmd_reject,
            "offer":     cmd_offer,
            "reset":     cmd_reset,
        }[cmd](ids)

    elif cmd in ("jobs", "jobs-all", "jobs-new", "jobs-reviewed", "jobs-pipeline",
                 "jobs-applied", "jobs-skipped", "jobs-interviewing", "jobs-rejected",
                 "jobs-offer", "jobs-disappeared"):
        company_filter = None
        if "--company" in args:
            idx = args.index("--company")
            if idx + 1 < len(args):
                company_filter = args[idx + 1]
        if cmd == "jobs":
            cmd_jobs()
        elif cmd == "jobs-all":
            cmd_jobs_all(company_filter=company_filter)
        elif cmd == "jobs-new":
            cmd_jobs_new(company_filter=company_filter)
        elif cmd == "jobs-reviewed":
            cmd_jobs_reviewed(company_filter=company_filter)
        elif cmd == "jobs-pipeline":
            cmd_jobs_pipeline()
        elif cmd == "jobs-applied":
            cmd_jobs_applied(company_filter=company_filter)
        elif cmd == "jobs-skipped":
            cmd_jobs_skipped(company_filter=company_filter)
        elif cmd == "jobs-interviewing":
            cmd_jobs_interviewing(company_filter=company_filter)
        elif cmd == "jobs-rejected":
            cmd_jobs_rejected(company_filter=company_filter)
        elif cmd == "jobs-offer":
            cmd_jobs_offer(company_filter=company_filter)
        elif cmd == "jobs-disappeared":
            cmd_jobs_disappeared(company_filter=company_filter)

    elif cmd == "job":
        if len(args) < 2 or not args[1].isdigit():
            print("Usage: python main.py job <job_id> [--full]")
            return
        full = "--full" in args[2:]
        cmd_job(int(args[1]), full=full)

    elif cmd == "ask":
        if len(args) < 2:
            print("Usage: python main.py ask <question>")
            return
        cmd_ask(" ".join(args[1:]))

    elif cmd == "doctor":
        model_override = None
        doctor_args = args[1:]
        if "--model" in doctor_args:
            idx = doctor_args.index("--model")
            if idx + 1 >= len(doctor_args):
                print("Usage: python main.py doctor [--model MODEL]")
                return
            model_override = doctor_args[idx + 1].strip()
            if not model_override:
                print("Usage: python main.py doctor [--model MODEL]")
                return
        cmd_doctor(model_override=model_override)

    elif cmd == "set-ats":
        if len(args) < 3 or not args[1].isdigit():
            print("Usage: python main.py set-ats <company_id> <ats_type> [<ats_id>]")
            return
        ats_id = args[3] if len(args) >= 4 else None
        cmd_set_ats(int(args[1]), args[2], ats_id)

    elif cmd == "set-url":
        if len(args) < 3 or not args[1].isdigit():
            print("Usage: python main.py set-url <company_id> <url>")
            return
        cmd_set_url(int(args[1]), args[2])

    elif cmd == "set-title-filter":
        if len(args) < 2 or not args[1].isdigit():
            print("Usage: python main.py set-title-filter <company_id> \"kw1,kw2,...\"")
            print("   or: python main.py set-title-filter <company_id>  (clears filter)")
            return
        company_id = int(args[1])
        patterns = None
        if len(args) >= 3:
            patterns = [k.strip() for k in args[2].split(",") if k.strip()]
        cmd_set_title_filter(company_id, patterns)

    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)


if __name__ == "__main__":
    main()
