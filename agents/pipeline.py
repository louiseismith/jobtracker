"""
agents/pipeline.py

Pipeline: fetch → score → report

Reference patterns:
  dsai/08_function_calling/04_multiple_agents_with_function_calling.py
  hackathon/app/chatbot/agent.py
"""

import re
import yaml
from pathlib import Path
from tqdm import tqdm

from db.database import get_active_companies
from agents.functions import agent_run
from agents.tools import (
    check_new_jobs,
    score_all_new_jobs,   tool_score_all_new_jobs,
    get_briefing_data,    tool_get_briefing_data,
    get_pipeline_data,    tool_get_pipeline_data,
)
from utils.resume_parser import get_resume_text

CONFIG_DIR = Path(__file__).parent.parent / "config"
PREFS_PATH = CONFIG_DIR / "preferences.yaml"
PREFS_MD   = CONFIG_DIR / "preferences.md"


def _load_scorer_config() -> tuple[str, str]:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    model    = cfg.get("model", "gpt-4.1")
    provider = cfg.get("provider", "openai")
    return model, str(provider).strip().lower()


def _load_reporter_config() -> tuple[str, str]:
    cfg = yaml.safe_load(PREFS_PATH.read_text())
    model    = cfg.get("reporter_model", cfg.get("model", "gpt-4.1"))
    provider = cfg.get("reporter_provider", cfg.get("provider", "openai"))
    return model, str(provider).strip().lower()



def run_pipeline(verbose: bool = True, run_brief: bool = True) -> str:
    """
    Run the full fetch → score → report pipeline.

    Returns the Reporter's plain-English briefing.
    """
    companies = get_active_companies()
    if not companies:
        return "No active companies to monitor. Add some with: python main.py add <url>"

    # ------------------------------------------------------------------
    # Step 1 — Fetcher
    # ------------------------------------------------------------------
    if verbose:
        print("Fetcher: checking job boards...")

    fetch_summaries = []
    skipped_names = []
    total_new = 0

    iterable = tqdm(companies, desc="  Fetching", unit="co") if verbose else companies
    for company in iterable:
        result = check_new_jobs(company["name"])
        fetch_summaries.append(result)

        if result.startswith("Skipped"):
            skipped_names.append(company["name"])
        elif verbose and ("Fetch failed" in result or "not found" in result or "not available" in result):
            tqdm.write(f"  ERROR — {result}")
        elif verbose:
            m = re.search(r"(\d+) new", result)
            new_count = int(m.group(1)) if m else 0
            total_new += new_count
            if new_count > 0:
                tqdm.write(f"  {result}")

    if verbose:
        if skipped_names:
            print(f"  Skipped {len(skipped_names)} (no fetcher): {', '.join(skipped_names)}")
        checked = len(companies) - len(skipped_names)
        print(f"  → {checked} boards checked, {total_new} new job(s) found")

    fetch_result = "\n".join(fetch_summaries)

    # ------------------------------------------------------------------
    # Step 2 — Scorer
    # ------------------------------------------------------------------
    scorer_model, scorer_provider = _load_scorer_config()

    if verbose:
        print("\nScorer: scoring new jobs...")

    score_result = agent_run(
        role="You are a job fit scorer. Call score_all_new_jobs to evaluate all new unscored job postings against the candidate's resume and preferences.",
        task=fetch_result,
        tools=[tool_score_all_new_jobs],
        tool_funcs={"score_all_new_jobs": score_all_new_jobs},
        output="text",
        model=scorer_model,
        provider=scorer_provider,
    )

    if verbose:
        print(score_result)

    # ------------------------------------------------------------------
    # Step 3 — Reporter
    # ------------------------------------------------------------------
    if not run_brief:
        if verbose:
            print("\nReporter: skipped (--no-brief).")
        return score_result

    reporter_model, reporter_provider = _load_reporter_config()
    resume_text = get_resume_text()
    prefs_text  = PREFS_MD.read_text(encoding="utf-8") if PREFS_MD.exists() else ""

    role = (
        "You are a job search assistant helping a candidate review their job tracker. "
        "The candidate's resume and preferences are provided below as context — use them "
        "to personalize your commentary.\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        f"--- PREFERENCES ---\n{prefs_text}"
    )

    task = (
        "Call get_briefing_data to retrieve all high-scoring unreviewed job postings. "
        "Also call get_pipeline_data to see what's already in flight and at what scores.\n\n"
        "Then write a briefing with two sections:\n\n"
        "**New this run** — jobs whose titles appear in the scoring summary below. "
        "For each, one sentence on the strongest fit signal (role function or company domain, "
        "whichever is more compelling). Include score and URL.\n\n"
        "**Still in your queue** — high-scoring jobs already in the database that are still "
        "unreviewed (date_found earlier than today's scored jobs). Call out anything that has "
        "been sitting a while or stands out as easy to overlook. Same format.\n\n"
        "For both sections: use the pipeline data to give the best advice. Factor in the scores "
        "of existing applications when deciding whether to surface additional roles from the same "
        "company — a strong new role at a company where there's already a weak application in "
        "flight is worth flagging; a middling role where something stronger is already in process "
        "probably isn't.\n\n"
        "If a section is empty, say so in one line. No filler phrases.\n\n"
        f"--- SCORING SUMMARY (this run) ---\n{score_result}"
    )

    if verbose:
        print("\nReporter: writing briefing...")

    briefing = agent_run(
        role=role,
        task=task,
        tools=[tool_get_briefing_data, tool_get_pipeline_data],
        tool_funcs={"get_briefing_data": get_briefing_data, "get_pipeline_data": get_pipeline_data},
        output="text",
        model=reporter_model,
        provider=reporter_provider,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("BRIEFING")
        print("=" * 60)
        print(briefing)

    return briefing
