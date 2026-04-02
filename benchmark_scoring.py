#!/usr/bin/env python3
"""
benchmark_scoring.py

Score a fixed set of known jobs and compare against expected ranges.
Use this to quickly evaluate prompt/weight changes without re-running the full pipeline.

Results are always saved to benchmark_results/ with a timestamp so you can compare runs.

Usage:
    uv run benchmark_scoring.py              # score all test cases, save results
    uv run benchmark_scoring.py --verbose    # also include component breakdown + rationale
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db import database as db
from agents.tools import _score_one_job, _get_model, _get_provider, PREFS_MD, PROMPT_VERSION
from utils.resume_parser import get_resume_text

# ---------------------------------------------------------------------------
# Test cases: (job_id, label, expected_min, expected_max, notes)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # Strong fits — applied roles, direct systems engineering match
    (99,  "Oklo — Systems Engineer, Req & Integration",   8.5, 10.0, "applied; perfect role match"),
    (129, "Thea Energy — Systems Engineer",               7.5, 10.0, "applied; NJ = NYC metro"),

    # Moderate-good — TPM at dream company, but too senior for current experience level
    (20,  "CFS — Principal R&D TPM",                     4.5,  7.0, "TPM role fit, but Principal = likely 10+ yrs req"),
    (40,  "CFS — Senior TPM",                            5.0,  7.5, "TPM role fit, Senior = stretch on seniority"),

    # Moderate — great company, wrong role type (specialist disciplines)
    # Key calibration cases: high industry_fit should NOT inflate role_fit
    (41,  "CFS — Shift Operations Engineer",             3.5,  5.5, "ops role, not systems eng"),
    (28,  "CFS — Senior Controls Engineer",              3.5,  5.5, "specialist discipline"),
    (37,  "CFS — Senior Multiphysics Modeler",           3.0,  5.5, "research specialist"),
    (11,  "CFS — Lead, Field Mechanical Engineering",    4.0,  6.0, "mechanical, not systems"),
    (76,  "Oklo — Operations Engineer",                  5.0,  7.5, "ops-adjacent; ConOps/systems integration language"),

    # Poor fits — finance, geo mismatch, intern
    (30,  "CFS — Senior Financial Analyst (Milpitas)",   0.0,  3.0, "finance + geo mismatch"),
    (43,  "CFS — VP of Financial Planning",              0.0,  4.0, "finance role"),
    (10,  "CFS — Intern, Cryogenics",                    0.0,  4.0, "intern"),
]

PASS_SYM  = "✓"
FAIL_SYM  = "✗"
WARN_SYM  = "~"


def within_range(score, lo, hi):
    return lo <= score <= hi


def near_range(score, lo, hi, tolerance=0.5):
    return (lo - tolerance) <= score <= (hi + tolerance)


def run_benchmark(verbose: bool = False):
    model    = _get_model()
    provider = _get_provider()
    resume   = get_resume_text()
    prefs    = PREFS_MD.read_text(encoding="utf-8") if PREFS_MD.exists() else ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out_path  = out_dir / f"benchmark_{PROMPT_VERSION}_{timestamp}.txt"

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    WIDTH = 104
    emit(f"\nBenchmark — prompt {PROMPT_VERSION} — model: {model} ({provider}) — {timestamp}")
    emit("=" * WIDTH)
    emit(f"  {'':2}  {'ID':>4}  {'Score':>6}  {'Expected':>10}  {'Label':<48}  Notes")
    emit("-" * WIDTH)

    passed = failed = near_miss = 0

    for job_id, label, lo, hi, notes in TEST_CASES:
        conn = db.get_connection()
        row = conn.execute(
            """
            SELECT j.*, c.name as company_name
            FROM jobs j JOIN companies c ON c.id = j.company_id
            WHERE j.id = ?
            """,
            (job_id,),
        ).fetchone()

        if row is None:
            emit(f"  ?  {job_id:>4}  {'N/A':>6}  {f'{lo:.1f}–{hi:.1f}':>10}  {label:<48}  job not found in DB")
            failed += 1
            continue

        score, rationale, components = _score_one_job(
            row, resume, prefs, model, provider=provider, return_components=True
        )

        if within_range(score, lo, hi):
            sym = PASS_SYM
            passed += 1
        elif near_range(score, lo, hi):
            sym = WARN_SYM
            near_miss += 1
        else:
            sym = FAIL_SYM
            failed += 1

        range_str = f"{lo:.1f}–{hi:.1f}"
        emit(f"  {sym}  {job_id:>4}  {score:>6.1f}  {range_str:>10}  {label:<48}  {notes}")

        if verbose and components:
            ind  = components.get("industry_fit", "?")
            role = components.get("role_fit", "?")
            loc  = components.get("location_fit", "?")
            lvl  = components.get("level_fit", "?")
            exc  = components.get("exclude_risk", "?")
            adj  = components.get("adjustments") or []
            emit(f"       components: industry={ind}  role={role}  location={loc}  level={lvl}  exclude_risk={exc}")
            if adj:
                emit(f"       guardrails: {', '.join(adj)}")
            emit(f"       rationale:  {rationale}")
            emit()

    emit("-" * WIDTH)
    total = len(TEST_CASES)
    emit(f"Results: {passed}/{total} passed  {near_miss} near-miss  {failed} failed")
    emit()

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score benchmark jobs and compare to expected ranges.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show component breakdown and rationale")
    args = parser.parse_args()
    run_benchmark(verbose=args.verbose)
