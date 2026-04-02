"""
tests/test_cli.py

CLI smoke tests for main.py.
Run with: .venv/bin/python -m pytest tests/test_cli.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import db.database as db_module

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
TEST_DB_PATH = Path(_tmp.name)
db_module.DB_PATH = TEST_DB_PATH

import main as cli
from db.database import init_db, upsert_company, upsert_job, get_connection, update_job_score


@pytest.fixture(autouse=True)
def fresh_db():
    db_module.DB_PATH = TEST_DB_PATH
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db()
    yield
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


def _run_cli(args: list[str]):
    with patch.object(sys, "argv", ["main.py", *args]):
        cli.main()


def test_jobs_command_when_none(capsys):
    _run_cli(["jobs"])
    out = capsys.readouterr().out
    assert "No high-scoring new jobs" in out


def test_apply_command_updates_state(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    job_id, _ = upsert_job(company_id, "lv-1", "Systems Engineer", None, None, "NYC", None)

    _run_cli(["apply", str(job_id)])
    out = capsys.readouterr().out
    assert "marked as 'applied'" in out

    row = get_connection().execute(
        "SELECT application_state FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    assert row["application_state"] == "applied"


def test_skip_command_updates_state(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    job_id, _ = upsert_job(company_id, "lv-2", "TPM", None, None, "Boston", None)

    _run_cli(["skip", str(job_id)])
    out = capsys.readouterr().out
    assert "marked as 'skipped'" in out

    row = get_connection().execute(
        "SELECT application_state FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    assert row["application_state"] == "skipped"


def test_apply_multiple_jobs_updates_all(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-11", "Role A", None, None, "NYC", None)
    j2, _ = upsert_job(company_id, "lv-12", "Role B", None, None, "NYC", None)

    _run_cli(["apply", str(j1), str(j2)])
    out = capsys.readouterr().out
    assert "Done: 2 updated, 0 missing." in out

    r1 = get_connection().execute("SELECT application_state FROM jobs WHERE id=?", (j1,)).fetchone()
    r2 = get_connection().execute("SELECT application_state FROM jobs WHERE id=?", (j2,)).fetchone()
    assert r1["application_state"] == "applied"
    assert r2["application_state"] == "applied"


def test_skip_multiple_jobs_with_missing_id_reports_summary(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-21", "Role A", None, None, "NYC", None)
    missing = 99999

    _run_cli(["skip", str(j1), str(missing)])
    out = capsys.readouterr().out
    assert "No job found with id=99999" in out
    assert "Done: 1 updated, 1 missing." in out

    r1 = get_connection().execute("SELECT application_state FROM jobs WHERE id=?", (j1,)).fetchone()
    assert r1["application_state"] == "skipped"


def test_score_usage_for_invalid_id(capsys):
    _run_cli(["score", "abc"])
    out = capsys.readouterr().out
    assert "Usage: python main.py score <job_id>" in out


def test_apply_usage_for_invalid_id(capsys):
    _run_cli(["apply", "abc"])
    out = capsys.readouterr().out
    assert "Usage: python main.py apply <job_id|range> ..." in out


def test_skip_usage_for_invalid_id(capsys):
    _run_cli(["skip", "abc"])
    out = capsys.readouterr().out
    assert "Usage: python main.py skip <job_id|range> ..." in out


def test_unknown_command_prints_help(capsys):
    _run_cli(["unknown-cmd"])
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "python main.py refresh" in out


def test_jobs_all_empty(capsys):
    _run_cli(["jobs-all"])
    out = capsys.readouterr().out
    assert "No jobs found" in out


def test_jobs_applied_empty(capsys):
    _run_cli(["jobs-applied"])
    out = capsys.readouterr().out
    assert "No jobs marked as applied yet." in out


def test_jobs_skipped_empty(capsys):
    _run_cli(["jobs-skipped"])
    out = capsys.readouterr().out
    assert "No jobs marked as skipped yet." in out


def test_jobs_all_sorted_by_score(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-1", "Role A", None, None, "NYC", None)
    j2, _ = upsert_job(company_id, "lv-2", "Role B", None, None, "NYC", None)
    j3, _ = upsert_job(company_id, "lv-3", "Role C", None, None, "NYC", None)

    update_job_score(j1, 6.0, "ok", "llama3.2", "v1")
    update_job_score(j2, 9.0, "great", "llama3.2", "v1")
    # j3 intentionally left unscored

    _run_cli(["jobs-all"])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]

    role_b_idx = next(i for i, line in enumerate(lines) if "Role B" in line)
    role_a_idx = next(i for i, line in enumerate(lines) if "Role A" in line)
    role_c_idx = next(i for i, line in enumerate(lines) if "Role C" in line)

    assert role_b_idx < role_a_idx < role_c_idx


def test_rescore_all_updates_scores(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-1", "Role A", None, None, "NYC", None)
    j2, _ = upsert_job(company_id, "lv-2", "Role B", None, None, "NYC", None)
    update_job_score(j1, 2.0, "old", "llama3.2", "v1")

    with patch("main._score_one_job", return_value=(8.5, "rescored")):
        _run_cli(["rescore", "all", "--model", "gpt-4o-mini"])

    out = capsys.readouterr().out
    assert "Rescoring 2 job(s) using model: gpt-4o-mini (openai) [all (prefiltered)]" in out
    rows = get_connection().execute(
        "SELECT id, ai_score, model_name FROM jobs ORDER BY id"
    ).fetchall()
    assert rows[0]["id"] == j1 and rows[0]["ai_score"] == 8.5 and rows[0]["model_name"] == "gpt-4o-mini"
    assert rows[1]["id"] == j2 and rows[1]["ai_score"] == 8.5 and rows[1]["model_name"] == "gpt-4o-mini"


def test_jobs_applied_and_jobs_skipped_split_views(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j_applied, _ = upsert_job(company_id, "lv-a", "Applied Role", None, None, "NYC", None)
    j_skipped, _ = upsert_job(company_id, "lv-s", "Skipped Role", None, None, "Boston", None)

    _run_cli(["apply", str(j_applied)])
    _ = capsys.readouterr()
    _run_cli(["skip", str(j_skipped)])
    _ = capsys.readouterr()

    _run_cli(["jobs-applied"])
    out_applied = capsys.readouterr().out
    assert "Applied Role" in out_applied
    assert "Skipped Role" not in out_applied

    _run_cli(["jobs-skipped"])
    out_skipped = capsys.readouterr().out
    assert "Skipped Role" in out_skipped
    assert "Applied Role" not in out_skipped


def test_rescore_all_respects_prefilter_and_labels_skipped(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    eligible_id, _ = upsert_job(
        company_id, "lv-1", "Systems Engineer", None, None, "NYC", None,
        exclude_flag=0, geo_match=1, title_excluded=0
    )
    filtered_id, _ = upsert_job(
        company_id, "lv-2", "Marketing Manager", None, None, "Austin, TX", None,
        exclude_flag=0, geo_match=0, title_excluded=1
    )

    scorer = patch("main._score_one_job", return_value=(8.0, "rescored"))
    with scorer as mocked:
        _run_cli(["rescore", "all", "--model", "gpt-4o-mini"])

    out = capsys.readouterr().out
    assert "SKIPPED_PREFILTER" in out
    assert mocked.call_count == 1

    row_ok = get_connection().execute(
        "SELECT ai_score FROM jobs WHERE id=?",
        (eligible_id,),
    ).fetchone()
    row_skip = get_connection().execute(
        "SELECT ai_score, ai_rationale FROM jobs WHERE id=?",
        (filtered_id,),
    ).fetchone()
    assert row_ok["ai_score"] == 8.0
    assert row_skip["ai_score"] is None
    assert row_skip["ai_rationale"].startswith("Skipped by pre-filter:")


def test_rescore_subset_only_updates_selected(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-1", "Role A", None, None, "NYC", None)
    j2, _ = upsert_job(company_id, "lv-2", "Role B", None, None, "NYC", None)
    update_job_score(j1, 3.0, "old", "llama3.2", "v1")
    update_job_score(j2, 4.0, "old", "llama3.2", "v1")

    with patch("main._score_one_job", return_value=(9.0, "rescored")):
        _run_cli(["rescore", str(j1)])

    out = capsys.readouterr().out
    assert "Rescoring 1 job(s)" in out
    row1 = get_connection().execute("SELECT ai_score FROM jobs WHERE id=?", (j1,)).fetchone()
    row2 = get_connection().execute("SELECT ai_score FROM jobs WHERE id=?", (j2,)).fetchone()
    assert row1["ai_score"] == 9.0
    assert row2["ai_score"] == 4.0


def test_rescore_failure_does_not_overwrite_existing_score(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    j1, _ = upsert_job(company_id, "lv-1", "Role A", None, None, "NYC", None)
    update_job_score(j1, 7.0, "previous", "llama3.2", "v1")

    with patch("main._score_one_job", return_value=(0.0, "Scoring failed: bad response format")):
        _run_cli(["rescore", str(j1), "--model", "gpt-4o"])

    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "left unchanged" in out

    row = get_connection().execute(
        "SELECT ai_score, ai_rationale, model_name FROM jobs WHERE id=?",
        (j1,),
    ).fetchone()
    assert row["ai_score"] == 7.0
    assert row["ai_rationale"] == "previous"
    assert row["model_name"] == "llama3.2"


def test_doctor_openai_model_requires_api_key(capsys):
    with patch.dict("os.environ", {}, clear=True):
        _run_cli(["doctor", "--model", "gpt-4o"])
    out = capsys.readouterr().out
    assert "Selected model: gpt-4o" in out
    assert "Provider: openai" in out
    assert "OPENAI_API_KEY is not set" in out


def test_doctor_reports_missing_ollama_cloud_key(capsys):
    with patch.dict("os.environ", {}, clear=True):
        _run_cli(["doctor", "--model", "llama3.2"])
    out = capsys.readouterr().out
    assert "Selected model: llama3.2" in out
    assert "Provider: ollama" in out
    assert "OLLAMA_API_KEY is not set" in out


def test_job_command_shows_detailed_fields(capsys):
    company_id = upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    job_id, _ = upsert_job(
        company_id,
        "lv-1",
        "Systems Engineer",
        "https://example.com/jobs/1",
        "A detailed description for testing output.",
        "NYC",
        None,
        exclude_flag=0,
        geo_match=1,
        title_excluded=0,
    )
    update_job_score(
        job_id,
        8.5,
        "Strong fit. [adjustments: systems_bonus]",
        "gpt-4o",
        "v2",
        components={
            "industry_fit": 9,
            "role_fit": 8,
            "location_fit": 8,
            "level_fit": 8,
            "exclude_risk": 9,
            "raw_score": 8.5,
            "component_score": 8.4,
            "pre_guardrail_score": 8.4,
            "final_score": 8.5,
            "adjustments": ["systems_bonus"],
        },
    )

    _run_cli(["job", str(job_id)])
    out = capsys.readouterr().out
    assert "=== Job" in out
    assert "Systems Engineer" in out
    assert "Score:" in out
    assert "gpt-4o" in out
    assert "Adjustments:" in out
    assert "systems_bonus" in out
    assert "Industry fit" in out
    assert "Role fit" in out
    assert "Final score" in out


def test_add_file_imports_supported_line_formats(tmp_path, capsys):
    import_file = tmp_path / "companies.txt"
    import_file.write_text(
        "\n".join(
            [
                "# comment",
                "Acme Lever | https://jobs.lever.co/acme",
                "Fusion Labs, https://boards.greenhouse.io/fusionlabs",
                "https://jobs.lever.co/urlonlyco",
            ]
        ),
        encoding="utf-8",
    )

    def fake_detect(url):
        if "greenhouse" in url:
            return "greenhouse", "fusionlabs"
        if "urlonlyco" in url:
            return "lever", "urlonlyco"
        return "lever", "acme"

    with patch("main.detect", side_effect=fake_detect):
        _run_cli(["add-file", str(import_file)])

    out = capsys.readouterr().out
    assert "Import complete:" in out
    rows = get_connection().execute("SELECT name, careers_url FROM companies ORDER BY name").fetchall()
    names = [r["name"] for r in rows]
    assert "Acme Lever" in names
    assert "Fusion Labs" in names
    assert "Urlonlyco" in names


def test_add_file_skips_duplicates_and_reports_errors(tmp_path, capsys):
    upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")

    import_file = tmp_path / "companies_bad.txt"
    import_file.write_text(
        "\n".join(
            [
                "Acme | https://jobs.lever.co/acme2",      # duplicate name
                "Beta | https://jobs.lever.co/acme",       # duplicate url
                "not-a-url",                               # parse error
            ]
        ),
        encoding="utf-8",
    )

    with patch("main.detect", return_value=("lever", "x")):
        _run_cli(["add-file", str(import_file)])

    out = capsys.readouterr().out
    assert "SKIP name exists" in out
    assert "SKIP url exists" in out
    assert "ERROR" in out


def test_add_file_missing_path(capsys):
    _run_cli(["add-file", "/tmp/does-not-exist-jobtracker.txt"])
    out = capsys.readouterr().out
    assert "File not found:" in out


def test_export_companies_writes_add_file_compatible_format(tmp_path, capsys):
    upsert_company("Acme", "https://jobs.lever.co/acme", "lever", "acme")
    upsert_company("Fusion Labs", "https://boards.greenhouse.io/fusionlabs", "greenhouse", "fusionlabs")

    out_path = tmp_path / "companies_export.txt"
    _run_cli(["export-companies", str(out_path)])

    out = capsys.readouterr().out
    assert "Exported 2 companies" in out

    content = out_path.read_text(encoding="utf-8")
    assert "Acme | https://jobs.lever.co/acme" in content
    assert "Fusion Labs | https://boards.greenhouse.io/fusionlabs" in content


def test_export_companies_empty_db_creates_empty_file(tmp_path, capsys):
    out_path = tmp_path / "empty_export.txt"
    _run_cli(["export-companies", str(out_path)])

    out = capsys.readouterr().out
    assert "Exported 0 companies" in out
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


def test_ask_usage_when_missing_question(capsys):
    _run_cli(["ask"])
    out = capsys.readouterr().out
    assert "Usage: python main.py ask <question>" in out


def test_ask_calls_agent_and_prints_reply(capsys):
    with patch("main.get_resume_text", return_value="Systems integration and controls background."), patch(
        "main.agent_loop", return_value="Top matches: [99], [73]"
    ) as mocked:
        _run_cli(["ask", "What", "are", "good", "systems", "roles", "?"])
    out = capsys.readouterr().out
    assert "Top matches" in out
    mocked.assert_called_once()


def test_ask_uses_dedicated_model_provider_and_detailed_tooling(capsys):
    with patch("main._get_ask_model", return_value="gpt-4.1"), patch(
        "main._get_ask_provider", return_value="openai"
    ), patch("main.get_resume_text", return_value="Resume context test"), patch(
        "main.agent_loop", return_value="answer"
    ) as mocked:
        _run_cli(["ask", "tell", "me", "about", "job", "71"])

    _ = capsys.readouterr().out
    kwargs = mocked.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1"
    assert kwargs["provider"] == "openai"
    tool_names = [t["function"]["name"] for t in kwargs["tools"]]
    assert "search_jobs" in tool_names
    assert "get_job_details" in tool_names
    assert "get_briefing_data" in tool_names
    assert "get_job_details" in kwargs["tool_funcs"]
    assert "Candidate resume context" in kwargs["task"]
    assert "Resume context test" in kwargs["task"]
