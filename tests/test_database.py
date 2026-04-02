"""
tests/test_database.py

Tests for db/database.py using a temporary database file.
Run with: .venv/bin/python -m pytest tests/
"""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

# Point DB_PATH at a temp file before importing database functions
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
TEST_DB_PATH = Path(_tmp.name)

import db.database as db_module
db_module.DB_PATH = TEST_DB_PATH


from db.database import (
    init_db,
    upsert_company,
    upsert_job,
    update_job_score,
    update_job_states,
    mark_missing_jobs,
    get_new_unscored_jobs,
    get_briefing_jobs,
    search_jobs,
)


@pytest.fixture(autouse=True)
def fresh_db():
    """Wipe and re-initialize the DB before each test."""
    db_module.DB_PATH = TEST_DB_PATH
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db()
    yield
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def company_id():
    return upsert_company("Acme Fusion", "https://acme.com/careers", "greenhouse", "acmefusion")


# ---------------------------------------------------------------------------
# upsert_job — deduplication
# ---------------------------------------------------------------------------

class TestUpsertJob:
    def test_new_job_returns_is_new_true(self, company_id):
        job_id, is_new = upsert_job(company_id, "gh-123", "Systems Engineer", None, "Great role", "NYC", None)
        assert is_new is True
        assert job_id is not None

    def test_duplicate_external_id_not_inserted(self, company_id):
        job_id1, is_new1 = upsert_job(company_id, "gh-123", "Systems Engineer", None, "Great role", "NYC", None)
        job_id2, is_new2 = upsert_job(company_id, "gh-123", "Systems Engineer", None, "Great role", "NYC", None)
        assert is_new1 is True
        assert is_new2 is False
        assert job_id1 == job_id2

    def test_same_external_id_different_company_is_new(self, company_id):
        company_id2 = upsert_company("Other Corp", "https://other.com/jobs", "lever", "othercorp")
        job_id1, is_new1 = upsert_job(company_id,  "gh-123", "Systems Engineer", None, None, "NYC", None)
        job_id2, is_new2 = upsert_job(company_id2, "gh-123", "Systems Engineer", None, None, "NYC", None)
        assert is_new1 is True
        assert is_new2 is True
        assert job_id1 != job_id2

    def test_null_external_id_always_inserts(self, company_id):
        # Scraped jobs have no external_id — each call should create a new row
        job_id1, is_new1 = upsert_job(company_id, None, "Systems Engineer", None, None, "NYC", None)
        job_id2, is_new2 = upsert_job(company_id, None, "Systems Engineer", None, None, "NYC", None)
        assert is_new1 is True
        assert is_new2 is True
        assert job_id1 != job_id2

    def test_exclude_and_geo_flags_stored(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-999", "Engineer", None, None, "Boston", None,
                                exclude_flag=1, geo_match=1, title_excluded=0)
        conn = db_module.get_connection()
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["exclude_flag"] == 1
        assert row["geo_match"] == 1
        assert row["title_excluded"] == 0

    def test_update_job_score_stores_components_json(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-321", "Systems Engineer", None, "desc", "NYC", None)
        update_job_score(
            job_id,
            8.0,
            "Strong fit",
            "gpt-4o",
            "v2",
            components={"industry_fit": 9, "role_fit": 8, "final_score": 8.0},
        )
        conn = db_module.get_connection()
        row = conn.execute("SELECT ai_components FROM jobs WHERE id=?", (job_id,)).fetchone()
        parsed = json.loads(row["ai_components"])
        assert parsed["industry_fit"] == 9
        assert parsed["final_score"] == 8.0


# ---------------------------------------------------------------------------
# mark_missing_jobs
# ---------------------------------------------------------------------------

class TestMarkMissingJobs:
    def test_missing_job_marked_disappeared(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, None, "NYC", None)
        # Fetch returns no jobs — gh-1 has disappeared
        mark_missing_jobs(company_id, current_external_ids=set())
        conn = db_module.get_connection()
        row = conn.execute("SELECT job_state FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["job_state"] == "disappeared"

    def test_present_job_stays_open(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, None, "NYC", None)
        mark_missing_jobs(company_id, current_external_ids={"gh-1"})
        conn = db_module.get_connection()
        row = conn.execute("SELECT job_state FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["job_state"] == "open"

    def test_only_affects_target_company(self, company_id):
        company_id2 = upsert_company("Other Corp", "https://other.com/jobs", "lever", "othercorp")
        job_id1, _ = upsert_job(company_id,  "gh-1", "Engineer", None, None, "NYC", None)
        job_id2, _ = upsert_job(company_id2, "lv-1", "Engineer", None, None, "NYC", None)
        # Mark missing for company 1 only
        mark_missing_jobs(company_id, current_external_ids=set())
        conn = db_module.get_connection()
        assert conn.execute("SELECT job_state FROM jobs WHERE id=?", (job_id1,)).fetchone()["job_state"] == "disappeared"
        assert conn.execute("SELECT job_state FROM jobs WHERE id=?", (job_id2,)).fetchone()["job_state"] == "open"

    def test_null_external_id_jobs_not_affected(self, company_id):
        # Scraped jobs (no external_id) should never be marked disappeared
        job_id, _ = upsert_job(company_id, None, "Engineer", None, None, "NYC", None)
        mark_missing_jobs(company_id, current_external_ids=set())
        conn = db_module.get_connection()
        row = conn.execute("SELECT job_state FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["job_state"] == "open"


# ---------------------------------------------------------------------------
# Query filters
# ---------------------------------------------------------------------------

class TestQueryFilters:
    def test_get_new_unscored_excludes_scored(self, company_id):
        job_id1, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, "desc", "NYC", None)
        job_id2, _ = upsert_job(company_id, "gh-2", "TPM", None, "desc", "Boston", None)
        update_job_score(job_id1, 8.0, "Good fit", "llama3.2", "v1")
        unscored = get_new_unscored_jobs()
        ids = [r["id"] for r in unscored]
        assert job_id1 not in ids
        assert job_id2 in ids

    def test_get_new_unscored_excludes_disappeared(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, "desc", "NYC", None)
        update_job_states(job_id, job_state="disappeared")
        assert get_new_unscored_jobs() == []

    def test_get_briefing_jobs_min_score(self, company_id):
        job_id_high, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, "desc", "NYC", None)
        job_id_low,  _ = upsert_job(company_id, "gh-2", "TPM", None, "desc", "Boston", None)
        update_job_score(job_id_high, 8.0, "Great fit", "llama3.2", "v1")
        update_job_score(job_id_low,  4.0, "Weak fit",  "llama3.2", "v1")
        briefing = get_briefing_jobs(min_score=6.0)
        ids = [r["id"] for r in briefing]
        assert job_id_high in ids
        assert job_id_low not in ids

    def test_get_briefing_jobs_excludes_applied(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, "desc", "NYC", None)
        update_job_score(job_id, 9.0, "Perfect fit", "llama3.2", "v1")
        update_job_states(job_id, application_state="applied")
        assert get_briefing_jobs() == []


# ---------------------------------------------------------------------------
# FTS5 search
# ---------------------------------------------------------------------------

class TestFTSSearch:
    def test_search_finds_matching_job(self, company_id):
        upsert_job(company_id, "gh-1", "Systems Integration Engineer", None,
                   "You will own requirements and V&V", "NYC", None)
        results = search_jobs("requirements")
        assert len(results) == 1
        assert results[0]["title"] == "Systems Integration Engineer"

    def test_search_no_false_positives(self, company_id):
        upsert_job(company_id, "gh-1", "Marketing Manager", None,
                   "Brand strategy and campaigns", "NYC", None)
        results = search_jobs("systems engineer")
        assert len(results) == 0

    def test_search_across_multiple_jobs(self, company_id):
        upsert_job(company_id, "gh-1", "Systems Engineer", None, "Integration and test", "NYC", None)
        upsert_job(company_id, "gh-2", "TPM", None, "Systems-level program management", "Boston", None)
        upsert_job(company_id, "gh-3", "Marketing", None, "Brand and campaigns", "NYC", None)
        results = search_jobs("systems")
        titles = [r["title"] for r in results]
        assert "Systems Engineer" in titles
        assert "TPM" in titles
        assert "Marketing" not in titles
