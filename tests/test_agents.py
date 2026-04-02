"""
tests/test_agents.py

Tests for the agent layer. External model/API calls are mocked throughout.
Run with: uv run -m pytest tests/test_agents.py -v
"""

import json
import types
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Redirect DB to a temp file
import db.database as db_module
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
TEST_DB_PATH = Path(_tmp.name)
db_module.DB_PATH = TEST_DB_PATH

from db.database import init_db, upsert_company, upsert_job, get_new_unscored_jobs
from agents.tools import (
    check_new_jobs,
    score_all_new_jobs,
    get_briefing_data,
    _score_one_job,
)
import agents.tools as tools_module
from agents.functions import agent, agent_run


@pytest.fixture(autouse=True)
def fresh_db():
    # Re-assert the path each test in case another test module changed it
    db_module.DB_PATH = TEST_DB_PATH
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db()
    yield
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def company_id():
    return upsert_company("Acme Fusion", "https://acmefusion.com/careers", "greenhouse", "acmefusion")


# ---------------------------------------------------------------------------
# check_new_jobs tool
# ---------------------------------------------------------------------------

class TestCheckNewJobs:
    def test_unknown_company_returns_message(self):
        result = check_new_jobs("Nonexistent Corp")
        assert "not found" in result.lower()

    def test_new_jobs_are_stored(self, company_id):
        mock_jobs = [
            {
                "external_id": "gh-1",
                "title": "Systems Engineer",
                "url": "https://example.com/job/1",
                "description": "Integration and test.",
                "location": "New York, NY",
                "date_posted": None,
                "exclude_flag": 0,
                "geo_match": 1,
                "title_excluded": 0,
            }
        ]
        with patch("agents.tools.fetch_company", return_value=mock_jobs):
            result = check_new_jobs("Acme Fusion")

        assert "1 new" in result
        jobs = get_new_unscored_jobs()
        assert len(jobs) == 1
        assert jobs[0]["title"] == "Systems Engineer"

    def test_duplicate_job_not_counted_as_new(self, company_id):
        upsert_job(company_id, "gh-1", "Systems Engineer", None, None, "NYC", None)
        mock_jobs = [
            {
                "external_id": "gh-1",
                "title": "Systems Engineer",
                "url": None,
                "description": None,
                "location": "NYC",
                "date_posted": None,
                "exclude_flag": 0,
                "geo_match": 1,
                "title_excluded": 0,
            }
        ]
        with patch("agents.tools.fetch_company", return_value=mock_jobs):
            result = check_new_jobs("Acme Fusion")

        assert "0 new" in result

    def test_fetch_failure_returns_error_message(self, company_id):
        with patch("agents.tools.fetch_company", side_effect=Exception("timeout")):
            result = check_new_jobs("Acme Fusion")
        assert "failed" in result.lower()

    def test_title_excluded_job_not_in_unscored_queue(self, company_id):
        mock_jobs = [
            {
                "external_id": "gh-2", "title": "Marketing Manager",
                "url": None, "description": "Lead brand campaigns.", "location": "NYC",
                "date_posted": None, "exclude_flag": 0, "geo_match": 1, "title_excluded": 1,
            }
        ]
        with patch("agents.tools.fetch_company", return_value=mock_jobs):
            check_new_jobs("Acme Fusion")
        assert get_new_unscored_jobs() == []

    def test_empty_successful_fetch_marks_prior_jobs_disappeared(self, company_id):
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", None, None, "NYC", None)
        with patch("agents.tools.fetch_company", return_value=[]):
            result = check_new_jobs("Acme Fusion")

        assert "fetched 0 postings" in result
        row = db_module.get_connection().execute(
            "SELECT job_state FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        assert row["job_state"] == "disappeared"


# ---------------------------------------------------------------------------
# _score_one_job (direct Ollama call, mocked)
# ---------------------------------------------------------------------------

def _make_ollama_response(content: str):
    """Create a mock ollama client.chat() response."""
    mock = MagicMock()
    mock.model_dump.return_value = {"message": {"content": content}}
    return mock


class TestScoreOneJob:
    def _patch_ollama(self, content):
        return patch("agents.tools.get_ollama_client", return_value=MagicMock(
            chat=MagicMock(return_value=_make_ollama_response(content))
        ))

    def _get_job(self, job_id):
        import db.database as db
        return db.get_connection().execute(
            "SELECT j.*, c.name as company_name FROM jobs j JOIN companies c ON j.company_id=c.id WHERE j.id=?",
            (job_id,)
        ).fetchone()

    def test_returns_score_and_rationale(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Systems Engineer", None, "Integration work", "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        content = json.dumps({"score": 8.5, "rationale": "Good systems fit"})
        with self._patch_ollama(content):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score == 9.0
        assert "Good systems fit" in rationale

    def test_score_clamped_to_0_10(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Engineer", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        job = self._get_job(job_id)

        with self._patch_ollama(json.dumps({"score": 15, "rationale": "Too high"})):
            score, _ = _score_one_job(job, "resume", "prefs", "llama3.2")
        assert score == 10.0

        with self._patch_ollama(json.dumps({"score": -3, "rationale": "Too low"})):
            score, _ = _score_one_job(job, "resume", "prefs", "llama3.2")
        assert score == 0.0

    def test_handles_json_in_markdown_fence(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Engineer", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        wrapped = '```json\n{"score": 7, "rationale": "Decent fit"}\n```'
        with self._patch_ollama(wrapped):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score == 7.0
        assert rationale == "Decent fit"

    def test_ollama_failure_returns_zero(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Engineer", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        with patch("agents.tools.get_ollama_client", side_effect=Exception("connection refused")):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score == 0.0
        assert "failed" in rationale.lower()

    def test_openai_missing_api_key_returns_failure(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Engineer", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        with patch.dict("os.environ", {}, clear=True):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "gpt-4o")
        assert score == 0.0
        assert "openai_api_key is not set" in rationale.lower()

    def test_openai_json_response_parsed(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-1", "Engineer", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )

        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content='{"score": 8.2, "rationale": "Strong fit"}'))
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        fake_openai_mod = types.SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": fake_openai_mod}):
                score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "gpt-4o")

        assert score == 8.2
        assert rationale == "Strong fit"
        fake_client.chat.completions.create.assert_called_once()

    def test_guardrails_differentiate_similar_high_raw_scores(self, company_id):
        j_systems, _ = upsert_job(
            company_id, "gh-11", "Systems Engineer, Requirements and Integration", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )
        j_pm, _ = upsert_job(
            company_id, "gh-12", "Senior Technical Project Manager", None, None, "NYC", None,
            exclude_flag=0, geo_match=1, title_excluded=0
        )

        content = json.dumps({
            "industry_fit": 9,
            "role_fit": 9,
            "location_fit": 9,
            "level_fit": 9,
            "exclude_risk": 9,
            "score": 9,
            "rationale": "High-level fit",
        })
        with self._patch_ollama(content):
            s1, _ = _score_one_job(self._get_job(j_systems), "resume", "prefs", "llama3.2")
        with self._patch_ollama(content):
            s2, _ = _score_one_job(self._get_job(j_pm), "resume", "prefs", "llama3.2")

        assert s1 > s2
        assert s2 <= 7.5

    def test_exclude_flag_without_strong_signal_does_not_hard_cap(self, company_id):
        job_id, _ = upsert_job(
            company_id,
            "gh-13",
            "Systems Engineer",
            None,
            "Uses layered reliability methodology for safety analysis.",
            "NYC",
            None,
            exclude_flag=1,
            geo_match=1,
            title_excluded=0,
        )
        content = json.dumps({"score": 8.0, "rationale": "Strong fit"})
        with self._patch_ollama(content):
            score, _ = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score > 4.0

    def test_nj_location_does_not_take_geo_penalty(self, company_id):
        job_id, _ = upsert_job(
            company_id, "gh-15", "Systems Engineer", None, "Systems integration work.",
            "Kearny, NJ", None,
            exclude_flag=0, geo_match=0,  # stale flag — NJ was not recognized at fetch time
            title_excluded=0,
        )
        content = json.dumps({"score": 8.0, "rationale": "Strong fit"})
        with self._patch_ollama(content):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score >= 8.0
        assert "geo_penalty" not in rationale

    def test_low_role_fit_cap_applied(self, company_id):
        # role_fit ≤ 4 should cap score regardless of how attractive the company is
        job_id, _ = upsert_job(
            company_id, "gh-16", "Senior Controls Engineer", None, "Specialist controls work.",
            "Boston, MA", None,
            exclude_flag=0, geo_match=1, title_excluded=0,
        )
        content = json.dumps({
            "industry_fit": 10, "role_fit": 3, "location_fit": 10,
            "level_fit": 8, "exclude_risk": 10, "score": 4, "rationale": "Wrong role type",
        })
        with self._patch_ollama(content):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score <= 4.5
        assert "low_role_fit_cap" in rationale

    def test_low_role_fit_cap_not_applied_above_threshold(self, company_id):
        # role_fit = 5 should NOT trigger the cap
        job_id, _ = upsert_job(
            company_id, "gh-17", "Operations Engineer", None, "ConOps development work.",
            "Boston, MA", None,
            exclude_flag=0, geo_match=1, title_excluded=0,
        )
        content = json.dumps({
            "industry_fit": 10, "role_fit": 5, "location_fit": 10,
            "level_fit": 8, "exclude_risk": 10, "score": 6, "rationale": "Adjacent role",
        })
        with self._patch_ollama(content):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score > 5.5
        assert "low_role_fit_cap" not in rationale

    def test_any_location_does_not_take_geo_penalty_even_if_stale_flag(self, company_id):
        job_id, _ = upsert_job(
            company_id,
            "gh-14",
            "Systems Engineer, Requirements and Integration",
            None,
            "Strong systems role in energy startup.",
            "Any location",
            None,
            exclude_flag=0,
            geo_match=0,  # simulate stale pre-fix value
            title_excluded=0,
        )
        content = json.dumps({"score": 8.0, "rationale": "Strong fit"})
        with self._patch_ollama(content):
            score, rationale = _score_one_job(self._get_job(job_id), "resume", "prefs", "llama3.2")
        assert score >= 8.0
        assert "geo_penalty" not in rationale

class TestModelSelection:
    def test_tools_get_model_uses_single_model_key(self):
        with patch("agents.tools._load_prefs", return_value={"model": "gpt-4o"}):
            with patch.dict("os.environ", {}, clear=True):
                assert tools_module._get_model() == "gpt-4o"

    def test_tools_get_provider_from_config(self):
        with patch("agents.tools._load_prefs", return_value={"provider": "openai"}):
            assert tools_module._get_provider() == "openai"


# ---------------------------------------------------------------------------
# get_briefing_data tool
# ---------------------------------------------------------------------------

class TestGetBriefingData:
    def test_no_jobs_returns_message(self):
        result = get_briefing_data()
        assert "no high-scoring" in result.lower()

    def test_returns_table_with_high_score_jobs(self, company_id):
        from db.database import update_job_score
        job_id, _ = upsert_job(company_id, "gh-1", "Systems Engineer", "https://example.com",
                               "Integration work", "NYC", None)
        update_job_score(job_id, 8.0, "Great fit", "llama3.2", "v1")

        result = get_briefing_data()
        assert "Systems Engineer" in result
        assert "Acme Fusion" in result
        assert "8" in result

    def test_low_score_jobs_excluded(self, company_id):
        from db.database import update_job_score
        job_id, _ = upsert_job(company_id, "gh-1", "Finance Analyst", None, None, "NYC", None)
        update_job_score(job_id, 2.0, "Wrong field", "llama3.2", "v1")

        result = get_briefing_data()
        assert "no high-scoring" in result.lower()


# ---------------------------------------------------------------------------
# agent() dispatcher — tool_funcs registry
# ---------------------------------------------------------------------------

class TestAgentDispatcher:
    def _mock_tool_response(self, func_name, func_args):
        """Simulate ollama client.chat() responding with a tool call."""
        mock = MagicMock()
        mock.model_dump.return_value = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": func_name, "arguments": func_args}}],
            }
        }
        return mock

    def _mock_text_response(self, text):
        mock = MagicMock()
        mock.model_dump.return_value = {"message": {"content": text}}
        return mock

    def test_tool_func_dispatched_from_registry(self):
        called_with = {}

        def my_tool(value: str) -> str:
            called_with["value"] = value
            return "tool result"

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            self._mock_tool_response("my_tool", {"value": "hello"}),
            self._mock_text_response("final synthesized answer"),
        ]

        with patch("agents.functions.get_ollama_client", return_value=mock_client):
            result = agent(
                messages=[{"role": "user", "content": "test"}],
                model="llama3.2",
                tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
                tool_funcs={"my_tool": my_tool},
                output="text",
            )

        assert called_with["value"] == "hello"
        assert result == "final synthesized answer"

    def test_output_tools_returns_raw_tool_calls(self):
        def my_tool(value: str) -> str:
            return f"echo:{value}"

        mock_client = MagicMock()
        mock_client.chat.return_value = self._mock_tool_response("my_tool", {"value": "hello"})

        with patch("agents.functions.get_ollama_client", return_value=mock_client):
            result = agent(
                messages=[{"role": "user", "content": "test"}],
                model="llama3.2",
                tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
                tool_funcs={"my_tool": my_tool},
                output="tools",
            )

        assert isinstance(result, list)
        assert result[0]["function"]["name"] == "my_tool"
        assert result[0]["output"] == "echo:hello"

    def test_no_tool_call_returns_text(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = self._mock_text_response("Hello from the model")

        with patch("agents.functions.get_ollama_client", return_value=mock_client):
            result = agent(messages=[{"role": "user", "content": "hi"}], model="llama3.2")

        assert result == "Hello from the model"

    def test_provider_override_routes_to_openai_path(self):
        with patch("agents.functions._openai_chat", return_value={"message": {"content": "openai text"}}) as mock_openai:
            result = agent(
                messages=[{"role": "user", "content": "hi"}],
                model="llama3.2",   # non-openai model name on purpose
                provider="openai",  # explicit provider should win
                output="text",
            )
        assert result == "openai text"
        mock_openai.assert_called_once()

    def test_provider_override_routes_to_anthropic_path(self):
        with patch("agents.functions._anthropic_chat", return_value={"message": {"content": "anthropic text"}}) as mock_anthropic:
            result = agent(
                messages=[{"role": "user", "content": "hi"}],
                model="llama3.2",    # non-anthropic model name on purpose
                provider="anthropic",  # explicit provider should win
                output="text",
            )
        assert result == "anthropic text"
        mock_anthropic.assert_called_once()

    def test_provider_override_routes_to_ollama_path(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = self._mock_text_response("ollama text")
        with patch("agents.functions.get_ollama_client", return_value=mock_client) as mock_ollama:
            result = agent(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",     # openai-like model name on purpose
                provider="ollama",  # explicit provider should win
                output="text",
            )
        assert result == "ollama text"
        mock_ollama.assert_called_once()
