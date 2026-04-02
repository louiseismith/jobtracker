"""
tests/test_fetchers.py

Tests for the fetcher layer. HTTP calls are mocked so tests never hit live APIs.
Run with: .venv/bin/python -m pytest tests/test_fetchers.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from fetchers.detector import detect, _detect_from_url, _detect_from_html
from fetchers import greenhouse, lever
from fetchers import fetch_company, _check_geo, _check_title_excluded, _check_exclude


# ---------------------------------------------------------------------------
# detector — URL pattern matching (no network calls needed)
# ---------------------------------------------------------------------------

class TestDetectorFromUrl:
    def test_greenhouse_boards_url(self):
        assert _detect_from_url("https://boards.greenhouse.io/acmecorp") == ("greenhouse", "acmecorp")

    def test_greenhouse_job_boards_url(self):
        assert _detect_from_url("https://job-boards.greenhouse.io/acmecorp") == ("greenhouse", "acmecorp")

    def test_lever_url(self):
        assert _detect_from_url("https://jobs.lever.co/acmecorp") == ("lever", "acmecorp")

    def test_ashby_url(self):
        assert _detect_from_url("https://jobs.ashbyhq.com/acmecorp") == ("ashby", "acmecorp")

    def test_unknown_url_returns_scrape(self):
        assert _detect_from_url("https://careers.acmecorp.com/jobs") == ("scrape", None)

    def test_slug_with_trailing_path(self):
        ats_type, ats_id = _detect_from_url("https://boards.greenhouse.io/acmecorp/jobs/123")
        assert ats_type == "greenhouse"
        assert ats_id == "acmecorp"


class TestDetectorFromHtml:
    def test_greenhouse_embed_in_html(self):
        html = '<script src="https://boards.greenhouse.io/embed/job_board?for=acmecorp"></script>'
        assert _detect_from_html(html, "https://careers.acme.com") == ("greenhouse", "acmecorp")

    def test_lever_link_in_html(self):
        html = '<a href="https://jobs.lever.co/acmecorp/123">Apply</a>'
        assert _detect_from_html(html, "https://careers.acme.com") == ("lever", "acmecorp")

    def test_unknown_html_returns_scrape(self):
        html = "<html><body>We are hiring!</body></html>"
        assert _detect_from_html(html, "https://careers.acme.com") == ("scrape", None)

    def test_lever_slug_differs_from_domain(self):
        # Real-world case: CFS careers page (cfs.energy) links to jobs.lever.co/cfsenergy
        # Slug in HTML may differ from company domain — detector must extract from the link
        html = '<a href="https://jobs.lever.co/cfsenergy">Current Openings</a>'
        assert _detect_from_html(html, "https://www.cfs.energy/careers/") == ("lever", "cfsenergy")


class TestDetectorWithNetwork:
    def test_detect_falls_through_to_html_on_redirect(self):
        """When URL doesn't match, detect() fetches the page and checks HTML."""
        mock_response = MagicMock()
        mock_response.text = '<script src="https://boards.greenhouse.io/embed/job_board?for=acmecorp">'
        mock_response.url = "https://careers.acme.com/jobs"
        mock_response.raise_for_status = MagicMock()

        with patch("fetchers.detector.requests.get", return_value=mock_response):
            ats_type, ats_id = detect("https://careers.acme.com/jobs")
        assert ats_type == "greenhouse"
        assert ats_id == "acmecorp"

    def test_detect_returns_scrape_on_network_failure(self):
        import requests as req
        with patch("fetchers.detector.requests.get", side_effect=req.RequestException("timeout")):
            # URL doesn't match any ATS pattern, and network call fails
            result = detect("https://careers.unknown.com")
        assert result == ("scrape", None)


# ---------------------------------------------------------------------------
# Greenhouse fetcher
# ---------------------------------------------------------------------------

GREENHOUSE_RESPONSE = {
    "jobs": [
        {
            "id": 12345,
            "title": "Systems Integration Engineer",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/12345",
            "content": "<p>You will own requirements and V&amp;V for our fusion reactor systems.</p>",
            "offices": [{"name": "New York, NY"}],
            "updated_at": "2026-03-01T00:00:00Z",
        },
        {
            "id": 12346,
            "title": "Marketing Manager",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/12346",
            "content": "<p>Lead brand campaigns.</p>",
            "offices": [{"name": "Remote"}],
            "updated_at": "2026-03-02T00:00:00Z",
        },
    ]
}


class TestGreenhouseFetcher:
    def _mock_response(self, data):
        mock = MagicMock()
        mock.json.return_value = data
        mock.raise_for_status = MagicMock()
        return mock

    def test_returns_normalized_jobs(self):
        with patch("fetchers.greenhouse.requests.get", return_value=self._mock_response(GREENHOUSE_RESPONSE)):
            jobs = greenhouse.fetch("acme")
        assert len(jobs) == 2
        assert jobs[0]["external_id"] == "12345"
        assert jobs[0]["title"] == "Systems Integration Engineer"
        assert jobs[0]["location"] == "New York, NY"

    def test_strips_html_from_description(self):
        with patch("fetchers.greenhouse.requests.get", return_value=self._mock_response(GREENHOUSE_RESPONSE)):
            jobs = greenhouse.fetch("acme")
        assert "<p>" not in jobs[0]["description"]
        assert "requirements" in jobs[0]["description"]
        assert "&amp;" not in jobs[0]["description"]

    def test_empty_board_returns_empty_list(self):
        with patch("fetchers.greenhouse.requests.get", return_value=self._mock_response({"jobs": []})):
            jobs = greenhouse.fetch("acme")
        assert jobs == []


# ---------------------------------------------------------------------------
# Lever fetcher
# ---------------------------------------------------------------------------

LEVER_RESPONSE = [
    {
        "id": "abc-123",
        "text": "Technical Program Manager",
        "hostedUrl": "https://jobs.lever.co/acme/abc-123",
        "descriptionPlain": "You will manage complex hardware programs.",
        "categories": {"location": "Boston, MA"},
        "createdAt": 1740000000000,
    },
    {
        "id": "abc-124",
        "text": "Finance Analyst",
        "hostedUrl": "https://jobs.lever.co/acme/abc-124",
        "descriptionPlain": "Manage financial reporting.",
        "categories": {"location": "Austin, TX"},
        "createdAt": 1740000000000,
    },
]


class TestLeverFetcher:
    def _mock_response(self, data):
        mock = MagicMock()
        mock.json.return_value = data
        mock.raise_for_status = MagicMock()
        return mock

    def test_returns_normalized_jobs(self):
        with patch("fetchers.lever.requests.get", return_value=self._mock_response(LEVER_RESPONSE)):
            jobs = lever.fetch("acme")
        assert len(jobs) == 2
        assert jobs[0]["external_id"] == "abc-123"
        assert jobs[0]["title"] == "Technical Program Manager"
        assert jobs[0]["location"] == "Boston, MA"

    def test_uses_plain_text_description(self):
        with patch("fetchers.lever.requests.get", return_value=self._mock_response(LEVER_RESPONSE)):
            jobs = lever.fetch("acme")
        assert jobs[0]["description"] == "You will manage complex hardware programs."

    def test_empty_board_returns_empty_list(self):
        with patch("fetchers.lever.requests.get", return_value=self._mock_response([])):
            jobs = lever.fetch("acme")
        assert jobs == []


# ---------------------------------------------------------------------------
# Rule-based filters
# ---------------------------------------------------------------------------

PREFS = {
    "geography": {"allowed": ["New York", "NYC", "Boston", "Remote"]},
    "role_types": {
        "exclude_titles": ["intern", "marketing", "finance", "legal", "recruiter"]
    },
    "hard_excludes": {
        "exclude_strong_signals": ["clearance required", "export controlled"],
        "exclude_generic_terms": ["restricted"],
        "exclude_context_terms": ["restricted program", "regulated industry"],
    },
}


class TestGeoFilter:
    def test_nyc_matches(self):
        assert _check_geo("New York, NY", PREFS) == 1

    def test_boston_matches(self):
        assert _check_geo("Boston, MA", PREFS) == 1

    def test_remote_matches(self):
        assert _check_geo("Remote", PREFS) == 1

    def test_austin_does_not_match(self):
        assert _check_geo("Austin, TX", PREFS) == 0

    def test_empty_location_returns_none(self):
        assert _check_geo("", PREFS) is None

    def test_any_location_matches(self):
        assert _check_geo("Any location", PREFS) == 1

    def test_massachusetts_matches_when_boston_allowed(self):
        assert _check_geo("Devens, MA", PREFS) == 1

    def test_nj_matches_when_nyc_allowed(self):
        assert _check_geo("Kearny, NJ", PREFS) == 1

    def test_nj_long_form_matches_when_nyc_allowed(self):
        assert _check_geo("Jersey City, New Jersey", PREFS) == 1

    def test_nj_does_not_match_when_nyc_not_in_allowed(self):
        prefs_no_nyc = {"geography": {"allowed": ["Boston", "Remote"]}}
        assert _check_geo("Kearny, NJ", prefs_no_nyc) == 0


class TestTitleExcludeFilter:
    def test_marketing_excluded(self):
        assert _check_title_excluded("Marketing Manager", PREFS) == 1

    def test_intern_excluded(self):
        assert _check_title_excluded("Software Engineering Intern", PREFS) == 1

    def test_systems_engineer_not_excluded(self):
        assert _check_title_excluded("Systems Integration Engineer", PREFS) == 0

    def test_case_insensitive(self):
        assert _check_title_excluded("FINANCE Analyst", PREFS) == 1


class TestExcludeFilter:
    def test_clearance_flagged(self):
        assert _check_exclude("Role is clearance required for deployment access", PREFS) == 1

    def test_export_control_flagged(self):
        assert _check_exclude("Must comply with export controlled project controls", PREFS) == 1

    def test_restricted_program_context_flagged(self):
        assert _check_exclude("Experience with restricted program delivery", PREFS) == 1

    def test_clean_posting_not_flagged(self):
        assert _check_exclude("Systems integration at a fusion energy startup", PREFS) == 0

    def test_case_insensitive(self):
        assert _check_exclude("Worked in RESTRICTED PROGRAM operations", PREFS) == 1

    def test_generic_term_without_context_does_not_always_flag(self):
        # Avoid false positives for standalone generic terms.
        assert _check_exclude("Experience with restricted data handling", PREFS) == 0

    def test_generic_with_context_flags(self):
        assert _check_exclude("Experience in restricted program operations", PREFS) == 1

    def test_non_matching_text_not_flagged(self):
        assert _check_exclude("Hazard-rated fire protection systems", PREFS) == 0

    def test_config_driven_generic_and_context_terms(self):
        prefs = {
            "hard_excludes": {
                "exclude_strong_signals": ["clearance required"],
                "exclude_generic_terms": ["restricted"],
                "exclude_context_terms": ["regulated industry"],
            }
        }
        assert _check_exclude("Restricted architecture patterns", prefs) == 0
        assert _check_exclude("Experience in restricted work in a regulated industry", prefs) == 1


# ---------------------------------------------------------------------------
# Integrated fetch_company with filters
# ---------------------------------------------------------------------------

class TestFetchCompany:
    def _make_company(self, ats_type, ats_id):
        row = MagicMock()
        row.__getitem__ = lambda self, key: {"ats_type": ats_type, "ats_id": ats_id}[key]
        return row

    def test_greenhouse_company_gets_filter_flags(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = GREENHOUSE_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        company = self._make_company("greenhouse", "acme")
        with patch("fetchers.greenhouse.requests.get", return_value=mock_resp):
            jobs = fetch_company(company)

        assert len(jobs) == 2
        # First job: Systems Integration Engineer in NYC — should pass geo, not excluded
        assert jobs[0]["geo_match"] == 1
        assert jobs[0]["title_excluded"] == 0
        # Second job: Marketing Manager — title excluded
        assert jobs[1]["title_excluded"] == 1

    def test_unsupported_ats_raises(self):
        company = self._make_company("scrape", None)
        with pytest.raises(NotImplementedError):
            fetch_company(company)
