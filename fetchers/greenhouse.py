"""
fetchers/greenhouse.py

Fetches job postings from the Greenhouse public API.
No authentication required.
API docs: https://developers.greenhouse.io/job-board.html
"""

import requests

BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{ats_id}/jobs"
TIMEOUT = 15


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings for a Greenhouse board.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    url = BASE_URL.format(ats_id=ats_id)
    resp = requests.get(url, params={"content": "true"}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for job in data.get("jobs", []):
        # Location: Greenhouse returns a list of office locations
        locations = job.get("offices", [])
        location_str = ", ".join(loc.get("name", "") for loc in locations if loc.get("name"))

        # Description: returned as HTML — strip to plain text
        description_html = job.get("content", "") or ""
        description = _strip_html(description_html)

        jobs.append({
            "external_id": str(job["id"]),
            "title":       job.get("title", ""),
            "url":         job.get("absolute_url", ""),
            "description": description,
            "location":    location_str,
            "date_posted": job.get("updated_at", ""),
        })

    return jobs


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string."""
    import re
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
