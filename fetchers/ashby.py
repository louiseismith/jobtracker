"""
fetchers/ashby.py

Fetches job postings from the Ashby public posting API.
No authentication required.
API: GET https://api.ashbyhq.com/posting-api/job-board/{slug}
"""

import re
import requests

BASE_URL = "https://api.ashbyhq.com/posting-api/job-board/{ats_id}"
TIMEOUT = 15


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings for an Ashby board.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    url = BASE_URL.format(ats_id=ats_id)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or ""

        description = job.get("descriptionPlain") or ""
        if not description:
            description_html = job.get("descriptionHtml") or ""
            description = _strip_html(description_html) if description_html else ""

        url = job.get("jobUrl") or job.get("applyUrl") or ""

        jobs.append({
            "external_id": str(job["id"]),
            "title":       job.get("title", ""),
            "url":         url,
            "description": description,
            "location":    location,
            "date_posted": job.get("publishedAt", ""),
        })

    return jobs


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
