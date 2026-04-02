"""
fetchers/lever.py

Fetches job postings from the Lever public API.
No authentication required.
API docs: https://help.lever.co/hc/en-us/articles/206439149
"""

import requests

BASE_URL = "https://api.lever.co/v0/postings/{ats_id}"
TIMEOUT = 15


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings for a Lever board.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    url = BASE_URL.format(ats_id=ats_id)
    resp = requests.get(url, params={"mode": "json"}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for job in data:
        # Description: Lever returns plain text in descriptionPlain, HTML in description
        description = job.get("descriptionPlain") or _strip_html(job.get("description", ""))

        # Location: Lever stores as a plain string
        location = job.get("categories", {}).get("location", "") or job.get("workplaceType", "")

        # Date: Lever returns Unix ms timestamp
        created_at = job.get("createdAt")
        date_posted = _ms_to_iso(created_at) if created_at else ""

        jobs.append({
            "external_id": job.get("id", ""),
            "title":       job.get("text", ""),
            "url":         job.get("hostedUrl", ""),
            "description": description,
            "location":    location,
            "date_posted": date_posted,
        })

    return jobs


def _ms_to_iso(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _strip_html(html: str) -> str:
    import re
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
