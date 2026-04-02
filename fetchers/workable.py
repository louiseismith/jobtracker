"""
fetchers/workable.py

Fetches job postings from the Workable public job board API.
No authentication required.

List API:  POST https://apply.workable.com/api/v3/accounts/{slug}/jobs
Detail API: GET https://apply.workable.com/api/v2/accounts/{slug}/jobs/{shortcode}
"""

import re
import requests

LIST_URL   = "https://apply.workable.com/api/v3/accounts/{ats_id}/jobs"
DETAIL_URL = "https://apply.workable.com/api/v2/accounts/{ats_id}/jobs/{shortcode}"
JOB_URL    = "https://apply.workable.com/{ats_id}/j/{shortcode}"
TIMEOUT    = 15


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings for a Workable board.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    resp = requests.post(
        LIST_URL.format(ats_id=ats_id),
        json={"query": "", "location": [], "department": [], "worktype": [], "remote": []},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    listings = resp.json().get("results", [])

    jobs = []
    for listing in listings:
        shortcode = listing.get("shortcode", "")
        loc = listing.get("location") or {}
        location_parts = [loc.get("city"), loc.get("region"), loc.get("country")]
        location_str = ", ".join(p for p in location_parts if p)

        description = _fetch_description(ats_id, shortcode)

        jobs.append({
            "external_id": shortcode,
            "title":       listing.get("title", ""),
            "url":         JOB_URL.format(ats_id=ats_id, shortcode=shortcode),
            "description": description,
            "location":    location_str,
            "date_posted": listing.get("published", ""),
        })

    return jobs


def _fetch_description(ats_id: str, shortcode: str) -> str:
    """Fetch full description for a single job via the v2 detail endpoint."""
    try:
        resp = requests.get(
            DETAIL_URL.format(ats_id=ats_id, shortcode=shortcode),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        html = (data.get("description") or "") + " " + (data.get("requirements") or "")
        return _strip_html(html).strip()
    except Exception:
        return ""


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
