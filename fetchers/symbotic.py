"""
fetchers/symbotic.py

Fetches job postings from Symbotic's custom careers page.
All job data (title, location, description) is embedded in the HTML as
data attributes on a single page — no per-job requests needed.

ats_type: "symbotic"
ats_id:   unused (pass None or any value)
"""

import html
import re
import requests

CAREERS_URL = "https://www.symbotic.com/careers/open-positions/"
BASE_URL    = "https://www.symbotic.com"
TIMEOUT     = 15
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Matches one job card element. Attribute order is consistent across the page.
_JOB_PATTERN = re.compile(
    r'data-location-title="(?P<location>[^"]*)"'
    r'[^>]*data-search="(?P<description>[^"]*)"'
    r'[^>]*>'                                          # close of the element tag
    r'.*?'                                             # inner content
    r'href="(?P<path>/careers/open-positions/(?P<req_id>R[^"]+))"[^>]*>'
    r'\s*(?P<title>[^<\n]+?)\s*</a>',
    re.DOTALL,
)


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings from Symbotic's careers page.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    resp = requests.get(CAREERS_URL, headers={"User-Agent": UA}, timeout=TIMEOUT)
    resp.raise_for_status()

    jobs = []
    for m in _JOB_PATTERN.finditer(resp.text):
        jobs.append({
            "external_id": m.group("req_id"),
            "title":       html.unescape(m.group("title")).strip(),
            "url":         BASE_URL + m.group("path"),
            "description": html.unescape(m.group("description")).strip(),
            "location":    html.unescape(m.group("location")).strip(),
            "date_posted": "",
        })

    return jobs
