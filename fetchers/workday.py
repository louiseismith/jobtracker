"""
fetchers/workday.py

Fetches job postings from Workday job boards via the hidden CXS API.
No authentication required.

ats_id format: "{tenant}.wd{n}|{board}"
  e.g. "bostondynamics.wd1|Boston_Dynamics"
  e.g. "insulet.wd5|insuletcareers"

List endpoint:   POST https://{tenant}.wd{n}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs
Description:     GET  https://{tenant}.wd{n}.myworkdayjobs.com/{board}{externalPath}
                 (parse JSON-LD JobPosting block from HTML)
"""

import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PAGE_SIZE = 20
TIMEOUT = 15

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
JSON_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": UA,
}


def fetch(ats_id: str) -> list[dict]:
    """
    Fetch all current job postings for a Workday board.

    Returns a normalized list of dicts:
        external_id, title, url, description, location, date_posted
    """
    subdomain, board = ats_id.split("|", 1)
    tenant = subdomain.split(".")[0]  # "bostondynamics" from "bostondynamics.wd1"
    base_url = f"https://{subdomain}.myworkdayjobs.com"
    list_url = f"{base_url}/wday/cxs/{tenant}/{board}/jobs"

    all_postings = _paginate(list_url)

    # Build metadata for each posting before fetching details.
    posting_meta = []
    for posting in all_postings:
        external_path = posting.get("externalPath", "")
        job_url = f"{base_url}/{board}{external_path}" if external_path else ""
        bullet = posting.get("bulletFields", [])
        external_id = bullet[0] if bullet else external_path.rstrip("/").split("/")[-1]
        posting_meta.append({
            "external_id":      external_id,
            "title":            posting.get("title", ""),
            "url":              job_url,
            "listing_location": posting.get("locationsText", ""),
            "date_posted":      posting.get("postedOn", ""),
        })

    # Fetch all detail pages in parallel.
    details: dict[str, tuple[str, str]] = {}  # url → (description, location)
    urls = [m["url"] for m in posting_meta if m["url"]]
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(_fetch_detail, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            details[url] = future.result()

    jobs = []
    for meta in posting_meta:
        description, detail_location = details.get(meta["url"], ("", ""))
        listing_location = meta["listing_location"]
        # Prefer the JSON-LD location from the detail page — it's always a real
        # city/state string. listing_location can be an internal office code
        # (e.g. "WAP C") or "N Locations", both useless for geo filtering.
        location = detail_location or listing_location
        jobs.append({
            "external_id": meta["external_id"],
            "title":       meta["title"],
            "url":         meta["url"],
            "description": description,
            "location":    location,
            "date_posted": meta["date_posted"],
        })

    return jobs


def _paginate(list_url: str) -> list[dict]:
    """Page through the jobs list endpoint until all results are collected."""
    all_postings = []
    offset = 0

    while True:
        resp = requests.post(
            list_url,
            json={"appliedFacets": {}, "limit": PAGE_SIZE, "offset": offset, "searchText": ""},
            headers=JSON_HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        postings = data.get("jobPostings", [])
        all_postings.extend(postings)

        total = data.get("total", 0)
        if not postings or len(all_postings) >= total:
            break
        offset += PAGE_SIZE

    return all_postings


def _fetch_detail(job_url: str) -> tuple[str, str]:
    """
    Fetch the job detail HTML page and extract description and location from the
    JSON-LD JobPosting block embedded by Workday on every job listing page.

    Returns (description, location). Both are empty strings on failure.
    """
    if not job_url:
        return "", ""
    try:
        resp = requests.get(job_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        blocks = re.findall(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
        )
        for block in blocks:
            try:
                data = json.loads(block.strip())
                if data.get("@type") != "JobPosting":
                    continue
                description = _strip_html(data.get("description", "")).strip()
                # addressLocality is typically "US - State (City - Office)"
                addr = (data.get("jobLocation") or {}).get("address") or {}
                raw_locality = addr.get("addressLocality", "")
                location = _parse_locality(raw_locality)
                return description, location
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception:
        pass
    return "", ""


def _parse_locality(locality: str) -> str:
    """
    Convert Workday's verbose addressLocality to a simple city, state string.
    e.g. "US - California (San Diego - Office)" → "San Diego, CA"
         "US - Massachusetts (Acton)" → "Acton, MA"
    Falls back to raw string if pattern doesn't match.
    """
    _STATE_ABBR = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
        "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
        "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
        "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
        "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    }
    # Pattern: "US - {State} ({City} - {suffix})" or "US - {State} ({City})"
    m = re.match(r'US\s*-\s*([^(]+)\s*\(([^)]+)\)', locality)
    if m:
        state_raw = m.group(1).strip().lower()
        city_raw = m.group(2).strip()
        # Strip trailing " - Office" or similar suffix from city
        city = re.sub(r'\s*-\s*\w+\s*$', '', city_raw).strip()
        abbr = _STATE_ABBR.get(state_raw)
        if abbr:
            return f"{city}, {abbr}"
    # Unrecognized format (e.g. internal office code like "WAP C") — return
    # empty so geo_match is treated as unknown rather than a false mismatch.
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
