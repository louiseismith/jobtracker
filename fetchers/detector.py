"""
fetchers/detector.py

Given a company careers URL, detect which ATS it uses and extract the company slug.
Returns (ats_type, ats_id) — e.g. ("greenhouse", "acmecorp") or ("scrape", None).
"""

import re
import requests

TIMEOUT = 10


def detect(careers_url: str) -> tuple[str, str | None]:
    """
    Detect ATS type and company slug from a careers page URL.

    Strategy:
    1. Check the URL itself for known ATS patterns.
    2. If not conclusive, fetch the page and check for ATS signals in the HTML.
    3. Fall back to ("scrape", None).
    """
    ats_type, ats_id = _detect_from_url(careers_url)
    if ats_type != "scrape":
        return ats_type, ats_id

    # URL didn't match — try fetching the page
    try:
        resp = requests.get(careers_url, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        final_url = resp.url
    except Exception:
        return "scrape", None

    # Re-check the final URL after redirects
    ats_type, ats_id = _detect_from_url(final_url)
    if ats_type != "scrape":
        return ats_type, ats_id

    # Check page source for embedded ATS signals
    return _detect_from_html(html, careers_url)


def _detect_from_url(url: str) -> tuple[str, str | None]:
    """Check URL string for known ATS patterns."""

    # Greenhouse: boards.greenhouse.io/acmecorp or job-boards.greenhouse.io/acmecorp
    m = re.search(r'greenhouse\.io/([^/?#]+)', url)
    if m:
        return "greenhouse", m.group(1)

    # Lever: jobs.lever.co/acmecorp
    m = re.search(r'lever\.co/([^/?#]+)', url)
    if m:
        return "lever", m.group(1)

    # Ashby: jobs.ashbyhq.com/acmecorp
    m = re.search(r'jobs\.ashbyhq\.com/([^/?#]+)', url)
    if m:
        return "ashby", m.group(1)

    # Workable: apply.workable.com/acmecorp
    m = re.search(r'apply\.workable\.com/([^/?#]+)', url)
    if m:
        return "workable", m.group(1)

    # Workday: {tenant}.wd{n}.myworkdayjobs.com/{board}
    m = re.search(r'([\w-]+\.wd\d+)\.myworkdayjobs\.com/([^/?#]+)', url)
    if m:
        return "workday", f"{m.group(1)}|{m.group(2)}"

    # Symbotic custom careers page
    if "symbotic.com/careers" in url:
        return "symbotic", None

    return "scrape", None


def _detect_from_html(html: str, original_url: str) -> tuple[str, str | None]:
    """Scan page HTML for embedded ATS iframes or API calls."""

    # Greenhouse embedded board
    m = re.search(r'greenhouse\.io/embed/job_board\?for=([^&"\']+)', html)
    if m:
        return "greenhouse", m.group(1)

    m = re.search(r'boards\.greenhouse\.io/([^/?#"\']+)', html)
    if m:
        return "greenhouse", m.group(1)

    # Lever embedded board
    m = re.search(r'jobs\.lever\.co/([^/?#"\']+)', html)
    if m:
        return "lever", m.group(1)

    # Ashby embedded board — match jobs.ashbyhq.com/slug or job-board/{slug}
    m = re.search(r'jobs\.ashbyhq\.com/([^/?#"\']+)', html)
    if m:
        return "ashby", m.group(1)
    m = re.search(r'ashbyhq\.com/posting-api/job-board/([^/?#"\']+)', html)
    if m:
        return "ashby", m.group(1)

    # Workable embedded board
    m = re.search(r'apply\.workable\.com/([^/?#"\']+)', html)
    if m:
        return "workable", m.group(1)

    # Workday embedded board
    m = re.search(r'([\w-]+\.wd\d+)\.myworkdayjobs\.com/([^/?#"\']+)', html)
    if m:
        return "workday", f"{m.group(1)}|{m.group(2)}"

    return "scrape", None


if __name__ == "__main__":
    test_urls = [
        "https://www.commonsense.org/education/careers#",
        "https://jobs.lever.co/openai",
        "https://boards.greenhouse.io/anthropic",
        "https://job-boards.greenhouse.io/confluentinc",
    ]
    for url in test_urls:
        result = detect(url)
        print(f"{url}\n  → {result}\n")
