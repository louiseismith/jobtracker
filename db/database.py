"""
db/database.py

SQLite connection and CRUD helpers.
All timestamps are stored as ISO 8601 UTC strings.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH     = Path(__file__).parent.parent / "jobtracker.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        _apply_schema_migrations(conn)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _apply_schema_migrations(conn: sqlite3.Connection):
    """Best-effort additive schema migrations for existing DB files."""
    if not _column_exists(conn, "jobs", "ai_components"):
        conn.execute("ALTER TABLE jobs ADD COLUMN ai_components TEXT")
    if not _column_exists(conn, "jobs", "exclude_flag"):
        conn.execute("ALTER TABLE jobs ADD COLUMN exclude_flag INTEGER NOT NULL DEFAULT 0")
    _migrate_legacy_exclude_flags(conn)
    if not _column_exists(conn, "companies", "title_filter"):
        conn.execute("ALTER TABLE companies ADD COLUMN title_filter TEXT")
    _migrate_companies_drop_ats_check(conn)


def _migrate_legacy_exclude_flags(conn: sqlite3.Connection):
    """
    Carry forward any legacy *_flag columns into jobs.exclude_flag.
    This preserves prior exclusion labeling after schema/key renames.
    """
    columns = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
    legacy_flag_cols = [c for c in columns if c.endswith("_flag") and c != "exclude_flag"]
    for col in legacy_flag_cols:
        conn.execute(
            f"""
            UPDATE jobs
            SET exclude_flag = CASE
                WHEN COALESCE(exclude_flag, 0) = 1 OR COALESCE({col}, 0) = 1 THEN 1
                ELSE 0
            END
            """
        )


def _migrate_companies_drop_ats_check(conn: sqlite3.Connection):
    """
    Rebuild companies table to remove the hardcoded ats_type CHECK constraint.
    Needed when new ATS types (e.g. workable) are added after the DB was created.
    No-op if the constraint is already gone.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='companies'"
    ).fetchone()
    if not row or "CHECK" not in (row["sql"] or ""):
        return  # constraint already removed, nothing to do

    conn.executescript("""
        PRAGMA foreign_keys=OFF;
        CREATE TABLE companies_new (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            careers_url  TEXT NOT NULL,
            ats_type     TEXT NOT NULL,
            ats_id       TEXT,
            active       INTEGER NOT NULL DEFAULT 1,
            date_added   TEXT NOT NULL,
            title_filter TEXT
        );
        INSERT INTO companies_new
            SELECT id, name, careers_url, ats_type, ats_id, active, date_added, title_filter
            FROM companies;
        DROP TABLE companies;
        ALTER TABLE companies_new RENAME TO companies;
        CREATE INDEX IF NOT EXISTS idx_companies_active ON companies(active);
        PRAGMA foreign_keys=ON;
    """)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

def upsert_company(
    name: str,
    careers_url: str,
    ats_type: str,
    ats_id: str | None,
    title_filter: list[str] | None = None,
) -> int:
    """Insert a new company and return its id. Does not update existing rows."""
    title_filter_json = json.dumps(title_filter) if title_filter else None
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO companies (name, careers_url, ats_type, ats_id, active, date_added, title_filter)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (name, careers_url, ats_type, ats_id, now_utc(), title_filter_json),
        )
        return cur.lastrowid


def set_company_title_filter(company_id: int, patterns: list[str] | None):
    """Set or clear the per-company title/description include filter."""
    title_filter_json = json.dumps(patterns) if patterns else None
    with get_connection() as conn:
        conn.execute(
            "UPDATE companies SET title_filter = ? WHERE id = ?",
            (title_filter_json, company_id),
        )


def get_active_companies() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM companies WHERE active = 1"
        ).fetchall()


def get_company_by_name(name: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM companies WHERE name = ?", (name,)
        ).fetchone()


def set_company_ats(company_id: int, ats_type: str, ats_id: str | None):
    with get_connection() as conn:
        conn.execute(
            "UPDATE companies SET ats_type = ?, ats_id = ? WHERE id = ?",
            (ats_type, ats_id, company_id),
        )


def set_company_url(company_id: int, url: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE companies SET careers_url = ? WHERE id = ?",
            (url, company_id),
        )


def set_company_active(company_id: int, active: bool):
    with get_connection() as conn:
        conn.execute(
            "UPDATE companies SET active = ? WHERE id = ?",
            (1 if active else 0, company_id),
        )


def list_companies() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM companies ORDER BY name").fetchall()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def upsert_job(
    company_id: int,
    external_id: str | None,
    title: str,
    url: str | None,
    description: str | None,
    location: str | None,
    date_posted: str | None,
    exclude_flag: int = 0,
    geo_match: int | None = None,
    title_excluded: int = 0,
) -> tuple[int, bool]:
    """
    Insert job if not already seen. Returns (job_id, is_new).
    Deduplication key: (company_id, external_id).
    For scraped jobs with no external_id, always inserts (no dedup).
    """
    with get_connection() as conn:
        if external_id is not None:
            existing = conn.execute(
                "SELECT id FROM jobs WHERE company_id = ? AND external_id = ?",
                (company_id, external_id),
            ).fetchone()
            if existing:
                return existing["id"], False

        cur = conn.execute(
            """
            INSERT INTO jobs (
                company_id, external_id, title, url, description, location,
                date_found, date_posted, job_state, application_state,
                exclude_flag, geo_match, title_excluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', 'new', ?, ?, ?)
            """,
            (
                company_id, external_id, title, url, description, location,
                now_utc(), date_posted,
                exclude_flag, geo_match, title_excluded,
            ),
        )
        return cur.lastrowid, True


def _auto_review_threshold() -> float:
    try:
        import yaml
        from pathlib import Path
        prefs = yaml.safe_load((Path(__file__).parent.parent / "config" / "preferences.yaml").read_text())
        return float(prefs.get("auto_review_below", 7.0))
    except Exception:
        return 7.0


def update_job_score(
    job_id: int,
    score: float,
    rationale: str,
    model_name: str,
    prompt_version: str,
    components: dict | None = None,
):
    threshold = _auto_review_threshold()
    # Only auto-review jobs still in 'new' state — don't override deliberate decisions.
    new_app_state = "reviewed" if score < threshold else None
    with get_connection() as conn:
        if new_app_state:
            conn.execute(
                """
                UPDATE jobs
                SET ai_score=?, ai_rationale=?, ai_components=?, model_name=?, prompt_version=?,
                    scored_at=?, application_state=?
                WHERE id=? AND application_state='new'
                """,
                (
                    score, rationale,
                    json.dumps(components) if components is not None else None,
                    model_name, prompt_version, now_utc(),
                    new_app_state, job_id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE jobs
                SET ai_score=?, ai_rationale=?, ai_components=?, model_name=?, prompt_version=?, scored_at=?
                WHERE id=?
                """,
                (
                    score, rationale,
                    json.dumps(components) if components is not None else None,
                    model_name, prompt_version, now_utc(),
                    job_id,
                ),
            )


def mark_job_unscored(
    job_id: int,
    rationale: str,
    model_name: str,
    prompt_version: str,
):
    """
    Mark a job as intentionally unscored (ai_score=NULL) with an explanatory rationale.
    Useful when a pre-filter excludes the role from scoring.
    """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET ai_score=NULL, ai_rationale=?, ai_components=NULL, model_name=?, prompt_version=?,
                scored_at=?, title_excluded=1
            WHERE id=?
            """,
            (rationale, model_name, prompt_version, now_utc(), job_id),
        )


def update_job_states(
    job_id: int,
    job_state: str | None = None,
    application_state: str | None = None,
):
    with get_connection() as conn:
        if job_state is not None:
            conn.execute(
                "UPDATE jobs SET job_state=? WHERE id=?", (job_state, job_id)
            )
        if application_state is not None:
            conn.execute(
                "UPDATE jobs SET application_state=? WHERE id=?",
                (application_state, job_id),
            )


def mark_missing_jobs(company_id: int, current_external_ids: set[str]):
    """
    Mark jobs as 'disappeared' when they no longer appear in a successful fetch.
    Only call this after a fetch succeeds — never on fetch failure.
    """
    with get_connection() as conn:
        open_jobs = conn.execute(
            "SELECT id, external_id FROM jobs WHERE company_id=? AND job_state='open' AND external_id IS NOT NULL",
            (company_id,),
        ).fetchall()
        for job in open_jobs:
            if job["external_id"] not in current_external_ids:
                conn.execute(
                    "UPDATE jobs SET job_state='disappeared' WHERE id=?",
                    (job["id"],),
                )


def get_new_unscored_jobs() -> list[sqlite3.Row]:
    """Jobs that are open, new, and haven't been scored yet."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT j.*, c.name as company_name
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.job_state = 'open'
              AND j.application_state = 'new'
              AND j.ai_score IS NULL
              AND j.title_excluded = 0
            ORDER BY j.date_found DESC
            """
        ).fetchall()


def get_briefing_jobs(min_score: float = 7.0) -> list[sqlite3.Row]:
    """High-scoring open jobs that haven't been acted on yet."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT j.*, c.name as company_name
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.job_state = 'open'
              AND j.application_state = 'new'
              AND j.ai_score >= ?
            ORDER BY j.ai_score DESC, j.date_found DESC
            """,
            (min_score,),
        ).fetchall()


def search_jobs(query: str) -> list[sqlite3.Row]:
    """FTS5 keyword search over job titles and descriptions."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT j.*, c.name as company_name
            FROM jobs_fts
            JOIN jobs j ON jobs_fts.rowid = j.id
            JOIN companies c ON j.company_id = c.id
            WHERE jobs_fts MATCH ?
            ORDER BY rank
            """,
            (query,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

def create_application(job_id: int, notes: str = "") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO applications (job_id, date_applied, notes, status_history)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, now_utc(), notes, json.dumps([])),
        )
        return cur.lastrowid


def get_application(job_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM applications WHERE job_id=?", (job_id,)
        ).fetchone()
