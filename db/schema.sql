-- db/schema.sql

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Companies being monitored
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    careers_url TEXT NOT NULL,
    ats_type    TEXT NOT NULL,
    ats_id      TEXT,           -- company slug used by the ATS API; NULL for scrape
    active      INTEGER NOT NULL DEFAULT 1,  -- 1 = monitored, 0 = paused
    date_added  TEXT NOT NULL   -- ISO 8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_companies_active ON companies(active);

-- Job postings discovered from monitored companies
CREATE TABLE IF NOT EXISTS jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id          INTEGER NOT NULL REFERENCES companies(id),
    external_id         TEXT,           -- ATS-assigned job ID; NULL for scraped jobs
    title               TEXT NOT NULL,
    url                 TEXT,
    description         TEXT,
    location            TEXT,
    date_found          TEXT NOT NULL,  -- ISO 8601 UTC
    date_posted         TEXT,           -- from ATS if available
    ai_score            REAL,           -- 0-10, set by Analyst agent
    ai_rationale        TEXT,           -- one-sentence explanation from agent
    ai_components       TEXT,           -- JSON object with component score breakdown
    model_name          TEXT,           -- Ollama model used for scoring
    prompt_version      TEXT,           -- scoring prompt version for reproducibility
    scored_at           TEXT,           -- ISO 8601 UTC
    job_state           TEXT NOT NULL DEFAULT 'open'
                            CHECK(job_state IN ('open', 'closed', 'disappeared')),
    application_state   TEXT NOT NULL DEFAULT 'new'
                            CHECK(application_state IN ('new', 'reviewed', 'skipped', 'applied', 'interviewing', 'rejected', 'offer')),
    exclude_flag        INTEGER NOT NULL DEFAULT 0,  -- 1 if configured exclusion signals are detected
    geo_match           INTEGER,        -- 1 if location passes geography filter, 0 if not, NULL if unknown
    title_excluded      INTEGER NOT NULL DEFAULT 0,  -- 1 if title matches an exclude rule

    UNIQUE(company_id, external_id)     -- prevents duplicates for ATS-backed jobs
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(job_state, application_state, ai_score);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_id);

-- Full-text search over job titles and descriptions
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    title,
    description,
    content='jobs',
    content_rowid='id'
);

-- Keep FTS index in sync with jobs table
CREATE TRIGGER IF NOT EXISTS jobs_fts_insert AFTER INSERT ON jobs BEGIN
    INSERT INTO jobs_fts(rowid, title, description)
    VALUES (new.id, new.title, new.description);
END;

CREATE TRIGGER IF NOT EXISTS jobs_fts_update AFTER UPDATE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, title, description)
    VALUES ('delete', old.id, old.title, old.description);
    INSERT INTO jobs_fts(rowid, title, description)
    VALUES (new.id, new.title, new.description);
END;

CREATE TRIGGER IF NOT EXISTS jobs_fts_delete AFTER DELETE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, title, description)
    VALUES ('delete', old.id, old.title, old.description);
END;

-- Application tracking (one row per job applied to)
CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          INTEGER NOT NULL REFERENCES jobs(id),
    date_applied    TEXT,           -- ISO 8601 UTC
    notes           TEXT,
    status_history  TEXT            -- JSON array of {status, date}
);
