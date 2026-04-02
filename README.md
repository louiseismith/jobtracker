# Job Tracker

CLI-first job monitoring tool for tracking company job boards, scoring roles with AI, and managing application decisions.

## What It Does

- Monitors companies on Greenhouse, Lever, Ashby, Workable, Workday, and Symbotic job boards
- Stores jobs in SQLite with deduplication and lifecycle tracking
- Scores jobs against your resume + preferences using cloud models
- Produces a concise daily briefing of high-scoring roles
- Lets you mark jobs as `applied` or `skipped` from the CLI

## Prerequisites

- `uv` installed for Python/dependency management
- Cloud API key for your selected provider/model:
  - `OPENAI_API_KEY` for `gpt-*` / `o*` models
  - `OLLAMA_API_KEY` for Ollama Cloud models
  - `ANTHROPIC_API_KEY` for `claude-*` models (used by `ask`/reporter if configured)
- macOS/Linux shell commands below (adapt as needed for Windows)

## Setup

1. Create a virtual environment with `uv`:
```bash
uv venv
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Add your resume:
- Place either `config/resume.docx` (preferred) or `config/resume.pdf`

4. Add preferences context (recommended):
- Create `config/preferences.md` with role/industry/location preferences and examples

5. Configure model settings:
- Edit `config/preferences.yaml`
- Scoring path (`score`, `rescore`, and the scorer inside `refresh`) uses:
  - `model`
  - `provider` (`openai` or `ollama`)
- `ask` command uses:
  - `ask_model` (defaults to `model` if omitted)
  - `ask_provider` (defaults to `provider` if omitted)
- Reporter inside `refresh` uses:
  - `reporter_model` (defaults to `model` if omitted)
  - `reporter_provider` (defaults to `provider` if omitted)
- `ask`/`reporter` providers supported: `openai`, `anthropic`, `ollama`

6. Set API keys in `.env` (depending on model)
- Create `.env` in the repo root:
```env
OPENAI_API_KEY=your_key_here
# OLLAMA_API_KEY=your_ollama_cloud_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

## CLI Usage

All commands are run from the project root:

```bash
uv run main.py <command>
```

### Commands

- `uv run main.py add <url> [--ats-type TYPE] [--ats-id SLUG] [--title-filter "kw1,kw2,..."]`
  - Detect ATS from a careers URL and add a company to monitoring.
  - `--ats-type` / `--ats-id`: override flags for when auto-detection misses the ATS type or slug (e.g. JS-rendered pages). Use with Ashby companies whose careers page doesn't expose the slug in the URL.
  - `--title-filter`: comma-separated keywords; only jobs matching at least one keyword in their title or description will be scored. Useful for large companies where you only want to see specific role types.
  - Example: `uv run main.py add https://jobs.lever.co/openai`
  - Example: `uv run main.py add https://formenergy.com/careers/ --ats-type ashby --ats-id formenergy`
  - Example: `uv run main.py add https://www.ll.mit.edu/careers --title-filter "systems engineer,integration,test engineer"`

- `uv run main.py add-file <path>`
  - Bulk add companies from a text file (one entry per line).
  - Supported line formats:
    - `Company Name | https://careers-url`
    - `Company Name, https://careers-url`
    - `https://careers-url` (name auto-derived)
  - Blank lines and `#` comments are ignored.

- `uv run main.py export-companies <path>`
  - Export monitored companies to a text file compatible with `add-file`.
  - Output format is one line per company: `Company Name | https://careers-url`

- `uv run main.py list`
  - List monitored companies and ATS info. Shows active title filter if one is set.

- `uv run main.py set-ats <company_id> <ats_type> [<ats_id>]`
  - Manually update ATS type/slug for an existing company.

- `uv run main.py set-url <company_id> <url>`
  - Update careers URL for an existing company.

- `uv run main.py set-title-filter <company_id> ["kw1,kw2,..."]`
  - Set or update the title filter for an existing company.
  - Omit the keyword argument to clear the filter entirely.
  - Example: `uv run main.py set-title-filter 5 "systems engineer,integration"`
  - Example: `uv run main.py set-title-filter 5` (clears filter)

- `uv run main.py refresh [--no-brief]`
  - Run the full pipeline: fetch jobs -> score new jobs -> generate briefing.
  - Use `--no-brief` to skip Reporter output.

- `uv run main.py status`
  - Show job pipeline summary and application state counts.

- `uv run main.py jobs`
  - List high-scoring open jobs (`score >= 6`) that are still `new`.

- `uv run main.py jobs-all`
  - List all jobs sorted by score (high to low, unscored last). Useful for testing.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-pipeline`
  - List all jobs in pipeline states (`applied`, `interviewing`, `rejected`, `offer`).

- `uv run main.py jobs-applied`
  - List jobs currently marked as `applied`.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-skipped`
  - List jobs currently marked as `skipped`.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py job <job_id> [--full]`
  - Show one job posting with detailed scoring metadata, component scores, filter flags, rationale, and description.
  - Use `--full` to print the complete description body.

- `uv run main.py ask <question>`
  - Ask a natural-language question; the agent uses tool-calling to retrieve matching jobs and detailed posting context.
  - Response is a direct answer with job summaries and fit assessment (not just a raw table).
  - Example: `uv run main.py ask "What are my best systems roles in fusion near Boston?"`

- `uv run main.py score <job_id>`
  - Manually score one job by database id.

- `uv run main.py rescore all [--model MODEL]`
  - Rescore all jobs in the database that pass pre-filter eligibility.
  - Jobs excluded by pre-filters are labeled as intentionally unscored with reasons.
  - Example: `uv run main.py rescore all --model gpt-4o-mini`

- `uv run main.py rescore <job_id|range> ... [--model MODEL]`
  - Rescore selected job IDs (supports ranges like `50-60`).
  - Example: `uv run main.py rescore 12 18 42 --model llama3.2`

- `uv run main.py doctor [--model MODEL]`
  - Run preflight checks for model/key/runtime readiness before scoring.
  - Example: `uv run main.py doctor --model gpt-4o`
  - Uses configured `provider` unless a `--model` override is passed (then provider is inferred).

- `uv run main.py apply <job_id|range> [...]`
  - Mark one or more jobs as `applied` (ranges supported, e.g. `42-50`).

- `uv run main.py skip <job_id|range> [...]`
  - Mark one or more jobs as `skipped` (reviewed but not applying).

- `uv run main.py review <job_id|range> [...]`
  - Mark one or more jobs as `reviewed` (seen, undecided).

- `uv run main.py interview <job_id|range> [...]`
  - Mark one or more jobs as `interviewing`.

- `uv run main.py reject <job_id|range> [...]`
  - Mark one or more jobs as `rejected`.

- `uv run main.py offer <job_id|range> [...]`
  - Mark one or more jobs as `offer`.

- `uv run main.py reset <job_id|range> [...]`
  - Reset one or more jobs back to `new`.

- `uv run main.py jobs-new`
  - List jobs currently in `new` state.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-reviewed`
  - List jobs currently in `reviewed` state.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-interviewing`
  - List jobs currently in `interviewing` state.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-rejected`
  - List jobs currently in `rejected` state.
  - Optional: `--company <substring>` to filter by company name.

- `uv run main.py jobs-offer`
  - List jobs currently in `offer` state.
  - Optional: `--company <substring>` to filter by company name.

## Typical Workflow

1. Add a few companies:
```bash
uv run main.py add https://jobs.lever.co/cfsenergy
uv run main.py add https://boards.greenhouse.io/anthropic
```

2. Fetch + score + briefing:
```bash
uv run main.py refresh
```

3. Review top roles:
```bash
uv run main.py jobs
uv run main.py status
```

4. Update your decisions:
```bash
uv run main.py apply 12
uv run main.py skip 18
```

5. Re-score with a different model (optional):
```bash
uv run main.py rescore all --model gpt-4o-mini
```

## Running Tests

```bash
uv run -m pytest -q
```

Current suite covers:
- DB schema/CRUD behavior
- Fetchers and ATS detection
- Agent tools
- CLI smoke tests

## Notes

- Supported ATS fetchers today: `greenhouse`, `lever`, `ashby`, `workable`, `workday`, `symbotic`
- `scrape` is detected as fallback but is not implemented in fetchers
- SQLite DB file is created at `jobtracker.db` in project root
- Scoring commands (`score`, `rescore`, scorer in `refresh`) currently support `openai` and `ollama` providers
- `ask` and reporter can also use `anthropic` when configured
- For Ollama cloud models, set `OLLAMA_API_KEY` in `.env`
