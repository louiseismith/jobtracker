"""
utils/resume_parser.py

Parses resume.docx (or resume.pdf) to plain text and caches the result.
Re-parses only when the source file is newer than the cache.
"""

import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
CACHE_PATH = CONFIG_DIR / "resume_cache.txt"
DOCX_PATH  = CONFIG_DIR / "resume.docx"
PDF_PATH   = CONFIG_DIR / "resume.pdf"


def _parse_docx(path: Path) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _source_path() -> Path:
    """Return whichever resume file exists, preferring DOCX."""
    if DOCX_PATH.exists():
        return DOCX_PATH
    if PDF_PATH.exists():
        return PDF_PATH
    raise FileNotFoundError(
        f"No resume found. Place resume.docx or resume.pdf in {CONFIG_DIR}"
    )


def get_resume_text(force: bool = False) -> str:
    """
    Return plain-text resume contents.
    Reads from cache unless the source file is newer or force=True.
    """
    source = _source_path()

    # Use cache if it exists and is up to date
    if not force and CACHE_PATH.exists():
        if CACHE_PATH.stat().st_mtime >= source.stat().st_mtime:
            return CACHE_PATH.read_text(encoding="utf-8")

    # Parse
    suffix = source.suffix.lower()
    if suffix == ".docx":
        text = _parse_docx(source)
    elif suffix == ".pdf":
        text = _parse_pdf(source)
    else:
        raise ValueError(f"Unsupported resume format: {suffix}")

    # Write cache
    CACHE_PATH.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    text = get_resume_text(force=True)
    print(f"Parsed {len(text)} characters from {_source_path().name}")
    print("---")
    print(text[:500])
