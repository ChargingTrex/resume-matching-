"""
Plain text out of a resume file (PDF or DOCX).

A file that cannot be read raises `ResumeParseError` rather than returning an
empty string, so a corrupt upload is distinguishable from a blank page.

Needs the `files` extra: `pip install "resume-matching[files]"`.
"""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

__all__ = ["ResumeParseError", "extract_text", "extract_text_from_pdf", "extract_text_from_docx"]


class ResumeParseError(Exception):
    """The file could not be read as a resume."""


def _normalise_whitespace(text: str) -> str:
    """
    Collapse the ragged whitespace PDF extraction produces.

    Layout-driven extraction emits runs of spaces and stray line breaks mid
    sentence; left alone they break the regex extractors ("2  years") and add
    noise to the similarity vectors.
    """
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _require(module: str, package: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"Reading resume files needs {package}. "
            f'Install it with: pip install "resume-matching[files]"'
        ) from exc


def extract_text_from_pdf(file_obj) -> str:
    pypdf = _require("pypdf", "pypdf")
    try:
        reader = pypdf.PdfReader(file_obj)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ResumeParseError(f"Could not read the PDF: {exc}") from exc
    # Page breaks become blank lines rather than joining two sentences.
    return _normalise_whitespace("\n".join(pages))


def extract_text_from_docx(file_obj) -> str:
    docx = _require("docx", "python-docx")
    try:
        document = docx.Document(file_obj)
        blocks = [paragraph.text for paragraph in document.paragraphs]
        # Tables are common in resumes (skills grids, education tables) and
        # python-docx does not include them in `paragraphs`.
        for table in document.tables:
            for row in table.rows:
                blocks.extend(cell.text for cell in row.cells)
    except Exception as exc:
        raise ResumeParseError(f"Could not read the DOCX file: {exc}") from exc
    return _normalise_whitespace("\n".join(blocks))


def extract_text(file_obj, filename: str) -> str:
    """Dispatch on the file extension. Raises ResumeParseError for anything unsupported."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_obj)
    elif suffix == ".docx":
        text = extract_text_from_docx(file_obj)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ResumeParseError(
            f"Unsupported resume file type '{suffix or filename}'. Supported: {supported}."
        )

    if not text.strip():
        raise ResumeParseError(
            "No text could be extracted. If this is a scanned resume, use a "
            "text-based PDF or a DOCX instead."
        )
    return text
