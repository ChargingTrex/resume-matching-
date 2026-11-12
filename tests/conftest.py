"""Fixtures shared by the engine tests."""

from pathlib import Path

import pytest

from resume_matching.engine import MatchingEngine
from resume_matching.samples import SAMPLE_RESUMES, render_samples


@pytest.fixture(scope="session")
def sample_resume_dir(tmp_path_factory) -> Path:
    """Real PDF and DOCX files, rendered once for the whole session."""
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    out = tmp_path_factory.mktemp("resumes")
    render_samples(out)
    return out


@pytest.fixture(scope="session")
def engine() -> MatchingEngine:
    """
    The engine on whichever backend this machine has.

    Session-scoped because loading the spaCy model is slow and the engine is
    stateless once built.
    """
    return MatchingEngine()


@pytest.fixture(scope="session")
def sample_texts(engine, sample_resume_dir) -> dict[str, str]:
    """Every sample resume parsed out of its real binary file."""
    texts = {}
    for sample in SAMPLE_RESUMES:
        with (sample_resume_dir / sample.filename).open("rb") as handle:
            texts[sample.key] = engine.parse_resume(handle, sample.filename).raw_text
    return texts
