"""Tunable numbers for the 0-100 match score and screening."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ScoringConfig", "DEFAULT_WEIGHTS", "DEFAULT_CALIBRATION"]

#: How much each part of the match score counts. Need not sum to 1: the
#: score divides by the total, and a part the JD gives nothing to score
#: (no known skills) has its weight shared out among the rest.
DEFAULT_WEIGHTS = {"semantic": 0.45, "skills": 0.35, "experience": 0.10, "education": 0.10}

#: Each similarity source has its own scale, so each raw score is mapped onto
#: 0-1 by its own FLOOR (-> 0) and CEILING (-> 1) before it is weighted.
#:
#: spaCy and TF-IDF were measured on the samples in `resume_matching.samples`
#: against SAMPLE_JOB_DESCRIPTION (relevant ones marked *):
#:
#:     sample                spacy    tfidf
#:     backend_engineer *    0.964    0.237
#:     data_engineer *       0.926    0.102
#:     frontend_intern       0.847    0.018
#:     mechanical_designer   0.836    0.048
#:     marketing_executive   0.801    0.014
#:
#: "embedding" is for vectors you compute yourself (measured on
#: nomic-embed-text; provisional -- recalibrate for your model).
#: A source with no entry is used as-is (FLOOR 0, CEILING 1).
DEFAULT_CALIBRATION = {
    "embedding": {"FLOOR": 0.40, "CEILING": 0.85},
    "spacy": {"FLOOR": 0.80, "CEILING": 0.97},
    "tfidf": {"FLOOR": 0.00, "CEILING": 0.25},
}


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    calibration: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_CALIBRATION.items()})
    #: Screening's similarity threshold. None -> the backend's own suggestion,
    #: because spaCy and TF-IDF do not share a scale.
    similarity_threshold: float | None = None
