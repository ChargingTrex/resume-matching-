"""
resume_matching: match and score resumes against a job description.

    from resume_matching import MatchingEngine

    engine = MatchingEngine()                     # TF-IDF, or spaCy if installed
    result = engine.score_resume(resume_text, jd_text)
    result.score            # 0-100
    result.matched_skills   # ["Python", "PostgreSQL", ...]
    result.missing_skills
    result.components       # semantic / skills / experience / education
"""

from resume_matching.config import ScoringConfig
from resume_matching.engine import (
    ExtractedProfile,
    MatchingEngine,
    MatchResult,
    MatchScore,
    ScreeningResult,
    calibrate_semantic,
)
from resume_matching.extraction import ResumeParseError, extract_text
from resume_matching.similarity import SimilarityBackend, SpacyBackend, TfidfBackend, cosine, get_backend
from resume_matching.skills import SkillGraph, slugify_skill

__version__ = "0.1.0"

__all__ = [
    "MatchingEngine",
    "MatchScore",
    "MatchResult",
    "ScreeningResult",
    "ExtractedProfile",
    "ScoringConfig",
    "SkillGraph",
    "SimilarityBackend",
    "SpacyBackend",
    "TfidfBackend",
    "ResumeParseError",
    "calibrate_semantic",
    "cosine",
    "extract_text",
    "get_backend",
    "slugify_skill",
]
