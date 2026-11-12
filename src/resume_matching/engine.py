"""
Resume parsing, keyword/skill matching and scoring against a job description.

Two ways to use it:

- `match_score` / `score_resume`: one resume against one JD, 0-100, with
  the breakdown (semantic similarity, required skills matched, experience,
  education) and the matched / missing skills.
- `screen`: a pool of resumes against one JD. Returns three groups --
  ranked, below threshold, and excluded by a hard filter with the reason --
  because an empty shortlist should say whether nobody applied, nobody
  scored well enough, or the filters were too tight.

No framework, no database, no network. The similarity backend, the skill
vocabulary and the scoring numbers are all passed in (or defaulted).
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from resume_matching.config import ScoringConfig
from resume_matching.extraction import ResumeParseError, extract_text
from resume_matching.extractors import (
    effective_education_levels,
    find_age,
    find_cgpa,
    find_date_of_birth,
    find_education_levels,
    find_emails,
    find_experience,
    find_experience_years,
    find_phones,
    find_required_experience,
    highest_education_level,
)
from resume_matching.similarity import SimilarityBackend, cosine, get_backend
from resume_matching.skills import SkillGraph

__all__ = [
    "ExtractedProfile",
    "MatchResult",
    "MatchScore",
    "ScreeningResult",
    "MatchingEngine",
    "ResumeParseError",
    "calibrate_semantic",
]


@dataclass
class ExtractedProfile:
    """What the engine pulls out of one resume or profile text."""

    raw_text: str
    experience_years: float
    education_levels: set[str] = field(default_factory=set)

    @property
    def effective_education_levels(self) -> set[str]:
        """Named levels plus the ones they imply (a master's implies a UG)."""
        return effective_education_levels(self.education_levels)

    @property
    def highest_education_level(self) -> str | None:
        return highest_education_level(self.education_levels)

    def as_dict(self) -> dict:
        return {
            "experience_years": self.experience_years,
            "education_levels": sorted(self.education_levels),
            "highest_education_level": self.highest_education_level,
            "text_length": len(self.raw_text),
        }


@dataclass
class MatchResult:
    """One candidate scored against one target text (screening)."""

    identifier: str
    similarity_score: float
    passed_threshold: bool
    extracted: ExtractedProfile
    #: Which hard filter removed this candidate, if any. Set means the
    #: similarity score was never computed.
    excluded_by: str | None = None

    @property
    def is_excluded(self) -> bool:
        return self.excluded_by is not None

    def as_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "similarity_score": self.similarity_score,
            "passed_threshold": self.passed_threshold,
            "excluded_by": self.excluded_by,
            **self.extracted.as_dict(),
        }


@dataclass
class ScreeningResult:
    """The outcome of scoring a pool of candidates against one target text."""

    threshold: float
    #: Passed every hard filter and met the threshold, best first.
    ranked: list[MatchResult] = field(default_factory=list)
    #: Passed the hard filters but scored below the threshold, best first.
    below_threshold: list[MatchResult] = field(default_factory=list)
    #: Removed by a hard filter before scoring.
    excluded: list[MatchResult] = field(default_factory=list)

    @property
    def considered(self) -> int:
        return len(self.ranked) + len(self.below_threshold) + len(self.excluded)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "considered": self.considered,
            "shortlisted": len(self.ranked),
            "below_threshold": len(self.below_threshold),
            "excluded": len(self.excluded),
            "ranked": [result.as_dict() for result in self.ranked],
            "below_threshold_results": [r.as_dict() for r in self.below_threshold],
            "excluded_results": [r.as_dict() for r in self.excluded],
        }


def calibrate_semantic(raw: float, source: str, calibration: Mapping | None = None) -> float:
    """
    A raw similarity on `source`'s scale, mapped onto 0-1.

    The sources do not share a scale: spaCy puts unrelated English near 0.8,
    TF-IDF near 0. Each has a FLOOR and CEILING in `calibration`
    (default: `ScoringConfig().calibration`).
    """
    if calibration is None:
        calibration = ScoringConfig().calibration
    bounds = calibration.get(source) or {"FLOOR": 0.0, "CEILING": 1.0}
    floor, ceiling = bounds["FLOOR"], bounds["CEILING"]
    if ceiling <= floor:
        return 1.0 if raw >= ceiling else 0.0
    return max(0.0, min(1.0, (raw - floor) / (ceiling - floor)))


@dataclass
class MatchScore:
    """
    One profile against one job description, 0-100, with its breakdown.

    A match, not a probability of fit: the breakdown always travels with the
    number, because a single figure looks more exact than it is.
    """

    score: int
    #: name -> {"score": 0-1, "weight": the share of the total it carried}.
    #: The semantic part also has "raw", the uncalibrated similarity.
    components: dict
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    #: Hard-filter reasons. Listed, never a reason to drop the score: the
    #: caller decides whether a flagged match is shown.
    hard_filter_failures: list[str] = field(default_factory=list)
    #: "embedding", or the similarity backend's name ("spacy" / "tfidf").
    semantic_source: str = ""

    @property
    def flagged(self) -> bool:
        return bool(self.hard_filter_failures)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "components": self.components,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "hard_filter_failures": self.hard_filter_failures,
            "flagged": self.flagged,
            "semantic_source": self.semantic_source,
        }


class MatchingEngine:
    """
    Parsing plus scoring.

    Build one and reuse it: a spaCy backend loads a model, which is the
    expensive part. Everything else is stateless.

    - `backend`: a `SimilarityBackend`, or a name ("auto", "spacy", "tfidf").
      Default "auto": spaCy if its model is installed, else TF-IDF.
    - `skills`: a `SkillGraph`. Default: the bundled vocabulary.
    - `config`: a `ScoringConfig` (weights, calibration, threshold).
    """

    def __init__(
        self,
        backend: SimilarityBackend | str | None = None,
        skills: SkillGraph | None = None,
        config: ScoringConfig | None = None,
        default_threshold: float | None = None,
    ):
        self._backend = get_backend(backend or "auto") if backend is None or isinstance(backend, str) else backend
        self.skills = skills if skills is not None else SkillGraph.default()
        self.config = config or ScoringConfig()
        # Explicit argument wins, then the configured value, then whatever
        # the live backend suggests for its own scale.
        if default_threshold is not None:
            self.default_threshold = default_threshold
        elif self.config.similarity_threshold is not None:
            self.default_threshold = self.config.similarity_threshold
        else:
            self.default_threshold = self._backend.suggested_threshold

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # --- parsing ------------------------------------------------------------

    def parse_text(self, text: str) -> ExtractedProfile:
        """Extract hard facts (experience, education) from resume text."""
        return ExtractedProfile(
            raw_text=text or "",
            experience_years=find_experience_years(text),
            education_levels=find_education_levels(text),
        )

    def parse_resume(self, file_obj, filename: str) -> ExtractedProfile:
        """
        Read a PDF or DOCX resume and extract its hard facts.

        Raises ResumeParseError for an unreadable or unsupported file.
        """
        return self.parse_text(extract_text(file_obj, filename))

    @staticmethod
    def regex_facts(text: str, today: datetime.date | None = None) -> dict:
        """Everything the regex extractors read, as JSON-ready values."""
        reading = find_experience(text, today)
        dob = find_date_of_birth(text)
        return {
            "experience_years": reading.years,
            "experience_source": reading.source,
            "education_levels": sorted(find_education_levels(text)),
            "age": find_age(text),
            "date_of_birth": dob.isoformat() if dob else None,
            "emails": find_emails(text),
            "phones": find_phones(text),
            "cgpa": find_cgpa(text),
        }

    # --- screening ----------------------------------------------------------

    def similarity(self, text_a: str, text_b: str) -> float:
        """Semantic similarity in [0, 1], rounded for stable comparisons."""
        return round(self._backend.similarity(text_a, text_b), 3)

    def score_one(
        self,
        identifier: str,
        candidate_text: str,
        target_text: str,
        *,
        similarity_threshold: float | None = None,
        min_experience_years: float | None = None,
        required_education: Iterable[str] | None = None,
    ) -> MatchResult:
        """
        Score one candidate against one target, applying the hard filters first.

        A candidate who fails a hard filter is excluded regardless of how
        well their text matches, and is never scored.
        """
        threshold = self.default_threshold if similarity_threshold is None else similarity_threshold
        extracted = self.parse_text(candidate_text)

        excluded_by = self._hard_filter_failure(extracted, min_experience_years, required_education)
        if excluded_by:
            return MatchResult(identifier, 0.0, False, extracted, excluded_by=excluded_by)

        score = self.similarity(candidate_text, target_text)
        return MatchResult(identifier, score, score >= threshold, extracted)

    @staticmethod
    def _hard_filter_failure(
        extracted: ExtractedProfile,
        min_experience_years: float | None,
        required_education: Iterable[str] | None,
    ) -> str | None:
        """The reason this candidate is excluded, or None if they pass."""
        if min_experience_years is not None and extracted.experience_years < min_experience_years:
            return (
                f"Needs {min_experience_years:g} years of experience; "
                f"resume shows {extracted.experience_years:g}."
            )

        if required_education:
            required = set(required_education)
            if not (required & extracted.effective_education_levels):
                held = extracted.highest_education_level or "none detected"
                return f"Needs {' or '.join(sorted(required))}; resume shows {held}."

        return None

    def screen(
        self,
        candidates: Mapping[str, str],
        target_text: str,
        *,
        similarity_threshold: float | None = None,
        min_experience_years: float | None = None,
        required_education: Iterable[str] | None = None,
    ) -> ScreeningResult:
        """
        Bulk screening: `{identifier: resume text}` against one job description.

        Ranking is by similarity, best first.
        """
        threshold = self.default_threshold if similarity_threshold is None else similarity_threshold
        result = ScreeningResult(threshold=threshold)

        for identifier, text in candidates.items():
            scored = self.score_one(
                identifier,
                text,
                target_text,
                similarity_threshold=threshold,
                min_experience_years=min_experience_years,
                required_education=required_education,
            )
            if scored.is_excluded:
                result.excluded.append(scored)
            elif scored.passed_threshold:
                result.ranked.append(scored)
            else:
                result.below_threshold.append(scored)

        result.ranked.sort(key=lambda item: item.similarity_score, reverse=True)
        result.below_threshold.sort(key=lambda item: item.similarity_score, reverse=True)
        result.excluded.sort(key=lambda item: item.identifier)
        return result

    # --- the 0-100 match score ---------------------------------------------------

    def job_requirements(self, jd_text: str) -> dict:
        """
        What the JD asks for: skills, minimum experience, education levels.

        Read from the text: known skill names and aliases through the skill
        graph, and the same regex extractors resumes go through. To use your
        own reading (an LLM, a form), build this dict yourself and pass it
        to `match_score(requirements=...)`.

        The minimum experience is set only by an explicit requirement: a
        JD's durations are mostly offer terms (an internship's length, a
        bond, the company's age) (`find_required_experience`).
        """
        return {
            "required_skills": self.skills.find_in_text(jd_text),
            "min_experience_years": find_required_experience(jd_text) or None,
            "education_levels": sorted(find_education_levels(jd_text)),
            "source": "text",
        }

    def match_score(
        self,
        profile: Mapping,
        jd_text: str,
        *,
        vector: list[float] | None = None,
        jd_vector: list[float] | None = None,
        requirements: Mapping | None = None,
    ) -> MatchScore:
        """
        Score one profile against one JD, 0-100, with the breakdown.

        `profile` is a plain dict: `text`, and optionally `skills` (a list of
        names), `experience_years`, `education_levels` ("UG", "Masters", "PG").
        Use `score_resume` to build it from resume text.

        - semantic: the embedding cosine when both vectors are given, else
          the similarity backend; calibrated per source to 0-1.
        - skills: the share of required skills held, through aliases and
          implications. Skills named in the profile text count as held.
        - experience: years held over the minimum, capped at 1.
        - education: 1 when a held level (or one it implies) is accepted.

        A requirement the JD does not state scores 1. A JD naming no known
        skill has no skill component, and its weight goes to the others.
        Hard-filter failures are listed and the score still computed.
        `requirements` lets a caller scoring many profiles read the JD once
        (`job_requirements`), or supply its own reading.
        """
        req = requirements or self.job_requirements(jd_text)
        text = profile.get("text") or ""
        held_levels = set(profile.get("education_levels") or [])
        years = float(profile.get("experience_years") or 0)

        if vector is not None and jd_vector is not None:
            source, raw = "embedding", round(cosine(vector, jd_vector), 4)
        else:
            source, raw = self.backend_name, self.similarity(text, jd_text)

        held = [*(profile.get("skills") or []), *self.skills.find_in_text(text)]
        matched, missing = self.skills.skill_overlap(req.get("required_skills") or [], held)
        required_count = len(matched) + len(missing)

        min_years = req.get("min_experience_years")
        experience = 1.0 if not min_years else min(1.0, years / min_years)
        accepted = set(req.get("education_levels") or [])
        education = 1.0 if not accepted or accepted & effective_education_levels(held_levels) else 0.0

        parts = {
            "semantic": calibrate_semantic(raw, source, self.config.calibration),
            "skills": len(matched) / required_count if required_count else 0.0,
            "experience": experience,
            "education": education,
        }
        weights = {name: float(self.config.weights.get(name, 0.0)) for name in parts}
        if not required_count:
            weights["skills"] = 0.0
        total_weight = sum(weights.values()) or 1.0
        components = {
            name: {"score": round(parts[name], 4), "weight": round(weights[name] / total_weight, 4)}
            for name in parts
        }
        components["semantic"]["raw"] = raw
        score = round(100 * sum(parts[name] * weights[name] for name in parts) / total_weight)

        extracted = ExtractedProfile(raw_text=text, experience_years=years, education_levels=held_levels)
        failures = [
            reason
            for reason in (
                self._hard_filter_failure(extracted, min_years, None),
                self._hard_filter_failure(extracted, None, accepted or None),
            )
            if reason
        ]
        return MatchScore(
            score=max(0, min(100, int(score))),
            components=components,
            matched_skills=matched,
            missing_skills=missing,
            hard_filter_failures=failures,
            semantic_source=source,
        )

    def score_resume(
        self,
        resume_text: str,
        jd_text: str,
        *,
        skills: Iterable[str] = (),
        requirements: Mapping | None = None,
    ) -> MatchScore:
        """
        `match_score` straight from resume text: experience and education are
        read by the regex extractors. `skills` adds any the resume lists
        that the text alone would not reveal.
        """
        extracted = self.parse_text(resume_text)
        return self.match_score(
            {
                "text": resume_text,
                "skills": list(skills),
                "experience_years": extracted.experience_years,
                "education_levels": sorted(extracted.education_levels),
            },
            jd_text,
            requirements=requirements,
        )
