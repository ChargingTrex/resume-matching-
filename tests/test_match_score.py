"""
The 0-100 match score against a job description .

The requirements are read from the JD text through the skill graph and the
regex extractors, unless the caller passes its own `requirements`.
"""

import pytest

from resume_matching.config import DEFAULT_CALIBRATION
from resume_matching.engine import MatchingEngine, MatchScore, calibrate_semantic
from resume_matching.samples import RELEVANT_KEYS, SAMPLE_JOB_DESCRIPTION, SAMPLE_RESUMES
from resume_matching.similarity import cosine, get_backend
from resume_matching.skills import SkillGraph

from .helpers import fake_embedding

skills = SkillGraph.default()


def payload(text, skills_held=(), experience_years=0.0, education_levels=()):
    return {
        "text": text,
        "skills": list(skills_held),
        "experience_years": experience_years,
        "education_levels": list(education_levels),
    }


def sample_payload(sample):
    """A sample resume as the profile helper would hand it over."""
    return payload(
        sample.text,
        experience_years=sample.expected_experience_years,
        education_levels=sample.expected_education_levels,
    )


def _engine(name):
    try:
        backend = get_backend(name)
    except (ImportError, RuntimeError):
        pytest.skip(f"the {name} backend is not installed here")
    if backend.name != name:
        pytest.skip(f"the {name} backend is not installed here")
    return MatchingEngine(backend=backend)


TFIDF = MatchingEngine(backend="tfidf")


@pytest.fixture(params=["spacy", "tfidf"])
def each_engine(request):
    return _engine(request.param)


class TestFindInText:
    def test_canonical_names_and_aliases_are_found_word_bounded(self):
        found = skills.find_in_text("We use Postgres, Django REST Framework and k8s. Pythonic code.")
        assert found == ["PostgreSQL", "Django REST Framework", "Kubernetes"]

    def test_a_skill_inside_another_word_is_not_found(self):
        assert skills.find_in_text("Javascripting, gitlab-ish, Reactor cores") == []

    def test_dotted_and_spaced_spellings_resolve(self):
        assert skills.find_in_text("Node.js, node js and NodeJS services") == ["Node.js"]

    def test_cplusplus_and_csharp_are_two_skills_in_text(self):
        assert skills.find_in_text("C++ and C#") == ["C++", "C#"]

    def test_the_c_family_and_dotnet_are_found_apart(self):
        found = skills.find_in_text("We use C, C++, C#, F# and .NET (ASP.NET Core).")
        assert found == ["C", "C++", "C#", "F#", ".NET", "ASP.NET"]

    def test_a_leading_dot_that_is_not_part_of_a_skill_is_ignored(self):
        assert skills.find_in_text("Frameworks: ...React, .Docker") == ["React", "Docker"]

    def test_a_lowercase_c_is_a_list_marker_not_the_language(self):
        assert skills.find_in_text("a) Python b) Java c) Git") == ["Python", "Java", "Git"]

    def test_a_skill_that_is_also_an_english_word_needs_its_capital(self):
        assert skills.find_in_text("you will go and react to express needs") == []
        assert skills.find_in_text("Experience with Go and React") == ["Go", "React"]

    def test_the_sample_jd_names_the_backend_skills(self):
        found = skills.find_in_text(SAMPLE_JOB_DESCRIPTION)
        for name in ["Python", "Django", "Django REST Framework", "PostgreSQL", "SQL", "Git", "Docker", "AWS"]:
            assert name in found
        assert "React" not in found

    def test_with_an_empty_graph_nothing_is_found(self):
        assert SkillGraph().find_in_text(SAMPLE_JOB_DESCRIPTION) == []


class TestCalibration:
    def test_each_source_maps_its_own_floor_and_ceiling_to_zero_and_one(self):
        for source, bounds in DEFAULT_CALIBRATION.items():
            assert calibrate_semantic(bounds["FLOOR"], source) == 0.0
            assert calibrate_semantic(bounds["CEILING"], source) == 1.0

    def test_values_outside_the_range_are_clamped(self):
        assert calibrate_semantic(-1.0, "tfidf") == 0.0
        assert calibrate_semantic(5.0, "spacy") == 1.0

    def test_the_constants_can_be_overridden(self):
        calibration = {**DEFAULT_CALIBRATION, "tfidf": {"FLOOR": 0.2, "CEILING": 0.4}}
        assert calibrate_semantic(0.3, "tfidf", calibration) == pytest.approx(0.5)

    def test_an_unknown_source_is_used_as_is(self):
        assert calibrate_semantic(0.3, "my-model") == pytest.approx(0.3)


class TestMatchScore:
    def test_the_score_is_an_int_from_0_to_100_and_components_are_bounded(self, each_engine):
        for sample in SAMPLE_RESUMES:
            result = each_engine.match_score(sample_payload(sample), SAMPLE_JOB_DESCRIPTION)
            assert isinstance(result, MatchScore)
            assert isinstance(result.score, int) and 0 <= result.score <= 100
            for name in ("semantic", "skills", "experience", "education"):
                part = result.components[name]
                assert 0.0 <= part["score"] <= 1.0
                assert 0.0 <= part["weight"] <= 1.0
            assert sum(part["weight"] for part in result.components.values()) == pytest.approx(1.0, abs=1e-3)

    def test_the_relevant_samples_outrank_the_irrelevant_ones_on_both_backends(self, each_engine):
        scores = {
            sample.key: each_engine.match_score(sample_payload(sample), SAMPLE_JOB_DESCRIPTION).score
            for sample in SAMPLE_RESUMES
        }
        worst_relevant = min(scores[key] for key in RELEVANT_KEYS)
        best_irrelevant = max(score for key, score in scores.items() if key not in RELEVANT_KEYS)
        assert worst_relevant > best_irrelevant, scores

    def test_the_semantic_source_is_the_backend_without_vectors(self, each_engine):
        result = each_engine.match_score(payload("Python developer " * 5), SAMPLE_JOB_DESCRIPTION)
        assert result.semantic_source == each_engine.backend_name

    def test_with_both_vectors_the_semantic_score_is_the_embedding_cosine(self, engine):
        text = SAMPLE_RESUMES[0].text
        vector, jd_vector = fake_embedding(text), fake_embedding(SAMPLE_JOB_DESCRIPTION)
        result = engine.match_score(payload(text), SAMPLE_JOB_DESCRIPTION, vector=vector, jd_vector=jd_vector)
        assert result.semantic_source == "embedding"
        assert result.components["semantic"]["raw"] == pytest.approx(cosine(vector, jd_vector), abs=1e-3)
        assert result.components["semantic"]["score"] == pytest.approx(
            calibrate_semantic(cosine(vector, jd_vector), "embedding"), abs=1e-3
        )

    def test_the_skill_graph_is_used_postgresql_in_the_jd_meets_postgres_on_the_profile(self, engine):
        jd = "We need an engineer who knows PostgreSQL well, for our analytics database team."
        result = engine.match_score(payload("I build things.", skills_held=["Postgres"]), jd)
        assert result.matched_skills == ["PostgreSQL"]
        assert result.missing_skills == []
        assert result.components["skills"]["score"] == 1.0

    def test_a_jd_needing_cplusplus_is_not_met_by_csharp(self, engine):
        jd = "We need an engineer who writes modern C++ for our trading systems team."
        result = engine.match_score(payload("I write C# services.", skills_held=["C#"]), jd)
        assert result.matched_skills == []
        assert result.missing_skills == ["C++"]
        assert result.components["skills"]["score"] == 0.0

    def test_skills_named_only_in_the_profile_text_count_as_held(self, engine):
        jd = "We need an engineer who knows PostgreSQL well, for our analytics database team."
        result = engine.match_score(payload("Tuned Postgres queries for a year."), jd)
        assert result.matched_skills == ["PostgreSQL"]

    def test_missing_skills_are_listed_by_canonical_name(self, engine):
        result = engine.match_score(payload("Python scripts.", skills_held=["Python"]), SAMPLE_JOB_DESCRIPTION)
        assert "Python" in result.matched_skills
        assert {"Django", "PostgreSQL", "Docker"} <= set(result.missing_skills)

    def test_caller_requirements_override_what_the_text_says(self, engine):
        requirements = {"required_skills": ["Rust"], "min_experience_years": None, "education_levels": []}
        result = engine.match_score(payload("Rust and Python.", skills_held=["Rust"]), SAMPLE_JOB_DESCRIPTION,
                                    requirements=requirements)
        assert result.matched_skills == ["Rust"]
        assert result.hard_filter_failures == []

    def test_a_jd_naming_no_known_skill_gives_the_skill_weight_to_the_rest(self, engine):
        jd = "We are hiring a friendly person to greet visitors at our front desk every day."
        result = engine.match_score(payload("Friendly and punctual."), jd)
        assert result.components["skills"]["weight"] == 0.0
        assert result.components["skills"]["score"] == 0.0
        assert sum(part["weight"] for part in result.components.values()) == pytest.approx(1.0, abs=1e-3)

    def test_experience_is_scored_against_the_jds_minimum(self, engine):
        half = engine.match_score(payload("Python.", experience_years=1.0), SAMPLE_JOB_DESCRIPTION)
        full = engine.match_score(payload("Python.", experience_years=3.0), SAMPLE_JOB_DESCRIPTION)
        assert half.components["experience"]["score"] == pytest.approx(0.5)
        assert full.components["experience"]["score"] == 1.0

    def test_education_is_scored_against_the_levels_the_jd_accepts(self, engine):
        ug = engine.match_score(payload("Python.", education_levels=["Masters"]), SAMPLE_JOB_DESCRIPTION)
        none = engine.match_score(payload("Python."), SAMPLE_JOB_DESCRIPTION)
        assert ug.components["education"]["score"] == 1.0  # a master's implies a UG
        assert none.components["education"]["score"] == 0.0


class TestHardFilters:
    def test_a_failed_filter_is_listed_and_the_score_is_still_computed(self, engine):
        result = engine.match_score(
            payload(SAMPLE_RESUMES[0].text, experience_years=0.5, education_levels=["UG"]),
            SAMPLE_JOB_DESCRIPTION,
        )
        assert len(result.hard_filter_failures) == 1
        assert "Needs 2 years" in result.hard_filter_failures[0]
        assert result.score > 0
        assert result.as_dict()["flagged"] is True

    def test_both_filters_are_listed_when_both_fail(self, engine):
        result = engine.match_score(payload("Python."), SAMPLE_JOB_DESCRIPTION)
        assert len(result.hard_filter_failures) == 2

    def test_a_candidate_meeting_both_is_not_flagged(self, engine):
        result = engine.match_score(
            payload("Python.", experience_years=3, education_levels=["UG"]), SAMPLE_JOB_DESCRIPTION
        )
        assert result.hard_filter_failures == []
        assert result.as_dict()["flagged"] is False

    def test_the_filters_read_like_screening_does(self, engine):
        from resume_matching.engine import ExtractedProfile

        reason = engine._hard_filter_failure(ExtractedProfile(raw_text="", experience_years=0.5), 2.0, None)
        result = engine.match_score(payload("Python.", experience_years=0.5, education_levels=["UG"]),
                                    SAMPLE_JOB_DESCRIPTION)
        assert result.hard_filter_failures == [reason]


class TestAsDict:
    def test_the_dict_carries_the_breakdown_and_nothing_from_the_text(self, engine):
        data = engine.match_score(payload(SAMPLE_RESUMES[0].text), SAMPLE_JOB_DESCRIPTION).as_dict()
        assert set(data) == {
            "score", "components", "matched_skills", "missing_skills",
            "hard_filter_failures", "flagged", "semantic_source",
        }
        assert "Aarav" not in str(data)


INTERNSHIP_BOND_JD = "Summer internship for 6 months with a 3 year service bond. Must know Python and Django."


class TestJobRequirementsFromText:
    """
    A JD's minimum experience comes only from an explicit requirement.
    Company history, an internship's length and a service bond are
    durations, not requirements.
    """

    @pytest.mark.parametrize(
        "jd",
        [
            "Acme has 25 years of history in logistics. We need a Python developer with Django.",
            "Internship for 6 months. Stipend. Must know React.",
            "6-month internship, 3 year service bond",
            INTERNSHIP_BOND_JD,
            "Founded 30 years ago, Acme builds logistics software in Java.",
            "Acme brings 25 years of experience in logistics. We hire Python developers.",
            "Internship duration: 6 months of hands-on experience with Django.",
            "Drive runs Jan 2026 - Jun 2026. Python and Django developers wanted.",
        ],
    )
    def test_a_duration_that_is_not_a_requirement_sets_no_minimum(self, jd):
        assert not TFIDF.job_requirements(jd)["min_experience_years"]

    @pytest.mark.parametrize(
        ("jd", "years"),
        [
            ("Requires 2+ years of professional experience in Python", 2.0),
            ("Minimum 3 years experience", 3.0),
            ("Minimum 2 years of experience in Java", 2.0),
            ("At least 4 years of experience with Go.", 4.0),
            ("Candidates must have 5 years of relevant experience.", 5.0),
            ("Experience of 3 years in data engineering is required.", 3.0),
            ("3 years of backend experience required.", 3.0),
            ("We need someone with 2 years experience in React.", 2.0),
            ("Experience: minimum 2 years", 2.0),
        ],
    )
    def test_an_explicit_requirement_still_sets_the_minimum(self, jd, years):
        assert TFIDF.job_requirements(jd)["min_experience_years"] == years

    def test_the_sample_jd_still_asks_for_two_years(self):
        assert TFIDF.job_requirements(SAMPLE_JOB_DESCRIPTION)["min_experience_years"] == 2.0

    @pytest.mark.parametrize(
        "sample", [s for s in SAMPLE_RESUMES if s.expected_experience_years < 1], ids=lambda s: s.key
    )
    def test_freshers_are_not_hard_filtered_by_an_internship_with_a_bond(self, engine, sample):
        result = engine.match_score(
            payload(sample.text, ["Python", "Django"], sample.expected_experience_years,
                    sample.expected_education_levels),
            INTERNSHIP_BOND_JD,
        )
        assert result.hard_filter_failures == []
        assert result.components["experience"]["score"] == 1.0


class TestScoreResume:
    def test_score_resume_reads_experience_and_education_from_the_text(self, engine):
        backend = next(s for s in SAMPLE_RESUMES if s.key == "backend_engineer")
        direct = engine.score_resume(backend.text, SAMPLE_JOB_DESCRIPTION)
        via_payload = engine.match_score(sample_payload(backend), SAMPLE_JOB_DESCRIPTION)
        assert direct.score == via_payload.score
        assert direct.hard_filter_failures == []

    def test_custom_weights_change_the_score(self):
        from resume_matching.config import ScoringConfig

        only_skills = MatchingEngine(
            backend="tfidf",
            config=ScoringConfig(weights={"semantic": 0, "skills": 1, "experience": 0, "education": 0}),
        )
        jd = "We need an engineer who knows PostgreSQL and Docker well."
        result = only_skills.score_resume("Postgres expert.", jd)
        assert result.score == 50
        assert result.components["skills"]["weight"] == 1.0

    def test_a_custom_skill_graph_is_used(self):
        graph = SkillGraph.default().extended([("Rust", ["rustlang"], []), ("Tokio", [], ["Rust"])])
        engine = MatchingEngine(backend="tfidf", skills=graph)
        result = engine.score_resume("Built async services with Tokio.", "Must know Rust.")
        assert result.matched_skills == ["Rust"]
