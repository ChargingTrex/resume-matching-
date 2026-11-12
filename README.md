# resume-matching

Match and score resumes against a job description. Pure Python, no web
framework, no database, no network calls.

For each resume and job description (JD) you get a **0-100 score with its
breakdown**:

| part         | what it measures                                                             | default weight |
|--------------|-------------------------------------------------------------------------------|---------------:|
| `semantic`   | text similarity (TF-IDF, spaCy vectors, or your own embeddings), calibrated   | 0.45 |
| `skills`     | share of the JD's required skills the resume covers, through aliases and implications | 0.35 |
| `experience` | years of experience over the JD's stated minimum, capped at 1                | 0.10 |
| `education`  | whether a held degree level (UG / Masters / PG) is one the JD accepts         | 0.10 |

You also get `matched_skills`, `missing_skills`, and the `hard_filter_failures`
(for example "Needs 2 years of experience; resume shows 0.5."). A failure is
reported, but the score is still computed.

For bulk screening there's `screen`. It splits a pool into **ranked**,
**below threshold**, and **excluded by a hard filter (with the reason)**.

## Install

```bash
pip install "resume-matching[files] @ git+https://github.com/ChargingTrex/resume-matching-.git"
```

Extras:

- `files` reads PDF and DOCX resumes (pypdf, python-docx).
- `spacy` adds the spaCy similarity backend. It also needs a model with
  vectors: `python -m spacy download en_core_web_md`.

Without spaCy the engine uses TF-IDF, which needs no model download.

## Quick start

```python
from resume_matching import MatchingEngine

engine = MatchingEngine()   # spaCy if its model is installed, else TF-IDF

result = engine.score_resume(resume_text, jd_text)
result.score              # 72
result.matched_skills     # ['Python', 'Django', 'PostgreSQL']
result.missing_skills     # ['Docker', 'AWS']
result.components         # {'semantic': {'score': .., 'weight': .., 'raw': ..}, 'skills': {...}, ...}
result.hard_filter_failures
result.as_dict()          # JSON-ready
```

Read a file:

```python
from resume_matching import extract_text

with open("resume.pdf", "rb") as f:
    text = extract_text(f, "resume.pdf")     # raises ResumeParseError if unreadable
```

Screen a pool:

```python
result = engine.screen(
    {"alice": alice_text, "bob": bob_text, "carol": carol_text},
    jd_text,
    min_experience_years=2,         # hard filters (optional)
    required_education={"UG"},      # a Masters or PG satisfies UG
    similarity_threshold=None,      # None -> the backend's own default
)
[r.identifier for r in result.ranked]
[(r.identifier, r.excluded_by) for r in result.excluded]
```

## Command line

```bash
resume-match score resume.pdf jd.txt
resume-match screen jd.txt resumes/*.pdf --min-experience 2 --education UG
resume-match requirements jd.txt          # what the JD is read as asking for
resume-match --skills my_skills.json score resume.pdf jd.txt
```

Every command prints JSON.

## How it works

**Requirements from the JD.** `engine.job_requirements(jd_text)` reads three
things:

- **Required skills:** every known skill name or alias, matched on whole
  words.
- **Minimum experience:** only an explicit requirement counts ("2+ years of
  experience", "Minimum 3 years"). An internship's length, a service bond
  and the company's age are not requirements.
- **Accepted education levels.**

If you already have a better reading (from an LLM, or a form), pass it in
directly:

```python
engine.match_score(profile, jd_text, requirements={
    "required_skills": ["Python", "Django"],
    "min_experience_years": 2,
    "education_levels": ["UG"],
})
```

**Facts from the resume.** The regex extractors in `resume_matching.extractors`
read years of experience and education level. They handle the usual traps:

- "Age: 22 years" and "4 years B.Tech programme" are not experience.
- Date ranges like "Jun 2023 - Present" are summed, with overlaps merged.
- "MS Office" is not a master's degree, and "be" is not a B.E.

They also pull out age, date of birth, emails, phones and CGPA, through
`engine.regex_facts(text)`.

**Skills.** A `SkillGraph` maps aliases to canonical names ("Postgres" becomes
PostgreSQL, "k8s" becomes Kubernetes). It also knows implications: Django
implies Python, and PostgreSQL implies SQL. Coverage runs one way only.
Holding Django meets a Python requirement, but holding Python does not meet
a Django one. C, C++, C#, F# and .NET stay five different skills.

The bundled vocabulary is about 40 common tech skills
(`src/resume_matching/data/skills.json`). You can extend it or replace it:

```python
from resume_matching import MatchingEngine, SkillGraph

graph = SkillGraph.default().extended([
    # (canonical name, aliases, implied skills)
    ("Rust", ["rustlang"], []),
    ("Tokio", [], ["Rust"]),
])
graph = SkillGraph.from_json("my_skills.json")   # {"skills": [{"name", "aliases", "implies"}]}

engine = MatchingEngine(skills=graph)
```

**Semantic similarity.** You can choose the backend:

- `MatchingEngine(backend="tfidf")`
- `backend="spacy"`
- `backend="auto"`
- any object with `name`, `suggested_threshold` and
  `similarity(a, b) -> float in [0, 1]`.

The backends don't share a scale. spaCy puts unrelated English text near
0.8, while TF-IDF puts it near 0. So each raw score is calibrated onto 0-1
with a per-source floor and ceiling before it is weighted.

To use your own embeddings, pass the two vectors:

```python
engine.match_score(profile, jd_text, vector=resume_vec, jd_vector=jd_vec)
```

The semantic part is then their cosine, calibrated with the `"embedding"`
bounds. Recalibrate those bounds for your model.

**Tuning.**

```python
from resume_matching import ScoringConfig

engine = MatchingEngine(config=ScoringConfig(
    weights={"semantic": 0.3, "skills": 0.5, "experience": 0.1, "education": 0.1},
    calibration={"tfidf": {"FLOOR": 0.0, "CEILING": 0.25}, "embedding": {"FLOOR": 0.3, "CEILING": 0.8}},
    similarity_threshold=0.12,     # for screen()
))
```

Weights don't need to sum to 1. If a JD names no known skill, the skills
weight is shared out among the other parts.

## A score is not a verdict

The number is a match, not a probability of fit, and it can only see what
the text says. That's why the breakdown always travels with the score. Use
it to sort and shortlist, and leave the hiring decision to a person.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

The tests run against a hand-checked sample set (`resume_matching.samples`).
Each sample states the experience, education, age and CGPA a person read off
it, and the tests check that relevant resumes outrank irrelevant ones on
every backend. Tests that need spaCy are skipped when it isn't installed.
