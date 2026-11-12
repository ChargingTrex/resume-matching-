# resume-matching

Match resumes against a job description (JD) and get a **0-100 match
percentage** that explains itself. The result shows which skills matched,
which are missing, and why the number is what it is.

It's pure Python, with no web framework, no database and no network calls,
so you can drop it into any app, script or pipeline. It's free for colleges,
placement cells and personal projects, under the
[AGPL-3.0](#license) license.

```text
$ resume-match score data_engineer.pdf job_description.txt
{
  "score": 65,
  "components": {
    "semantic":   {"score": 0.408, "weight": 0.45, "raw": 0.102},
    "skills":     {"score": 0.75,  "weight": 0.35},
    "experience": {"score": 1.0,   "weight": 0.1},
    "education":  {"score": 1.0,   "weight": 0.1}
  },
  "matched_skills": ["Django", "PostgreSQL", "Python", "SQL", "Git", "Docker"],
  "missing_skills": ["Django REST Framework", "AWS"],
  "hard_filter_failures": [],
  "flagged": false,
  "semantic_source": "tfidf"
}
```

---

## Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [How the matching works](#how-the-matching-works)
5. [How the match percentage is calculated](#how-the-match-percentage-is-calculated)
6. [Bulk screening](#bulk-screening)
7. [Customising](#customising)
8. [Improving accuracy with embeddings and a vector database](#improving-accuracy-with-embeddings-and-a-vector-database)
9. [Command line](#command-line)
10. [Development](#development)
11. [License](#license)

---

## Features

- **A 0-100 match score with a breakdown.** The score combines four parts:
  semantic similarity, required skills, experience and education. You always
  see how much each part contributed.
- **Skill matching through a skill graph.** It understands **aliases**
  ("Postgres" is PostgreSQL, "k8s" is Kubernetes, "JS" is JavaScript). It
  also understands **implications**: a Django developer has Python, and a
  PostgreSQL user has SQL.
- **Matched and missing skills by name.** This tells a candidate or recruiter
  exactly what the gap is.
- **Requirements read from the JD.** Required skills, the minimum years of
  experience, and the accepted education level.
- **Facts read from the resume with careful regex:**
  - years of experience, from both stated claims and date ranges
  - education level
  - age, date of birth, CGPA, emails and phones
- **Hard filters** for minimum experience and required education. A failure
  is reported with a readable reason, and the score is still shown.
- **Bulk screening.** A pool of resumes is ranked against one JD and split
  into *shortlisted*, *below threshold* and *excluded (with the reason)*.
- **Pluggable similarity:**
  - TF-IDF, the default, which needs no model download
  - spaCy word vectors
  - your own embeddings
- **PDF and DOCX reading**, including text inside tables. An unreadable file
  raises a clear error instead of returning an empty string.
- **Fully configurable.** You can change the weights, the per-model
  calibration, the screening threshold and the skill vocabulary.
- **A `resume-match` command line** that prints JSON.
- **Tests:** about 290 checks against a hand-checked sample set, run on CI.

### What's new compared with the original in-app engine

This engine started inside a college placements platform. It has been split
out as its own open-source project, so other colleges and individuals can run
it in their own systems. Here's what changed:

| Added / changed | Why |
|---|---|
| No Django, database or settings dependency | Usable from any Python project |
| `SkillGraph` held in memory, loadable from JSON, extendable with `.extended()` | Bring your own skill vocabulary; no database tables needed |
| `ScoringConfig` (weights, calibration, threshold) | Replaces hard-wired Django settings |
| `score_resume(resume_text, jd_text)` | One call from raw text to a score |
| `requirements=` override on `match_score` | Plug in requirements from an LLM, a form or an ATS |
| `vector=` / `jd_vector=` on `match_score` | Use any embedding model |
| Backend chosen by name or as any object | `MatchingEngine(backend="tfidf")`, or your own class |
| `resume-match` CLI | Score and screen from the shell |
| `pypdf` instead of the deprecated `PyPDF2` | Maintained PDF library |
| Install with `pip`, with optional extras, plus CI | Easy for others to adopt |

---

## Installation

### With requirements.txt

```bash
git clone https://github.com/ChargingTrex/resume-matching-.git
cd resume-matching-
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt          # core + PDF/DOCX reading
pip install -e .                         # the package itself (adds the resume-match command)
```

| File | What it installs | When you need it |
|---|---|---|
| `requirements.txt` | `scikit-learn`, `numpy`, `pypdf`, `python-docx` | Always |
| `requirements-spacy.txt` | the above + `spacy` | Better semantic matching (then run `python -m spacy download en_core_web_md`, about 40 MB) |
| `requirements-dev.txt` | the above + `pytest`, `reportlab` | Running the tests |

| Package | Used for |
|---|---|
| `scikit-learn` | TF-IDF similarity (the default backend) |
| `numpy` | Vector maths |
| `pypdf` | Reading PDF resumes |
| `python-docx` | Reading DOCX resumes, including tables |
| `spacy` + `en_core_web_md` | *Optional.* Word-vector similarity |
| `pytest`, `reportlab` | *Dev only.* Tests, and rendering the sample resumes to PDF |

Python 3.10 or newer is required.

### With pip directly

```bash
pip install "resume-matching[files] @ git+https://github.com/ChargingTrex/resume-matching-.git"
# extras: [files] PDF/DOCX reading, [spacy] spaCy backend, [dev] tests
```

---

## Quick start

```python
from resume_matching import MatchingEngine, extract_text

engine = MatchingEngine()          # spaCy if its model is installed, else TF-IDF

with open("resume.pdf", "rb") as f:
    resume_text = extract_text(f, "resume.pdf")
jd_text = open("job_description.txt").read()

result = engine.score_resume(resume_text, jd_text)

result.score                  # 65  -> the match percentage
result.matched_skills         # ['Django', 'PostgreSQL', 'Python', ...]
result.missing_skills         # ['Django REST Framework', 'AWS']
result.components             # per-part score and weight
result.hard_filter_failures   # ['Needs 2 years of experience; resume shows 0.5.']
result.as_dict()              # JSON-ready
```

---

## How the matching works

Matching runs in three steps. The resume and the JD are read separately and
then compared.

```text
            ┌──────────────┐                       ┌──────────────┐
            │  Resume      │                       │  Job         │
            │  (PDF/DOCX/  │                       │  description │
            │   text)      │                       │              │
            └──────┬───────┘                       └──────┬───────┘
                   │ 1. read the resume                   │ 2. read the JD
                   ▼                                      ▼
   experience years, education level,        required skills, minimum
   skills named in the text                  experience, accepted education
                   │                                      │
                   └──────────────┬───────────────────────┘
                                  ▼
                       3. compare & score
          semantic · skills · experience · education
                                  ▼
               score 0-100 + breakdown + matched/missing skills
```

### Step 1: reading the resume

`resume_matching.extractors` reads the hard facts with regular expressions.
They are written to avoid the mistakes naive patterns make on real resumes:

- **Experience in years.** Durations near the word "experience" win:
  "3 years of experience", "Experience: 5 years", "2 yrs 6 months".
  - If none are stated, **date ranges** under an experience or internship
    heading, or on a line naming a role, are added up. "Jun 2023 - Present"
    is understood, and overlapping jobs are counted once.
  - Some durations are *not* experience: an age ("Age: 22 years",
    "21 years old"), study length ("4 years B.Tech programme",
    "10 years of schooling"), and anything above 50 years.
- **Education level** is normalised to `UG`, `Masters` or `PG` (PhD).
  - It matches B.E./B.Tech/B.Sc/BCA/BBA/B.Com, M.Tech/MCA/MBA/M.Sc/MS, PhD,
    and the spelled-out forms.
  - Some words are *not* degrees: the English word "be", and "MS Office" /
    "MS SQL".
  - A higher degree implies the lower ones, so a Masters satisfies a UG
    requirement.
- **Skills** are any skill names or aliases from the skill graph found in the
  text (see below), plus any skills you pass in explicitly.
- **Extras**, for verification only, through `engine.regex_facts(text)`: age,
  date of birth, CGPA (a percentage is converted at 9.5 points per CGPA
  point), emails and phone numbers. None of these affect the score.

### Step 2: reading the job description

`engine.job_requirements(jd_text)` returns:

```python
{
  "required_skills": ["Django", "PostgreSQL", "Python", "Django REST Framework",
                      "SQL", "Git", "Docker", "AWS"],
  "min_experience_years": 2.0,
  "education_levels": ["UG"],
  "source": "text"
}
```

- **Required skills (keyword matching).** The JD is split into tokens such as
  `Node.js`, `C++`, `C#` and `.NET`. Runs of up to 4 words are checked
  against every skill name and alias, longest first. That way "Django REST
  Framework" is one skill, not Django plus noise.
  - Spellings are compared by a normalised key, so "Postgres", "node js" and
    "NodeJS" all resolve. `+`, `#` and a leading `.` are kept, so C, C++, C#
    and .NET stay different skills.
  - Skill names that are also English words (Go, React, Express, Node, Spring,
    Flask…) count only when capitalised. "react to feedback" is not React.
- **Minimum experience** is set only by an explicit requirement, such as
  "2+ years of experience" or "Minimum 3 years". An internship's length, a
  service bond, or "25 years of company history" is ignored.
- **Accepted education** comes from the degree levels the JD names.

### Step 3: the skill graph (aliases and implications)

The graph is what makes keyword matching tolerant of how people actually
write:

```text
  "Postgres" ──alias──▶ PostgreSQL ──implies──▶ SQL
  "DRF" ──alias──▶ Django REST Framework ──implies──▶ Django ──implies──▶ Python
  "k8s" ──alias──▶ Kubernetes
```

A resume **covers** a required skill if it holds that skill, any alias of it,
or any skill that implies it (followed transitively). This only works in one
direction: holding Django meets a Python requirement, but holding Python does
not meet a Django one.

The bundled vocabulary is about 40 common tech skills, in
`src/resume_matching/data/skills.json`. Extend it for your domain (see
[Customising](#customising)).

---

## How the match percentage is calculated

The score is a **weighted average of four parts**. Each part is first turned
into a number between 0 and 1:

| Part | How the 0-1 value is computed | Default weight |
|---|---|---:|
| **semantic** | How similar the two texts are overall: TF-IDF cosine, spaCy vector cosine, or your embedding cosine, **calibrated** to 0-1 (see below) | 0.45 |
| **skills** | `matched required skills ÷ all required skills` | 0.35 |
| **experience** | `min(1, resume years ÷ JD minimum years)`; 1 if the JD states no minimum | 0.10 |
| **education** | 1 if the resume's level (or a level it implies) is one the JD accepts, else 0; 1 if the JD states none | 0.10 |

```text
                Σ ( part_score × weight )
score = round( 100 × ───────────────────────── )
                     Σ weight
```

**Calibration of the semantic part.** Raw similarity scales differ by method.
spaCy rates even unrelated English text around 0.8, while TF-IDF rarely goes
above 0.3 even for a strong match. Each method therefore has a **FLOOR** (it
and anything below it count as 0) and a **CEILING** (it and anything above it
count as 1). Values in between are mapped linearly:

```text
semantic = clamp( (raw − FLOOR) / (CEILING − FLOOR), 0, 1 )
```

| Source | FLOOR | CEILING | Measured on the bundled samples |
|---|---:|---:|---|
| `tfidf` | 0.00 | 0.25 | strong match 0.237, unrelated 0.014 |
| `spacy` | 0.80 | 0.97 | strong match 0.964, unrelated 0.801 |
| `embedding` | 0.40 | 0.85 | provisional, for `nomic-embed-text`; recalibrate for your model |

**When a part is missing, its weight moves to the others.** If the JD names no
known skill, there's nothing to measure skills against. The skills weight
becomes 0, and the other three parts share 100% between them. The weights in
the output are always the *shares actually used*, so they always sum to 1.

### Worked example

These numbers come from the sample "data engineer" resume against the sample
backend JD, using the TF-IDF backend:

| Part | Calculation | Part score | × weight | = contribution |
|---|---|---:|---:|---:|
| semantic | raw TF-IDF 0.102 → (0.102 − 0) / (0.25 − 0) | 0.408 | 0.45 | 0.184 |
| skills | 6 of 8 required skills held | 0.750 | 0.35 | 0.263 |
| experience | 3 years held, 2 required → min(1, 3/2) | 1.000 | 0.10 | 0.100 |
| education | Masters held, UG accepted → a Masters implies UG | 1.000 | 0.10 | 0.100 |
| | | | **total** | **0.646 → 65%** |

Against the same JD, scores on the sample set range from **98%** (the backend
engineer) down to **12-29%** for the unrelated resumes.

### Hard filters

Two requirements are also checked as **pass/fail filters**: minimum
experience and required education. In `match_score`, a failure is **listed,
not enforced**:

```python
result.hard_filter_failures   # ['Needs 2 years of experience; resume shows 0.5.']
result.flagged                # True
```

The score is still computed, and your app decides whether to hide or show
flagged candidates. In `screen`, the filters *are* enforced (see below).

### Reading the score honestly

The score is a **match, not a probability of being a good hire**. It only
sees what the text says. That's why the breakdown and the matched/missing
lists always come with the number. Use it to sort and shortlist, and leave
the hiring decision to a person.

---

## Bulk screening

```python
result = engine.screen(
    {"alice": alice_text, "bob": bob_text, "carol": carol_text},
    jd_text,
    min_experience_years=2,        # enforced hard filter (optional)
    required_education={"UG"},    # enforced hard filter; Masters/PG also pass
    similarity_threshold=None,     # None -> the backend's own default
)

[r.identifier for r in result.ranked]                # passed, best first
[r.identifier for r in result.below_threshold]       # passed filters, low similarity
[(r.identifier, r.excluded_by) for r in result.excluded]
# [('carol', 'Needs 2 years of experience; resume shows 0.5.')]
```

Screening ranks by semantic similarity only. Each candidate is scored
independently, so a person gets the same score whether they're screened
alone or in a batch of 500. An empty shortlist tells you *why* it's empty:
nobody applied, nobody scored high enough, or the filters were too strict.

Default thresholds: TF-IDF uses `0.10` and spaCy uses `0.85`. Each method has
its own scale, so a single fixed number would pass everyone on one and nobody
on the other.

---

## Customising

### Your own skill vocabulary

```python
from resume_matching import MatchingEngine, SkillGraph

graph = SkillGraph.default().extended([
    # (canonical name, aliases, implied skills)
    ("Rust", ["rustlang"], []),
    ("Tokio", [], ["Rust"]),
    ("AutoCAD", ["Auto CAD"], []),
])
engine = MatchingEngine(skills=graph)
```

Or load a JSON file, either on top of the defaults or instead of them:

```json
{"skills": [
  {"name": "SolidWorks", "aliases": ["Solid Works"], "implies": []},
  {"name": "Tally ERP",  "aliases": ["Tally"],       "implies": ["Accounting"]},
  {"name": "Accounting", "aliases": [],              "implies": []}
]}
```

```python
graph = SkillGraph.from_json("skills.json")                               # instead of the defaults
graph = SkillGraph.default().extended(SkillGraph.from_json("skills.json").entries())   # on top
```

### Weights, calibration and threshold

```python
from resume_matching import MatchingEngine, ScoringConfig

engine = MatchingEngine(
    backend="tfidf",                       # "auto" | "spacy" | "tfidf" | your object
    config=ScoringConfig(
        weights={"semantic": 0.3, "skills": 0.5, "experience": 0.1, "education": 0.1},
        calibration={"tfidf": {"FLOOR": 0.0, "CEILING": 0.25}},
        similarity_threshold=0.12,         # used by screen()
    ),
)
```

### Requirements from somewhere else

If an LLM, a form or your ATS already knows what the job needs, skip the
JD-reading step:

```python
engine.match_score(
    {"text": resume_text, "skills": ["Python"], "experience_years": 3, "education_levels": ["UG"]},
    jd_text,
    requirements={"required_skills": ["Python", "Django"], "min_experience_years": 2,
                  "education_levels": ["UG"]},
)
```

When scoring many resumes against one JD, call `engine.job_requirements(jd_text)`
once and pass the result to every `match_score` call.

### Your own similarity backend

```python
class MyBackend:
    name = "my-model"
    suggested_threshold = 0.5
    def similarity(self, a: str, b: str) -> float:   # must return 0..1
        ...

engine = MatchingEngine(backend=MyBackend())
# add a calibration entry for "my-model", or the raw value is used as-is
```

---

## Improving accuracy with embeddings and a vector database

### Why

TF-IDF counts shared words. A resume saying *"built REST services in Flask"*
and a JD asking for *"backend API development in Python"* share almost no
words, so TF-IDF scores them low. spaCy helps a little but averages word
vectors. A modern **sentence-embedding model** captures meaning: those two
texts land close together.

A **vector database** turns that into fast search. Instead of scoring every
resume against every JD, you store each resume's vector once and ask the
database for the 50 nearest resumes to a JD. Then you score only those 50 in
full.

### Step 1: compute embeddings

Any embedding model works. The engine only needs two lists of floats.

```python
# Option A: local, free, no server (pip install sentence-transformers)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")           # or "BAAI/bge-small-en-v1.5"
def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

# Option B: a self-hosted Ollama server (keeps resumes on your own machines)
import httpx
def embed(text: str) -> list[float]:
    r = httpx.post("http://localhost:11434/api/embed",
                   json={"model": "nomic-embed-text", "input": text})
    return r.json()["embeddings"][0]

# Option C: a hosted embedding API. Check your privacy obligations first:
# resumes are personal data.
```

### Step 2: pass the vectors to the engine

```python
from resume_matching import MatchingEngine

engine = MatchingEngine()
result = engine.match_score(
    {"text": resume_text, "experience_years": 3, "education_levels": ["UG"]},
    jd_text,
    vector=embed(resume_text),
    jd_vector=embed(jd_text),
)
result.semantic_source   # "embedding"
```

Skills, experience and education are scored exactly as before. Only the
semantic part changes, to the embedding cosine, calibrated with the
`"embedding"` bounds.

### Step 3: calibrate for your model (important)

Every embedding model has its own cosine range. One might rate unrelated
texts 0.2, another 0.6. Measure yours once:

```python
from resume_matching import cosine
# a handful of resume/JD pairs you know are good matches and poor matches
good = [cosine(embed(r), embed(jd)) for r, jd in known_good_pairs]
poor = [cosine(embed(r), embed(jd)) for r, jd in known_poor_pairs]
print(sum(poor)/len(poor), sum(good)/len(good))
```

Set **FLOOR ≈ the typical poor-match cosine** and **CEILING ≈ the typical
good-match cosine**:

```python
from resume_matching.config import DEFAULT_CALIBRATION
config = ScoringConfig(calibration={**DEFAULT_CALIBRATION,
                                    "embedding": {"FLOOR": 0.35, "CEILING": 0.80}})
engine = MatchingEngine(config=config)
```

Once embeddings carry the meaning, you can also give the semantic part more
weight, for example `{"semantic": 0.55, "skills": 0.30, ...}`.

### Step 4: add a vector database for search at scale

The pattern is **retrieve, then re-rank**:

```text
 new resume ──embed──▶ store vector + metadata ──▶ ┌─────────────┐
                                                   │  vector DB  │
 JD ──embed──▶ "50 nearest, experience ≥ 2" ─────▶ └──────┬──────┘
                                                          │ top 50 candidates
                                                          ▼
                              engine.match_score(...) on each (skills, experience,
                              education, calibrated cosine) ──▶ final ranked list
```

Store **metadata** next to each vector: experience years, education level and
skill slugs. The database can then filter *before* the nearest-neighbour
search, and the full scoring only runs on a small, relevant set.

**Example with Chroma** (`pip install chromadb`, embedded, no server):

```python
import chromadb
from resume_matching import MatchingEngine

engine = MatchingEngine()
client = chromadb.PersistentClient(path="./resume_index")
resumes = client.get_or_create_collection("resumes", metadata={"hnsw:space": "cosine"})

# --- index each resume once (when it is uploaded or changed) ---
def index_resume(resume_id: str, text: str):
    facts = engine.parse_text(text)
    skills = engine.skills.find_in_text(text)
    resumes.upsert(
        ids=[resume_id],
        embeddings=[embed(text)],
        documents=[text],
        metadatas=[{
            "experience_years": facts.experience_years,
            "education": facts.highest_education_level or "",
            "skills": ",".join(skills),          # metadata values must be scalars
        }],
    )

# --- search: nearest candidates, then full scoring ---
def search(jd_text: str, k: int = 50, min_years: float = 0):
    req = engine.job_requirements(jd_text)               # read the JD once
    jd_vec = embed(jd_text)
    hits = resumes.query(
        query_embeddings=[jd_vec],
        n_results=k,
        where={"experience_years": {"$gte": min_years}},
        include=["embeddings", "documents", "metadatas"],
    )
    results = []
    for rid, text, vec, meta in zip(hits["ids"][0], hits["documents"][0],
                                    hits["embeddings"][0], hits["metadatas"][0]):
        score = engine.match_score(
            {"text": text, "experience_years": meta["experience_years"],
             "education_levels": [meta["education"]] if meta["education"] else []},
            jd_text, vector=list(vec), jd_vector=jd_vec, requirements=req,
        )
        results.append((rid, score))
    return sorted(results, key=lambda r: r[1].score, reverse=True)
```

**Other vector databases** follow the same pattern:

| Option | Good for |
|---|---|
| **pgvector** (PostgreSQL extension) | You already run Postgres; vectors live next to your data. `ORDER BY embedding <=> %(jd)s LIMIT 50` |
| **Chroma** / **FAISS** | Local, embedded, zero ops; prototypes and single-server apps |
| **Qdrant** / **Weaviate** / **Milvus** | Dedicated servers, rich metadata filtering, millions of vectors |
| **MongoDB Atlas Vector Search**, **Elasticsearch/OpenSearch kNN** | You already run these |

### Practical tips

- **Never mix vectors from two models.** Store the model name with each
  vector, and re-embed everything when you change models.
- **Normalise vectors** (unit length) if the database uses inner product, so
  it equals cosine.
- **Long resumes.** Most models read about 512 tokens. Either embed a summary
  (skills + experience lines), or split the resume into sections, embed each,
  and keep the best section score.
- **Re-index on change.** Re-embed when a resume is edited. JDs can be
  embedded on the fly, or cached by a hash of their text.
- **Keep the keyword part.** Embeddings are good at meaning but vague on
  specifics ("Java" vs "JavaScript", "C++" vs "C#"). The skill graph is
  exact, and the combination beats either alone.
- **Privacy.** Resumes are personal data. Prefer a local model
  (sentence-transformers, Ollama) unless you have consent and a processing
  agreement for a hosted API.

### Further ideas

- Let an LLM read the JD into `requirements={...}` (required vs nice-to-have
  skills, real minimum experience), and pass it to `match_score`.
- Weight skills by importance (required more than nice-to-have) instead of
  counting them equally.
- Learn the weights from past shortlisting decisions instead of setting them
  by hand.
- Grow the skill graph from data: suggest aliases for unknown skill names by
  their embedding similarity to known ones, and have a person approve them.

---

## Command line

```bash
resume-match score resume.pdf jd.txt
resume-match screen jd.txt resumes/*.pdf --min-experience 2 --education UG
resume-match requirements jd.txt              # what the JD is read as asking for
resume-match --backend tfidf score resume.docx jd.txt
resume-match --skills my_skills.json score resume.pdf jd.txt
resume-match --skills my_skills.json --replace-skills requirements jd.txt
```

Inputs can be `.pdf`, `.docx` or plain text. Output is JSON.

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
python -m pytest
```

The tests run against a hand-checked sample set in `resume_matching.samples`.
Each sample resume states the experience, education, age and CGPA a person
read off it. The tests check that the extractors reproduce those values, and
that relevant resumes outrank irrelevant ones on every backend. The spaCy
tests are skipped when spaCy isn't installed.

```text
src/resume_matching/
  engine.py       MatchingEngine: parsing, match_score, screen
  extractors.py   regex: experience, education, age, DOB, CGPA, contacts
  extraction.py   PDF / DOCX → text
  skills.py       SkillGraph: aliases, implications, keyword finding
  similarity.py   TF-IDF and spaCy backends, cosine
  config.py       ScoringConfig: weights, calibration, threshold
  cli.py          the resume-match command
  data/skills.json  the bundled skill vocabulary
  samples.py      hand-checked sample resumes and JD
```

---

## License

Copyright (C) 2026 the resume-matching authors.

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0-only).
See [`LICENSE`](LICENSE).

In plain terms (the license text is what actually counts):

- **You can** use, study, change and share this engine freely, for a
  college, a placement cell, research or your own projects.
- **If you distribute it, or let people use a modified version over a
  network** (for example a web portal built on it), you must make your full
  source code, including your changes, available under the AGPL-3.0 too.
- **Commercial use is allowed**, but under the same terms. A company can't
  take it into a closed, proprietary product or service.

For use outside these terms, contact the maintainers through the
[GitHub repository](https://github.com/ChargingTrex/resume-matching-).
