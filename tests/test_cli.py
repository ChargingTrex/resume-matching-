import json

from resume_matching.cli import main
from resume_matching.samples import SAMPLE_JOB_DESCRIPTION, SAMPLES_BY_KEY


def test_score_prints_json(tmp_path, capsys):
    resume = tmp_path / "resume.txt"
    jd = tmp_path / "jd.txt"
    resume.write_text(SAMPLES_BY_KEY["backend_engineer"].text)
    jd.write_text(SAMPLE_JOB_DESCRIPTION)

    assert main(["--backend", "tfidf", "score", str(resume), str(jd)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert 0 <= out["score"] <= 100
    assert "Django" in out["matched_skills"]


def test_screen_and_requirements(tmp_path, capsys):
    jd = tmp_path / "jd.txt"
    jd.write_text(SAMPLE_JOB_DESCRIPTION)
    paths = []
    for key in ("backend_engineer", "marketing_executive"):
        path = tmp_path / f"{key}.txt"
        path.write_text(SAMPLES_BY_KEY[key].text)
        paths.append(str(path))

    assert main(["--backend", "tfidf", "screen", str(jd), *paths, "--threshold", "0"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ranked"][0]["identifier"].endswith("backend_engineer.txt")

    assert main(["requirements", str(jd)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["min_experience_years"] == 2.0


def test_custom_skills_file(tmp_path, capsys):
    skills = tmp_path / "skills.json"
    skills.write_text(json.dumps({"skills": [{"name": "Rust", "aliases": ["rustlang"]}]}))
    jd = tmp_path / "jd.txt"
    jd.write_text("Must know Rust and Python.")

    assert main(["--skills", str(skills), "requirements", str(jd)]) == 0
    assert json.loads(capsys.readouterr().out)["required_skills"] == ["Rust", "Python"]


def test_a_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    assert main(["score", str(tmp_path / "nope.txt"), str(tmp_path / "jd.txt")]) == 1
    assert "resume-match:" in capsys.readouterr().err
