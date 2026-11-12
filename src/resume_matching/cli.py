"""
Command line:

    resume-match score RESUME JD            one resume, 0-100 with breakdown
    resume-match screen JD RESUME [RESUME...] rank a pool against one JD
    resume-match requirements JD            what the JD is read as asking for

RESUME and JD are .pdf, .docx or plain-text files. Output is JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resume_matching.config import ScoringConfig
from resume_matching.engine import MatchingEngine
from resume_matching.extraction import ResumeParseError, extract_text
from resume_matching.skills import SkillGraph


def read_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() in {".pdf", ".docx"}:
        with p.open("rb") as handle:
            return extract_text(handle, p.name)
    return p.read_text(encoding="utf-8")


def build_engine(args) -> MatchingEngine:
    skills = SkillGraph.default()
    if args.skills:
        extra = SkillGraph.from_json(args.skills)
        skills = extra if args.replace_skills else skills.extended(extra.entries())
    return MatchingEngine(
        backend=args.backend,
        skills=skills,
        config=ScoringConfig(similarity_threshold=getattr(args, "threshold", None)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resume-match", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="auto", choices=["auto", "spacy", "tfidf"])
    parser.add_argument("--skills", help="JSON skill vocabulary to add (see README)")
    parser.add_argument("--replace-skills", action="store_true",
                        help="use --skills instead of the bundled vocabulary, not on top of it")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score one resume against a JD")
    score.add_argument("resume")
    score.add_argument("jd")

    screen = sub.add_parser("screen", help="rank many resumes against a JD")
    screen.add_argument("jd")
    screen.add_argument("resumes", nargs="+")
    screen.add_argument("--threshold", type=float)
    screen.add_argument("--min-experience", type=float)
    screen.add_argument("--education", action="append", choices=["UG", "Masters", "PG"])

    req = sub.add_parser("requirements", help="show what the JD asks for")
    req.add_argument("jd")

    args = parser.parse_args(argv)
    try:
        engine = build_engine(args)
        if args.command == "score":
            out = engine.score_resume(read_text(args.resume), read_text(args.jd)).as_dict()
        elif args.command == "requirements":
            out = engine.job_requirements(read_text(args.jd))
        else:
            pool = {path: read_text(path) for path in args.resumes}
            out = engine.screen(
                pool,
                read_text(args.jd),
                similarity_threshold=args.threshold,
                min_experience_years=args.min_experience,
                required_education=args.education,
            ).as_dict()
    except (ResumeParseError, OSError, ValueError) as exc:
        print(f"resume-match: {exc}", file=sys.stderr)
        return 1
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
