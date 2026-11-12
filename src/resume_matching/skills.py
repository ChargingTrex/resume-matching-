"""
Skill matching through aliases and implications.

A JD asking for "PostgreSQL" must match a resume that says "Postgres", and
one asking for "Python" must match a Django developer. Every skill
comparison goes through a `SkillGraph` rather than comparing raw names.

A graph is an immutable, in-memory value. `SkillGraph.default()` loads the
bundled vocabulary (`data/skills.json`); build your own with
`SkillGraph.from_entries`, `SkillGraph.from_dict` or `SkillGraph.from_json`,
or extend the default with `graph.extended(...)`.

With an empty graph everything still works: a name is its own canonical
form, compared by slug, so "Node.js" still meets "node js" -- but
`find_in_text` finds nothing, because it only knows the skills it is told.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

__all__ = ["SkillGraph", "SkillEntry", "slugify_skill"]

#: (canonical name, aliases, implied skills by canonical name)
SkillEntry = tuple[str, Iterable[str], Iterable[str]]


def slugify_skill(name: str | None) -> str:
    """
    A comparison key for a skill name.

    Lowercase alphanumerics only, with "+", "#" and a leading dot spelled out
    so C, C++, C# and .NET stay four different skills.
    """
    name = (name or "").strip().lower()
    name = re.sub(r"^\.", "dot", name).replace("+", "plus").replace("#", "sharp")
    return re.sub(r"[^a-z0-9]+", "", name)


#: A token: "Node.js", "C++", "C#", ".NET", "k8s". A trailing full stop is
#: the sentence's; a leading one may be the skill's (see `find_in_text`).
_TOKEN = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9+#.]*")
#: What may sit between the words of one skill name: "node js", "scikit-learn".
_JOINER = re.compile(r"[ \t]{1,3}|-")
#: The longest skill name, in words ("Google Cloud Platform").
_MAX_WORDS = 4
#: Skill names that are also ordinary English words. In running text they
#: count only with a capital: "Go" and "React" are skills, "go" and "react"
#: are what a JD asks you to do. A lone "c" is far more often a list
#: marker ("c) Git") than the language.
_COMMON_WORDS = frozenset({"go", "express", "node", "react", "angular", "spring", "flask", "next", "c"})


def _tokens(text: str) -> list[tuple[str, int, int]]:
    tokens = []
    for match in _TOKEN.finditer(text or ""):
        word = match.group().rstrip(".")
        if word:
            tokens.append((word, match.start(), match.start() + len(word)))
    return tokens


@dataclass(frozen=True)
class SkillGraph:
    #: canonical slug -> display name
    names: Mapping[str, str] = field(default_factory=dict)
    #: any known slug (alias or canonical) -> canonical slug
    resolve: Mapping[str, str] = field(default_factory=dict)
    #: canonical slug -> canonical slugs it directly implies
    implies: Mapping[str, frozenset[str]] = field(default_factory=dict)

    # --- building ----------------------------------------------------------------

    @classmethod
    def from_entries(cls, entries: Iterable[SkillEntry]) -> SkillGraph:
        """
        Build a graph from (name, aliases, implies) triples.

        An implied skill must itself be an entry. A canonical name always
        wins over an alias spelled the same way; the first alias of a
        spelling wins over later ones.
        """
        entries = [(name, list(aliases), list(implied)) for name, aliases, implied in entries]
        names: dict[str, str] = {}
        resolve: dict[str, str] = {}
        for name, _, _ in entries:
            slug = slugify_skill(name)
            if not slug:
                raise ValueError(f"Skill name {name!r} has no letters or digits.")
            names.setdefault(slug, name)
            resolve[slug] = slug
        for name, aliases, _ in entries:
            for alias in aliases:
                alias_slug = slugify_skill(alias)
                if alias_slug:
                    resolve.setdefault(alias_slug, slugify_skill(name))
        implies: dict[str, set[str]] = {slug: set() for slug in names}
        for name, _, implied in entries:
            for target in implied:
                target_slug = resolve.get(slugify_skill(target))
                if target_slug is None:
                    raise ValueError(f"{name!r} implies {target!r}, which is not a skill in the graph.")
                implies[slugify_skill(name)].add(target_slug)
        return cls(names=names, resolve=resolve, implies={k: frozenset(v) for k, v in implies.items()})

    @classmethod
    def from_dict(cls, data: Mapping) -> SkillGraph:
        """
        Build from `{"skills": [{"name": ..., "aliases": [...], "implies": [...]}]}`.
        """
        return cls.from_entries(
            (item["name"], item.get("aliases") or (), item.get("implies") or ())
            for item in data.get("skills", [])
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SkillGraph:
        """Build from a JSON file in the `from_dict` shape."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def default() -> SkillGraph:
        """The bundled vocabulary (about 40 common tech skills). Cached."""
        return _default_graph()

    def entries(self) -> list[tuple[str, list[str], list[str]]]:
        """This graph as (name, aliases, implies) triples, to extend or save."""
        aliases: dict[str, list[str]] = {slug: [] for slug in self.names}
        for alias_slug, slug in self.resolve.items():
            if alias_slug != slug:
                aliases[slug].append(alias_slug)
        return [
            (name, sorted(aliases[slug]), sorted(self.names[t] for t in self.implies.get(slug, ())))
            for slug, name in self.names.items()
        ]

    def extended(self, entries: Iterable[SkillEntry]) -> SkillGraph:
        """A new graph: this one plus `entries`. Existing skills keep their name."""
        return SkillGraph.from_entries([*self.entries(), *entries])

    def to_dict(self) -> dict:
        return {
            "skills": [
                {"name": name, "aliases": aliases, "implies": implied}
                for name, aliases, implied in self.entries()
            ]
        }

    # --- comparisons -------------------------------------------------------------

    def canonical_slug(self, name: str) -> str:
        """The comparison key of the skill `name` names: an alias's skill, or its own slug."""
        slug = slugify_skill(name)
        return self.resolve.get(slug, slug) if slug else ""

    def canonical(self, name: str) -> str:
        """The canonical spelling: "Postgres" -> "PostgreSQL". Unknown names as written."""
        slug = self.canonical_slug(name)
        if not slug:
            return ""
        return self.names.get(slug) or (name or "").strip()

    def expand_slugs(self, skills: Iterable[str]) -> set[str]:
        """Canonical slugs of `skills` and everything they imply, transitively."""
        seen: set[str] = set()
        stack = [slug for slug in (self.canonical_slug(name) for name in skills) if slug]
        while stack:
            slug = stack.pop()
            if slug in seen:
                continue  # cycle-safe: each skill is visited once
            seen.add(slug)
            stack.extend(self.implies.get(slug, ()))
        return seen

    def expand(self, skills: Iterable[str]) -> set[str]:
        """Canonical names of `skills` plus everything they imply."""
        skills = list(skills)
        as_written = {self.canonical_slug(name): (name or "").strip() for name in reversed(skills)}
        return {self.names.get(slug) or as_written.get(slug, slug) for slug in self.expand_slugs(skills)}

    def skill_overlap(self, required: Iterable[str], held: Iterable[str]) -> tuple[list[str], list[str]]:
        """
        (matched, missing): which required skills `held` covers.

        Through aliases and implications, in one direction only: holding Django
        meets a Python requirement, holding Python does not meet a Django one.
        Both lists hold canonical names, in the order required, each once.
        """
        covered = self.expand_slugs(held)
        matched: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for name in required:
            slug = self.canonical_slug(name)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            (matched if slug in covered else missing).append(self.canonical(name))
        return matched, missing

    def find_in_text(self, text: str) -> list[str]:
        """
        Known skills named in `text`, by canonical name, in order of appearance.

        Every canonical name and alias in the graph, matched on whole words
        (and runs of up to four words, longest first, so "Django REST
        Framework" is one skill, not Django plus noise). Spellings are
        compared by slug, so "Postgres", "node js" and "NodeJS" resolve.
        Only skills the graph knows are found; with an empty graph, none.
        """
        if not self.resolve:
            return []
        tokens = _tokens(text)
        found: list[str] = []
        seen: set[str] = set()
        i = 0
        while i < len(tokens):
            for width in range(min(_MAX_WORDS, len(tokens) - i), 0, -1):
                run = tokens[i : i + width]
                if any(not _JOINER.fullmatch(text[a[2] : b[1]]) for a, b in zip(run, run[1:])):
                    continue
                spelled = text[run[0][1] : run[-1][2]]
                slug = self.resolve.get(slugify_skill(spelled))
                if slug is None and spelled.startswith("."):
                    # ".NET" is a skill; "...React" is React after an ellipsis.
                    spelled = spelled[1:]
                    slug = self.resolve.get(slugify_skill(spelled))
                if slug is None:
                    continue
                if width == 1 and slugify_skill(spelled) in _COMMON_WORDS and spelled.islower():
                    continue
                if slug not in seen:
                    seen.add(slug)
                    found.append(self.names.get(slug, spelled))
                i += width
                break
            else:
                i += 1
        return found


@lru_cache(maxsize=1)
def _default_graph() -> SkillGraph:
    data = resources.files("resume_matching").joinpath("data/skills.json").read_text(encoding="utf-8")
    return SkillGraph.from_dict(json.loads(data))
