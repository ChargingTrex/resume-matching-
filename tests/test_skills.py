"""The in-memory skill graph: building it, aliases, implications."""

import json

import pytest

from resume_matching.skills import SkillGraph, slugify_skill


@pytest.fixture
def graph():
    return SkillGraph.from_entries([
        ("Python", [], []),
        ("Django", [], ["Python"]),
        ("Django REST Framework", ["DRF"], ["Django"]),
        ("PostgreSQL", ["Postgres"], ["SQL"]),
        ("SQL", [], []),
    ])


class TestSlug:
    @pytest.mark.parametrize(
        "name, slug",
        [
            ("Node.js", "nodejs"),
            ("node js", "nodejs"),
            ("C++", "cplusplus"),
            ("C#", "csharp"),
            (".NET", "dotnet"),
            (" .net ", "dotnet"),
            ("ASP.NET", "aspnet"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_slugs(self, name, slug):
        assert slugify_skill(name) == slug


class TestBuilding:
    def test_an_alias_resolves_to_its_skill(self, graph):
        assert graph.canonical("postgres") == "PostgreSQL"
        assert graph.canonical("DRF") == "Django REST Framework"

    def test_an_unknown_name_is_its_own_canonical_form(self, graph):
        assert graph.canonical(" Haskell ") == "Haskell"
        assert graph.canonical_slug("Node.js") == graph.canonical_slug("node js")

    def test_implying_a_skill_not_in_the_graph_is_an_error(self):
        with pytest.raises(ValueError, match="Rust"):
            SkillGraph.from_entries([("Tokio", [], ["Rust"])])

    def test_a_canonical_name_wins_over_an_alias_spelled_the_same(self):
        graph = SkillGraph.from_entries([("Go", [], []), ("Golang", [], []), ("Other", ["golang"], [])])
        assert graph.canonical("golang") == "Golang"

    def test_dict_and_json_round_trip(self, graph, tmp_path):
        path = tmp_path / "skills.json"
        path.write_text(json.dumps(graph.to_dict()))
        again = SkillGraph.from_json(path)
        assert again.resolve == graph.resolve
        assert again.implies == graph.implies

    def test_extended_adds_without_losing_anything(self, graph):
        bigger = graph.extended([("FastAPI", [], ["Python"])])
        assert bigger.expand(["FastAPI"]) == {"FastAPI", "Python"}
        assert bigger.canonical("DRF") == "Django REST Framework"
        assert graph.canonical("FastAPI") == "FastAPI"  # the original is unchanged

    def test_the_default_vocabulary_loads(self):
        default = SkillGraph.default()
        assert default.canonical("k8s") == "Kubernetes"
        assert default.expand(["Next.js"]) >= {"Next.js", "React", "JavaScript"}


class TestComparisons:
    def test_implications_are_transitive(self, graph):
        assert graph.expand(["DRF"]) == {"Django REST Framework", "Django", "Python"}

    def test_overlap_goes_one_way(self, graph):
        assert graph.skill_overlap(["Python"], ["Django"]) == (["Python"], [])
        assert graph.skill_overlap(["Django"], ["Python"]) == ([], ["Django"])

    def test_overlap_lists_each_requirement_once_by_canonical_name(self, graph):
        matched, missing = graph.skill_overlap(["Postgres", "PostgreSQL", "SQL", "Django"], ["postgres"])
        assert matched == ["PostgreSQL", "SQL"]
        assert missing == ["Django"]

    def test_cycles_do_not_loop(self):
        graph = SkillGraph.from_entries([("A", [], ["B"]), ("B", [], ["A"])])
        assert graph.expand(["A"]) == {"A", "B"}
