"""Unit tests for the eval-log difficulty bucketing script."""

from pathlib import Path

from scripts.eval_difficulty_trajectory import (
    LINE_RE,
    difficulty_of,
    parse_log,
    summarize,
)

SAMPLE_LINE = (
    "[ep 7] scene=FloorPlan_Val2_4 id=bowl_1 success=True steps=42 "
    "collisions=3 spl=0.612 path=7.2 shortest=4.4"
)


def test_line_re_matches_evaluator_format() -> None:
    m = LINE_RE.search(SAMPLE_LINE)
    assert m is not None
    assert m.group("id") == "bowl_1"
    assert m.group("success") == "True"
    assert m.group("steps") == "42"
    assert m.group("spl") == "0.612"
    assert m.group("shortest") == "4.4"


def test_difficulty_of_buckets() -> None:
    assert difficulty_of({"shortest": 2.0}, easy_max=3.0, medium_max=6.0) == "easy"
    assert difficulty_of({"shortest": 3.0}, easy_max=3.0, medium_max=6.0) == "easy"
    assert difficulty_of({"shortest": 3.5}, easy_max=3.0, medium_max=6.0) == "medium"
    assert difficulty_of({"shortest": 6.0}, easy_max=3.0, medium_max=6.0) == "medium"
    assert difficulty_of({"shortest": 6.1}, easy_max=3.0, medium_max=6.0) == "hard"


def test_summarize_empty() -> None:
    assert summarize([]) == {"n": 0, "sr": 0.0, "spl": 0.0}


def test_summarize_values() -> None:
    eps = [
        {"success": True, "spl": 1.0},
        {"success": True, "spl": 0.5},
        {"success": False, "spl": 0.0},
        {"success": True, "spl": 0.0},
    ]
    s = summarize(eps)
    assert s["n"] == 4
    assert s["sr"] == 0.75
    assert s["spl"] == 0.375


def test_parse_log_extracts_episodes(tmp_path: Path) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        "\n".join(
            [
                "[ep 1] scene=A id=a success=False steps=10 collisions=1 spl=0.0 path=9.0 shortest=5.0",
                SAMPLE_LINE,
                "unrelated noise line that should be skipped",
            ]
        ),
        encoding="utf-8",
    )
    episodes = parse_log(str(log))
    assert len(episodes) == 2
    assert episodes[0]["id"] == "a"
    assert episodes[0]["success"] is False
    assert episodes[1]["success"] is True
    assert episodes[1]["spl"] == 0.612
