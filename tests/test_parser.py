"""Tests for the answer parser."""

import pytest

from biasinear.models._parser import parse_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        # Standalone letter
        ("A", "A"),
        ("B", "B"),
        ("The answer is C.", "C"),
        ("D is correct", "D"),
        # Letter in a sentence
        ("I think the answer is B because...", "B"),
        # Multiple standalone letters — first wins
        ("A or B", "A"),
        # Fallback: letter inside a word (no standalone match)
        ("ABCD", "A"),
        # No valid answer
        ("I don't know", None),
        ("", None),
        ("123", None),
        # Lowercase should NOT match
        ("the answer is a", None),
        # Mixed: standalone takes priority over embedded
        ("AB is wrong, C is correct", "C"),
        # Whitespace around
        ("  B  ", "B"),
        # Newline
        ("The correct answer is:\nA", "A"),
    ],
)
def test_parse_answer(text: str, expected: str | None) -> None:
    assert parse_answer(text) == expected
