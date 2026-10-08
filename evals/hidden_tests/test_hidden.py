"""Held-out tests. Never copied into the agent's workspace; run only by the
harness after the agent finishes, to catch fixes that pass the visible
tests by accident or by overfitting."""
import pytest

from textkit.ranges import binary_search, merge_intervals, overlaps, paginate, total_covered, total_pages
from textkit.stats import clamp, mean, median, moving_average, percentile, variance
from textkit.strings import count_vowels, initials, is_palindrome, slugify, truncate, wrap_words


def test_h_slugify_digits_and_unicode_punct():
    assert slugify("Top 10 Tips!!") == "top-10-tips"
    assert slugify("---") == ""


def test_h_truncate_boundaries():
    assert truncate("abcdef", 6) == "abcdef"
    assert truncate("abcdefg", 6) == "abc..."
    assert truncate("abcdefg", 3) == "..."
    assert truncate("abcdefg", 4, suffix="~") == "abc~"


def test_h_wrap():
    assert wrap_words("", 4) == []
    assert wrap_words("aa bb cc", 5) == ["aa bb", "cc"]
    assert wrap_words("one two three four", 9) == ["one two", "three", "four"]


def test_h_palindrome_and_vowels():
    assert is_palindrome("")
    assert is_palindrome("No 'x' in Nixon")
    assert not is_palindrome("ab")
    assert count_vowels("rhythm") == 0
    assert count_vowels("AEIOU aeiou") == 10


def test_h_initials():
    assert initials("grace") == "G"
    assert initials("  alan   mathison  turing  ") == "AMT"
    assert initials("a b c d", max_letters=4) == "ABCD"


def test_h_stats():
    assert mean([5]) == 5
    assert median([7]) == 7
    assert median([1, 2]) == 1.5
    assert median([10, 2, 38, 23, 38, 23]) == 23
    assert variance([1, 1, 1]) == 0
    assert variance([1, 3]) == 1.0


def test_h_percentile():
    data = list(range(1, 11))
    assert percentile(data, 10) == 1
    assert percentile(data, 25) == 3
    assert percentile(data, 90) == 9
    assert percentile(data, 91) == 10
    with pytest.raises(ValueError):
        percentile(data, -1)
    assert percentile([4], 100) == 4


def test_h_clamp_edges():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
    assert clamp(3, 3, 3) == 3
    assert clamp(-1.5, -1, 1) == -1


def test_h_moving_average():
    assert moving_average([1, 2, 3], 1) == [1.0, 2.0, 3.0]
    assert moving_average([1, 2], 3) == []
    assert moving_average([0, 10, 20, 30], 3) == [10.0, 20.0]
    with pytest.raises(ValueError):
        moving_average([1], 0)


def test_h_intervals():
    assert overlaps((0, 10), (2, 3))
    assert not overlaps((5, 8), (1, 5))
    assert overlaps((3, 4), (3, 4))
    assert merge_intervals([]) == []
    assert merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
    assert merge_intervals([(6, 8), (1, 3), (2, 4), (4, 5)]) == [(1, 5), (6, 8)]
    assert total_covered([(1, 2), (1, 2)]) == 1


def test_h_paging():
    assert total_pages(1, 10) == 1
    assert total_pages(11, 10) == 2
    with pytest.raises(ValueError):
        total_pages(5, 0)
    assert paginate(list(range(10)), 2, 4) == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        paginate([1], 0, 1)


def test_h_binary_search():
    data = list(range(0, 100, 5))
    for i, v in enumerate(data):
        assert binary_search(data, v) == i
    assert binary_search(data, -5) == -1
    assert binary_search(data, 100) == -1
    assert binary_search([4], 4) == 0
