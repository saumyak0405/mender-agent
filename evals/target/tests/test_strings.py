import pytest

from textkit.strings import count_vowels, initials, is_palindrome, slugify, truncate, wrap_words


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_collapses_separators():
    assert slugify("  a -- b  ") == "a-b"


def test_truncate_short_text_unchanged():
    assert truncate("hello", 10) == "hello"


def test_truncate_exact_width_unchanged():
    assert truncate("hello", 5) == "hello"


def test_truncate_cuts_with_suffix():
    assert truncate("hello world", 8) == "hello..."


def test_truncate_rejects_tiny_width():
    with pytest.raises(ValueError):
        truncate("hello", 2)


def test_wrap_words():
    assert wrap_words("the quick brown fox jumps", 10) == ["the quick", "brown fox", "jumps"]


def test_wrap_words_exact_fit():
    assert wrap_words("ab cd", 5) == ["ab cd"]


def test_wrap_words_long_word():
    assert wrap_words("a extraordinarily b", 5) == ["a", "extraordinarily", "b"]


def test_palindrome_true():
    assert is_palindrome("A man, a plan, a canal: Panama")


def test_palindrome_false():
    assert not is_palindrome("hello")


def test_count_vowels():
    assert count_vowels("Education") == 5


def test_initials():
    assert initials("ada king lovelace") == "AKL"


def test_initials_limit():
    assert initials("a b c d e", max_letters=2) == "AB"
