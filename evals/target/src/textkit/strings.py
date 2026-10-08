"""String helpers."""
import re


def slugify(text: str) -> str:
    """'Hello, World!' -> 'hello-world'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-")


def truncate(text: str, width: int, suffix: str = "...") -> str:
    """Shorten text to at most `width` characters, ending with `suffix` when cut."""
    if width < len(suffix):
        raise ValueError("width must be at least len(suffix)")
    if len(text) <= width:
        return text
    return text[: width - len(suffix)] + suffix


def wrap_words(text: str, width: int) -> list[str]:
    """Greedy word wrap. Words longer than `width` get a line of their own."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = current + " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def is_palindrome(text: str) -> bool:
    """Ignores case and non-alphanumeric characters."""
    chars = [c.lower() for c in text if c.isalnum()]
    return chars == chars[::-1]


def count_vowels(text: str) -> int:
    return sum(1 for c in text.lower() if c in "aeiou")


def initials(name: str, max_letters: int = 3) -> str:
    """'ada king lovelace' -> 'AKL'."""
    parts = [p for p in name.split() if p]
    return "".join(p[0].upper() for p in parts[:max_letters])
