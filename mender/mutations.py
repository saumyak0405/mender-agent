"""Token-level source mutations.

Used twice: by the eval generator to inject realistic single-token bugs
(off-by-one, flipped comparison, wrong operator, swapped boolean), and by
the offline heuristic policy, which searches the same operator space in
reverse. Working on tokens (not AST unparse) keeps formatting and comments
intact, so a mutant differs from the original by exactly one token.
"""
from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass

OPS = {"<": ["<="], "<=": ["<"], ">": [">="], ">=": [">"], "==": ["!="], "!=": ["=="],
       "+": ["-"], "-": ["+"], "*": ["//"], "//": ["*"], "+=": ["-="], "-=": ["+="]}
NAMES = {"and": ["or"], "or": ["and"], "True": ["False"], "False": ["True"], "min": ["max"], "max": ["min"]}


@dataclass(frozen=True)
class Mutation:
    line: int        # 1-based
    col: int
    end_col: int
    original: str
    replacement: str

    @property
    def desc(self) -> str:
        return f"line {self.line}: '{self.original}' -> '{self.replacement}'"


def _tokens(source: str):
    try:
        yield from tokenize.generate_tokens(io.StringIO(source).readline)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return


def candidates(source: str, line_range: tuple[int, int] | None = None) -> list[Mutation]:
    out: list[Mutation] = []
    for tok in _tokens(source):
        (row, col), (erow, ecol) = tok.start, tok.end
        if row != erow or (line_range and not line_range[0] <= row <= line_range[1]):
            continue
        reps: list[str] = []
        if tok.type == tokenize.OP:
            reps = OPS.get(tok.string, [])
        elif tok.type == tokenize.NAME:
            reps = NAMES.get(tok.string, [])
        elif tok.type == tokenize.NUMBER and tok.string.isdigit():
            n = int(tok.string)
            reps = [str(n + 1)] + ([str(n - 1)] if n > 0 else [])
        out += [Mutation(row, col, ecol, tok.string, r) for r in reps]
    return out


def apply(source: str, m: Mutation) -> str:
    lines = source.splitlines(keepends=True)
    line = lines[m.line - 1]
    assert line[m.col:m.end_col] == m.original, "mutation does not match source"
    lines[m.line - 1] = line[:m.col] + m.replacement + line[m.end_col:]
    return "".join(lines)
