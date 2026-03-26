from __future__ import annotations

import difflib


def create_unified_diff(old: str, new: str, path: str, context_lines: int = 3) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context_lines,
    )
    return "".join(diff)
