from __future__ import annotations

import math
import re

_PLAIN_INTEGER = re.compile(r"-?\d+")
# 2**53 has 16 digits, so a shorter plain integer is always held exactly by a float.
_MIN_DIGITS_FOR_INEXACT = 16


def parse_number(text: str | None) -> float | None:
    """Numeric text as a float, or None when blank, not a number, or not finite.

    Always a float: an int above 2**63 - 1 raises OverflowError in sqlite3, while a
    float reaches about 1.8e308.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def inexact_integer_text(text: str | None) -> str | None:
    """The text itself when it is a plain integer that a float cannot hold exactly.

    A float holds every integer up to 2**53 and only some beyond it. Keep this text
    (for example in extra_json) so the original digits are not lost.
    """
    if text is None:
        return None
    text = text.strip()
    if len(text) < _MIN_DIGITS_FOR_INEXACT or not _PLAIN_INTEGER.fullmatch(text):
        return None
    as_float = float(text)
    if not math.isfinite(as_float) or int(as_float) != int(text):
        return text
    return None
