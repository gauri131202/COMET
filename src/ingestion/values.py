from __future__ import annotations

import math


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

