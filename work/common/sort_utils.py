from __future__ import annotations

import re
from typing import Any


def natural_sort_key(value: Any) -> int:
    s = str(value)
    parts = re.findall(r"\d+", s)
    if parts:
        return int(parts[-1])
    return 0

