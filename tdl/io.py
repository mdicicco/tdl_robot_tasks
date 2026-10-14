from __future__ import annotations

from pathlib import Path

from tdl.include import load_raw, resolve_includes
from tdl.schema import Task


def load_task(path: str | Path) -> Task:
    path = Path(path)
    data = resolve_includes(load_raw(path), path)
    return Task.model_validate(data)
