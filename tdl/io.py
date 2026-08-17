from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tdl.schema import Task


def load_task(path: str | Path) -> Task:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data: dict[str, Any] = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported task file type: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a mapping/object")
    return Task.model_validate(data)
