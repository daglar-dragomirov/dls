from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None = None) -> None:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path(".env"),
            Path("../.env"),
            Path("/workspace/final_projects/.env"),
            Path("/workspace/final_projects/dls/.env"),
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def first_openrouter_key() -> str | None:
    load_env_file()
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY")
    for idx in range(1, 100):
        value = os.getenv(f"RIDER_KEY_{idx}")
        if value:
            return value
    return None

