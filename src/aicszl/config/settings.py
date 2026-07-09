from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectSettings:
    start_date: int


@dataclass(frozen=True)
class PathSettings:
    raw_db: Path
    feature_db: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class TushareSettings:
    token_file: Path


@dataclass(frozen=True)
class Settings:
    project: ProjectSettings
    paths: PathSettings
    tushare: TushareSettings


def load_settings(path: str | Path = "configs/settings.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    project = _section(raw, "project")
    paths = _section(raw, "paths")
    tushare = _section(raw, "tushare")

    start_date = int(_required(project, "start_date"))
    if start_date < 20200101:
        raise ValueError("project.start_date must be >= 20200101")

    return Settings(
        project=ProjectSettings(start_date=start_date),
        paths=PathSettings(
            raw_db=Path(_required(paths, "raw_db")),
            feature_db=Path(_required(paths, "feature_db")),
            artifacts_dir=Path(_required(paths, "artifacts_dir")),
        ),
        tushare=TushareSettings(token_file=Path(_required(tushare, "token_file"))),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section")
    return section


def _required(section: dict[str, Any], key: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required setting '{key}'")
    return value
