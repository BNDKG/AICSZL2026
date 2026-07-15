from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    features: list[str]


def load_feature_groups(
    path: str | Path = "configs/features.yaml",
) -> dict[str, FeatureGroup]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    groups_raw = raw.get("feature_groups")
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise ValueError("Missing or invalid 'feature_groups' section")

    groups: dict[str, FeatureGroup] = {}
    for group_name, group_raw in groups_raw.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("Feature group name must be a non-empty string")
        if not isinstance(group_raw, dict):
            raise ValueError(f"Feature group '{group_name}' must be a mapping")
        features = _feature_list(group_name, group_raw)
        groups[group_name] = FeatureGroup(name=group_name, features=features)
    return groups


def _feature_list(group_name: str, group_raw: dict[str, Any]) -> list[str]:
    raw_features = group_raw.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError(f"Feature group '{group_name}' must contain at least one feature")
    features: list[str] = []
    seen: set[str] = set()
    for raw_feature in raw_features:
        if not isinstance(raw_feature, str):
            raise ValueError(f"Feature group '{group_name}' feature names must be strings")
        _validate_feature_name(raw_feature)
        if raw_feature in seen:
            raise ValueError(
                f"Feature group '{group_name}' contains duplicate feature: {raw_feature}"
            )
        seen.add(raw_feature)
        features.append(raw_feature)
    return features


def _validate_feature_name(feature_name: str) -> None:
    parts = feature_name.split(".")
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("v"):
        raise ValueError(f"Feature name must use domain.name.version: {feature_name}")
