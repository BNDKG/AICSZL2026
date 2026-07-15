from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from aicszl.config import FeatureGroup


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class DataConfig:
    feature_cutoff: int


@dataclass(frozen=True)
class TrainConfig:
    start: int
    end: int
    target: str


@dataclass(frozen=True)
class PredictConfig:
    start: int
    end: int


@dataclass(frozen=True)
class ModelConfig:
    label: str
    feature_group: str


@dataclass(frozen=True)
class ModelParams:
    n_estimators: int
    learning_rate: float
    min_data_in_leaf: int
    verbose: int


@dataclass(frozen=True)
class ExperimentBacktestConfig:
    topk: int
    n_drop: int
    initial_cash: float
    random_seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    train: TrainConfig
    predict: PredictConfig
    models: tuple[ModelConfig, ...]
    model_params: ModelParams
    backtest: ExperimentBacktestConfig


@dataclass(frozen=True)
class ResolvedModel:
    label: str
    feature_group: str
    features: tuple[str, ...]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    root = _mapping(
        raw,
        "experiment",
        {"name", "data", "train", "predict", "models", "model_params", "backtest"},
    )
    name = _safe_name(root["name"], "experiment name")

    data_raw = _mapping(root["data"], "data", {"feature_cutoff"})
    train_raw = _mapping(root["train"], "train", {"start", "end", "target"})
    predict_raw = _mapping(root["predict"], "predict", {"start", "end"})
    params_raw = _mapping(
        root["model_params"],
        "model_params",
        {"n_estimators", "learning_rate", "min_data_in_leaf", "verbose"},
    )
    backtest_raw = _mapping(
        root["backtest"],
        "backtest",
        {"topk", "n_drop", "initial_cash", "random_seed"},
    )

    feature_cutoff = _integer(data_raw["feature_cutoff"], "feature_cutoff")
    train_start = _integer(train_raw["start"], "train.start")
    train_end = _integer(train_raw["end"], "train.end")
    predict_start = _integer(predict_raw["start"], "predict.start")
    predict_end = _integer(predict_raw["end"], "predict.end")
    target = _nonempty_string(train_raw["target"], "train.target")

    models_raw = root["models"]
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("Experiment must contain at least one model")
    models: list[ModelConfig] = []
    for index, item in enumerate(models_raw):
        model_raw = _mapping(item, f"models[{index}]", {"label", "feature_group"})
        models.append(
            ModelConfig(
                label=_safe_name(model_raw["label"], f"models[{index}].label"),
                feature_group=_nonempty_string(
                    model_raw["feature_group"], f"models[{index}].feature_group"
                ),
            )
        )
    _reject_duplicates([model.label for model in models], "duplicate model label")
    _reject_duplicates(
        [model.feature_group for model in models], "duplicate feature group"
    )

    n_estimators = _integer(params_raw["n_estimators"], "n_estimators")
    learning_rate = _number(params_raw["learning_rate"], "learning_rate")
    min_data_in_leaf = _integer(params_raw["min_data_in_leaf"], "min_data_in_leaf")
    verbose = _integer(params_raw["verbose"], "verbose")
    topk = _integer(backtest_raw["topk"], "topk")
    n_drop = _integer(backtest_raw["n_drop"], "n_drop")
    initial_cash = _number(backtest_raw["initial_cash"], "initial_cash")
    random_seed = _integer(backtest_raw["random_seed"], "random_seed")

    if train_start > train_end:
        raise ValueError("Invalid training date range: start must not exceed end")
    if predict_start > predict_end:
        raise ValueError("Invalid prediction date range: start must not exceed end")
    if predict_end > feature_cutoff:
        raise ValueError("Prediction end must not exceed feature cutoff")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if min_data_in_leaf < 1:
        raise ValueError("min_data_in_leaf must be positive")
    if topk < 1:
        raise ValueError("topk must be at least 1")
    if n_drop < 1 or n_drop > topk:
        raise ValueError("n_drop must be between 1 and topk")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    return ExperimentConfig(
        name=name,
        data=DataConfig(feature_cutoff=feature_cutoff),
        train=TrainConfig(start=train_start, end=train_end, target=target),
        predict=PredictConfig(start=predict_start, end=predict_end),
        models=tuple(models),
        model_params=ModelParams(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            verbose=verbose,
        ),
        backtest=ExperimentBacktestConfig(
            topk=topk,
            n_drop=n_drop,
            initial_cash=initial_cash,
            random_seed=random_seed,
        ),
    )


def resolve_feature_groups(
    config: ExperimentConfig,
    groups: dict[str, FeatureGroup],
) -> tuple[ResolvedModel, ...]:
    resolved: list[ResolvedModel] = []
    for model in config.models:
        if model.feature_group not in groups:
            raise ValueError(f"Unknown feature group: {model.feature_group}")
        features = tuple(groups[model.feature_group].features)
        if not features:
            raise ValueError(f"Feature group '{model.feature_group}' must not be empty")
        duplicates = _duplicates(list(features))
        if duplicates:
            raise ValueError(
                f"Feature group '{model.feature_group}' contains duplicate feature: "
                f"{duplicates[0]}"
            )
        resolved.append(
            ResolvedModel(
                label=model.label,
                feature_group=model.feature_group,
                features=features,
            )
        )
    return tuple(resolved)


def normalized_config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _mapping(value: Any, name: str, required_keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping")
    unknown = set(value).difference(required_keys)
    if unknown:
        raise ValueError(f"'{name}' contains unknown keys: {sorted(unknown)}")
    missing = required_keys.difference(value)
    if missing:
        raise ValueError(f"'{name}' is missing required keys: {sorted(missing)}")
    return value


def _safe_name(value: Any, name: str) -> str:
    normalized = _nonempty_string(value, name)
    if SAFE_NAME.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a safe name using letters, numbers, '_' or '-'")
    return normalized


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _reject_duplicates(values: list[str], message: str) -> None:
    duplicates = _duplicates(values)
    if duplicates:
        raise ValueError(f"{message}: {duplicates[0]}")


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
