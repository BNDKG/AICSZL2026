from __future__ import annotations

from pathlib import Path

import pytest

from aicszl.config import FeatureGroup
from aicszl.experiments.config import (
    load_experiment_config,
    normalized_config_hash,
    resolve_feature_groups,
)


VALID_YAML = """\
name: pv_5_vs_10_long_202607
data:
  feature_cutoff: 20260710
train:
  start: 20200101
  end: 20240101
  target: target.ret_5d_rank_pct.v1
predict:
  start: 20240101
  end: 20260701
models:
  - label: 5_features
    feature_group: base_v1
  - label: 10_features
    feature_group: base_plus_price_volume_v1
model_params:
  n_estimators: 50
  learning_rate: 0.1
  min_data_in_leaf: 1
  verbose: -1
backtest:
  topk: 50
  n_drop: 5
  initial_cash: 1000000
  random_seed: 42
"""


def _write(tmp_path: Path, text: str = VALID_YAML) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_config_returns_typed_ordered_contract(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path))

    assert config.name == "pv_5_vs_10_long_202607"
    assert config.data.feature_cutoff == 20260710
    assert config.train.start == 20200101
    assert config.train.end == 20240101
    assert config.train.target == "target.ret_5d_rank_pct.v1"
    assert config.predict.start == 20240101
    assert config.predict.end == 20260701
    assert [model.label for model in config.models] == ["5_features", "10_features"]
    assert config.model_params.n_estimators == 50
    assert config.model_params.learning_rate == pytest.approx(0.1)
    assert config.backtest.random_seed == 42


def test_executable_five_day_config_keeps_models_and_changes_timing_contract():
    config = load_experiment_config(
        Path("configs/experiments/pv_5_vs_10_executable_5d_202607.yaml")
    )

    assert config.name == "pv_5_vs_10_executable_5d_202607"
    assert config.train.target == "target.ret_open_t1_open_t6_rank_pct.v1"
    assert [model.feature_group for model in config.models] == [
        "base_v1",
        "base_plus_price_volume_v1",
    ]
    assert config.backtest.topk == 50
    assert config.backtest.n_drop == 10
    assert config.backtest.random_seed == 42


def test_resolve_feature_groups_copies_ordered_feature_lists(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path))
    groups = {
        "base_v1": FeatureGroup("base_v1", ["market.close.v1"]),
        "base_plus_price_volume_v1": FeatureGroup(
            "base_plus_price_volume_v1",
            ["market.close.v1", "market.ret_20d_rank.v1"],
        ),
    }

    resolved = resolve_feature_groups(config, groups)

    assert [model.label for model in resolved] == ["5_features", "10_features"]
    assert resolved[0].features == ("market.close.v1",)
    assert resolved[1].features == (
        "market.close.v1",
        "market.ret_20d_rank.v1",
    )
    groups["base_v1"].features.append("market.amount.v1")
    assert resolved[0].features == ("market.close.v1",)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (VALID_YAML + "unknown: true\n", "unknown keys"),
        (VALID_YAML.replace("  random_seed: 42", "  random_seed: 42\n  extra: 1"), "unknown keys"),
        (VALID_YAML.replace("label: 10_features", "label: 5_features"), "duplicate model label"),
        (VALID_YAML.replace("base_plus_price_volume_v1", "base_v1"), "duplicate feature group"),
        (VALID_YAML.replace("5_features", "../escape", 1), "safe name"),
        (VALID_YAML.replace("models:\n  - label", "models: []\nignored:\n  - label"), "unknown keys"),
        (VALID_YAML.replace("start: 20200101", "start: 20250101", 1), "training date range"),
        (VALID_YAML.replace("start: 20240101", "start: 20260702", 1), "prediction date range"),
        (VALID_YAML.replace("end: 20260701", "end: 20260711", 1), "feature cutoff"),
        (VALID_YAML.replace("topk: 50", "topk: 0"), "topk"),
        (VALID_YAML.replace("n_drop: 5", "n_drop: 51"), "n_drop"),
        (VALID_YAML.replace("initial_cash: 1000000", "initial_cash: 0"), "initial_cash"),
        (VALID_YAML.replace("n_estimators: 50", "n_estimators: true"), "n_estimators"),
    ],
)
def test_load_experiment_config_rejects_invalid_contract(
    tmp_path: Path, text: str, message: str
):
    with pytest.raises(ValueError, match=message):
        load_experiment_config(_write(tmp_path, text))


def test_load_experiment_config_rejects_empty_models(tmp_path: Path):
    text = VALID_YAML.replace(
        "models:\n  - label: 5_features\n    feature_group: base_v1\n"
        "  - label: 10_features\n    feature_group: base_plus_price_volume_v1\n",
        "models: []\n",
    )

    with pytest.raises(ValueError, match="at least one model"):
        load_experiment_config(_write(tmp_path, text))


def test_resolve_feature_groups_rejects_missing_and_duplicate_features(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path))
    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_feature_groups(config, {"base_v1": FeatureGroup("base_v1", ["market.close.v1"])})

    groups = {
        "base_v1": FeatureGroup("base_v1", ["market.close.v1"]),
        "base_plus_price_volume_v1": FeatureGroup(
            "base_plus_price_volume_v1",
            ["market.close.v1", "market.close.v1"],
        ),
    }
    with pytest.raises(ValueError, match="duplicate feature"):
        resolve_feature_groups(config, groups)


def test_normalized_hash_ignores_yaml_mapping_order(tmp_path: Path):
    first = load_experiment_config(_write(tmp_path, VALID_YAML))
    reordered = VALID_YAML.replace(
        "  n_estimators: 50\n  learning_rate: 0.1",
        "  learning_rate: 0.1\n  n_estimators: 50",
    )
    second_path = tmp_path / "second.yaml"
    second_path.write_text(reordered, encoding="utf-8")
    second = load_experiment_config(second_path)

    assert normalized_config_hash(first) == normalized_config_hash(second)
    assert len(normalized_config_hash(first)) == 8
