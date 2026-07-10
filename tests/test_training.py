import json
from pathlib import Path

import pandas as pd

from aicszl.features.store import FeatureMeta, FeatureStore
from aicszl.models.training import TrainingJob, compute_artifact_identity, train_lightgbm_regressor


def test_compute_artifact_identity_is_stable_and_includes_config_inputs():
    job = TrainingJob(
        name="lgb_rank5_base_v1",
        x_group="base_v1",
        features=["market.close.v1", "market.amount.v1"],
        target="target.ret_5d_rank_pct.v1",
        train_range=(20200101, 20200131),
        filters=["market.amount.v1 > 500"],
        model_params={"n_estimators": 3, "learning_rate": 0.1},
    )
    feature_hashes = {"market.close.v1": "close-hash", "market.amount.v1": "amount-hash"}

    first = compute_artifact_identity(job, feature_hashes)
    second = compute_artifact_identity(job, dict(reversed(feature_hashes.items())))
    changed = compute_artifact_identity(
        TrainingJob(
            name="lgb_rank5_base_v1",
            x_group="base_v1",
            features=["market.close.v1", "market.amount.v1"],
            target="target.ret_5d_rank_pct.v1",
            train_range=(20200101, 20200131),
            filters=["market.amount.v1 > 1000"],
            model_params={"n_estimators": 3, "learning_rate": 0.1},
        ),
        feature_hashes,
    )

    assert first == second
    assert first != changed
    assert len(first) == 8


def test_train_lightgbm_regressor_writes_model_and_metadata(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _seed_training_values(store)
    _seed_feature_meta(store)
    job = TrainingJob(
        name="lgb_rank5_base_v1",
        x_group="base_v1",
        features=["market.close.v1", "market.amount.v1"],
        target="target.ret_5d_rank_pct.v1",
        train_range=(20200102, 20200105),
        filters=["market.amount.v1 > 500"],
        model_params={"n_estimators": 3, "learning_rate": 0.1, "min_data_in_leaf": 1, "verbose": -1},
    )

    artifact = train_lightgbm_regressor(store, job, tmp_path / "artifacts" / "models")

    assert artifact.model_path.exists()
    assert artifact.meta_path.exists()
    assert artifact.model_path.name == f"{job.name}__{artifact.artifact_hash}.pkl"
    assert artifact.meta_path.name == f"{job.name}__{artifact.artifact_hash}.meta.json"

    metadata = json.loads(artifact.meta_path.read_text(encoding="utf-8"))
    assert metadata["artifact_hash"] == artifact.artifact_hash
    assert metadata["job"]["name"] == "lgb_rank5_base_v1"
    assert metadata["job"]["features"] == ["market.close.v1", "market.amount.v1"]
    assert metadata["job"]["target"] == "target.ret_5d_rank_pct.v1"
    assert metadata["feature_code_hashes"] == {
        "market.close.v1": "close-hash",
        "market.amount.v1": "amount-hash",
    }
    assert metadata["train_rows"] == 4


def _seed_training_values(store: FeatureStore) -> None:
    feature_rows = []
    target_rows = []
    samples = [
        ("000001.SZ", 20200102, 10.0, 1000.0, 0.1),
        ("000002.SZ", 20200102, 20.0, 2000.0, 0.3),
        ("000001.SZ", 20200103, 11.0, 1100.0, 0.2),
        ("000002.SZ", 20200103, 21.0, 2100.0, 0.4),
    ]
    for ts_code, trade_date, close, amount, target in samples:
        feature_rows.extend(
            [
                _feature(ts_code, trade_date, "market.close.v1", close),
                _feature(ts_code, trade_date, "market.amount.v1", amount),
            ]
        )
        target_rows.append(_target(ts_code, trade_date, "target.ret_5d_rank_pct.v1", target))
    store.upsert_feature_values(pd.DataFrame(feature_rows))
    store.upsert_target_values(pd.DataFrame(target_rows))


def _seed_feature_meta(store: FeatureStore) -> None:
    store.register_feature_meta(
        FeatureMeta(
            feature_name="market.close.v1",
            domain="market",
            version="v1",
            kind="raw_field",
            owner_plugin="market",
            input_tables=["raw.daily"],
            lookback_days=0,
            code_hash="close-hash",
        )
    )
    store.register_feature_meta(
        FeatureMeta(
            feature_name="market.amount.v1",
            domain="market",
            version="v1",
            kind="raw_field",
            owner_plugin="market",
            input_tables=["raw.daily"],
            lookback_days=0,
            code_hash="amount-hash",
        )
    )


def _feature(ts_code: str, trade_date: int, feature_name: str, value: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "feature_name": feature_name,
        "value": value,
    }


def _target(ts_code: str, trade_date: int, target_name: str, value: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "target_name": target_name,
        "value": value,
    }
