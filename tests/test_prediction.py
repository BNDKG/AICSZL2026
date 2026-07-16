import json
import pickle
from pathlib import Path

import pandas as pd

from aicszl.features.store import FeatureMeta, FeatureStore
from aicszl.predictions.runner import (
    PredictionRequest,
    predict_from_artifact,
    prediction_available_dates,
)


class LinearModel:
    def predict(self, data: pd.DataFrame):
        return data["market.close.v1"] * 0.1 + data["market.amount.v1"] * 0.001


def test_predict_from_artifact_writes_prediction_pkl_with_expected_columns(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _register_features(store)
    store.append_plugin_values(
        "market.raw_fields.v1",
        ["market.close.v1", "market.amount.v1"],
        pd.DataFrame(
            [
                _feature("000001.SZ", 20200102, 10.0, 1000.0),
                _feature("000002.SZ", 20200102, 20.0, 500.0),
                _feature("000001.SZ", 20200103, 30.0, 100.0),
            ]
        ),
    )
    store.append_target_values(
        "target.ret_5d_rank_pct.v1",
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": 20200102, "value": 0.8},
                {"ts_code": "000002.SZ", "trade_date": 20200102, "value": 0.2},
            ]
        ),
    )
    model_path, meta_path = _write_model_artifact(tmp_path)

    artifact = predict_from_artifact(
        store,
        PredictionRequest(
            model_path=model_path,
            meta_path=meta_path,
            start_date=20200102,
            end_date=20200103,
        ),
        tmp_path / "artifacts" / "predictions",
    )

    assert artifact.prediction_path.exists()
    result = pd.read_pickle(artifact.prediction_path)
    assert result.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "score_raw": 2.0,
            "score_rank": 0.5,
            "target.ret_5d_rank_pct.v1": 0.8,
            "model_artifact_id": "modelhash1",
            "train_job_id": "lgb_rank5_base_v1",
            "x_group": "base_v1",
            "y_name": "target.ret_5d_rank_pct.v1",
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "score_raw": 2.5,
            "score_rank": 1.0,
            "target.ret_5d_rank_pct.v1": 0.2,
            "model_artifact_id": "modelhash1",
            "train_job_id": "lgb_rank5_base_v1",
            "x_group": "base_v1",
            "y_name": "target.ret_5d_rank_pct.v1",
        },
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200103,
            "score_raw": 3.1,
            "score_rank": 1.0,
            "target.ret_5d_rank_pct.v1": None,
            "model_artifact_id": "modelhash1",
            "train_job_id": "lgb_rank5_base_v1",
            "x_group": "base_v1",
            "y_name": "target.ret_5d_rank_pct.v1",
        },
    ]


def test_prediction_available_dates_uses_complete_feature_rows_not_calendar_continuity(
    tmp_path: Path,
):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _register_features(store)
    store.append_plugin_values(
        "market.raw_fields.v1",
        ["market.close.v1", "market.amount.v1"],
        pd.DataFrame(
            [
                _feature("000001.SZ", 20200102, 10.0, 1000.0),
                _feature("000001.SZ", 20200103, 11.0, None),
                _feature("000001.SZ", 20200106, 12.0, 1200.0),
            ]
        ),
    )

    assert prediction_available_dates(
        store,
        ["market.close.v1", "market.amount.v1"],
        20200102,
        20200106,
    ) == [20200102, 20200106]


def _write_model_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.pkl"
    meta_path = tmp_path / "model.meta.json"
    with model_path.open("wb") as file:
        pickle.dump(LinearModel(), file)
    meta_path.write_text(
        json.dumps(
            {
                "artifact_hash": "modelhash1",
                "job": {
                    "name": "lgb_rank5_base_v1",
                    "x_group": "base_v1",
                    "features": ["market.close.v1", "market.amount.v1"],
                    "target": "target.ret_5d_rank_pct.v1",
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path, meta_path


def _feature(ts_code: str, trade_date: int, close: float, amount: float | None) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "market.close.v1": close,
        "market.amount.v1": amount,
    }


def _register_features(store: FeatureStore) -> None:
    for feature in ["market.close.v1", "market.amount.v1"]:
        store.register_feature_meta(
            FeatureMeta(
                feature_name=feature,
                domain="market",
                version="v1",
                kind="raw_field",
                owner_plugin="market.raw_fields.v1",
                input_tables=["daily"],
                lookback_days=0,
                code_hash=f"hash:{feature}",
            )
        )
