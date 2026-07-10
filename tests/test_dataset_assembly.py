from pathlib import Path

import pandas as pd
import pytest

from aicszl.datasets.assembly import DatasetRequest, assemble_dataset, validate_feature_group
from aicszl.features.store import FeatureStore


def test_assemble_dataset_pivots_feature_and_target_values(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _seed_values(store)

    dataset = assemble_dataset(
        store,
        DatasetRequest(
            features=["market.close.v1", "market.amount.v1"],
            target="target.ret_5d_rank_pct.v1",
            start_date=20200102,
            end_date=20200103,
        ),
    )

    assert dataset.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "market.close.v1": 10.0,
            "market.amount.v1": 1000.0,
            "target.ret_5d_rank_pct.v1": 0.8,
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "market.close.v1": 20.0,
            "market.amount.v1": 2000.0,
            "target.ret_5d_rank_pct.v1": 0.2,
        },
    ]


def test_assemble_dataset_applies_filter_expressions(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _seed_values(store)

    dataset = assemble_dataset(
        store,
        DatasetRequest(
            features=["market.close.v1", "market.amount.v1", "limit.high_stop.v1"],
            target="target.ret_5d_rank_pct.v1",
            start_date=20200102,
            end_date=20200103,
            filters=["limit.high_stop.v1 == 0", "market.amount.v1 > 1500"],
        ),
    )

    assert dataset.to_dict("records") == [
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "market.close.v1": 20.0,
            "market.amount.v1": 2000.0,
            "limit.high_stop.v1": 0.0,
            "target.ret_5d_rank_pct.v1": 0.2,
        }
    ]


def test_validate_feature_group_rejects_unknown_feature(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    store.upsert_feature_values(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "feature_name": "market.close.v1",
                    "value": 10.0,
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="Unknown features"):
        validate_feature_group(store, ["market.close.v1", "market.missing.v1"])


def _seed_values(store: FeatureStore) -> None:
    store.upsert_feature_values(
        pd.DataFrame(
            [
                _feature("000001.SZ", 20200102, "market.close.v1", 10.0),
                _feature("000001.SZ", 20200102, "market.amount.v1", 1000.0),
                _feature("000001.SZ", 20200102, "limit.high_stop.v1", 1.0),
                _feature("000002.SZ", 20200102, "market.close.v1", 20.0),
                _feature("000002.SZ", 20200102, "market.amount.v1", 2000.0),
                _feature("000002.SZ", 20200102, "limit.high_stop.v1", 0.0),
                _feature("000001.SZ", 20200103, "market.close.v1", 11.0),
            ]
        )
    )
    store.upsert_target_values(
        pd.DataFrame(
            [
                _target("000001.SZ", 20200102, "target.ret_5d_rank_pct.v1", 0.8),
                _target("000002.SZ", 20200102, "target.ret_5d_rank_pct.v1", 0.2),
                _target("000001.SZ", 20200103, "target.ret_5d_rank_pct.v1", 0.7),
            ]
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
