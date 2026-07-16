from pathlib import Path

import pandas as pd
import pytest

from aicszl.datasets import DatasetRequest, assemble_dataset, validate_feature_group
from aicszl.features.store import FeatureMeta, FeatureStore


FEATURES = ["market.close.v1", "market.amount.v1"]
TARGET = "target.ret_5d_rank_pct.v1"
PLUGIN = "market.raw_fields.v1"


def test_assemble_dataset_joins_wide_features_and_target_in_requested_order(
    tmp_path: Path,
):
    store = _seed_store(tmp_path)

    result = assemble_dataset(
        store,
        DatasetRequest(FEATURES, TARGET, 20200102, 20200103),
    )

    assert result.columns.tolist() == ["ts_code", "trade_date", *FEATURES, TARGET]
    assert result[["ts_code", "trade_date"]].to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102},
        {"ts_code": "000002.SZ", "trade_date": 20200102},
        {"ts_code": "000001.SZ", "trade_date": 20200103},
    ]


def test_assemble_dataset_applies_filter_expressions_to_wide_columns(tmp_path: Path):
    store = _seed_store(tmp_path)

    result = assemble_dataset(
        store,
        DatasetRequest(
            FEATURES,
            TARGET,
            20200102,
            20200103,
            filters=["market.amount.v1 >= 1500", "market.close.v1 < 25"],
        ),
    )

    assert result[["ts_code", "trade_date"]].to_dict("records") == [
        {"ts_code": "000002.SZ", "trade_date": 20200102}
    ]


def test_validate_feature_group_rejects_empty_unknown_and_inactive_features(
    tmp_path: Path,
):
    store = _seed_store(tmp_path)
    store.conn.execute(
        "UPDATE feature_meta SET status = 'retired' WHERE feature_name = ?",
        [FEATURES[1]],
    )

    with pytest.raises(ValueError, match="at least one"):
        validate_feature_group(store, [])
    with pytest.raises(ValueError, match="Unknown features"):
        validate_feature_group(store, ["market.unknown.v1"])
    with pytest.raises(ValueError, match="Unknown features"):
        validate_feature_group(store, [FEATURES[1]])


def _seed_store(tmp_path: Path) -> FeatureStore:
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    for feature in FEATURES:
        store.register_feature_meta(
            FeatureMeta(
                feature_name=feature,
                domain="market",
                version="v1",
                kind="raw_field",
                owner_plugin=PLUGIN,
                input_tables=["raw.daily"],
                lookback_days=0,
                code_hash="hash-v1",
            )
        )
    store.append_plugin_values(
        PLUGIN,
        FEATURES,
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    FEATURES[0]: 10.0,
                    FEATURES[1]: 1000.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": 20200102,
                    FEATURES[0]: 20.0,
                    FEATURES[1]: 2000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200103,
                    FEATURES[0]: 30.0,
                    FEATURES[1]: 3000.0,
                },
            ]
        ),
    )
    store.append_target_values(
        TARGET,
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": 20200102, "value": 0.1},
                {"ts_code": "000002.SZ", "trade_date": 20200102, "value": 0.2},
                {"ts_code": "000001.SZ", "trade_date": 20200103, "value": 0.3},
            ]
        ),
    )
    return store
