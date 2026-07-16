from pathlib import Path

import pandas as pd

from aicszl.artifact_cache import prediction_data_fingerprint
from aicszl.datasets import DatasetRequest, assemble_dataset
from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.features.registry import FeatureRegistry
from aicszl.features.store import FeatureMeta, FeatureStore
from aicszl.features.updater import FeatureUpdater
from aicszl.raw import RawStore
from aicszl.targets import TargetCalcContext, calculate_target


def test_feature_store_persists_plugin_outputs_in_wide_table(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    outputs = ["market.close.v1", "market.amount.v1"]
    for feature_name in outputs:
        store.register_feature_meta(
            FeatureMeta(
                feature_name=feature_name,
                domain="market",
                version="v1",
                kind="raw_field",
                owner_plugin="market.raw_fields.v1",
                input_tables=["raw.daily"],
                lookback_days=0,
                code_hash="abc",
            )
        )

    rows = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": 20200102,
                "market.close.v1": 10.5,
                "market.amount.v1": 1000.0,
            }
        ]
    )
    assert store.append_plugin_values("market.raw_fields.v1", outputs, rows) == 1

    tables = set(store.fetch_df("SHOW TABLES")["name"])
    assert "feature_values" not in tables
    assert "target_values" not in tables
    assert "fv_market_raw_fields_v1" in tables
    loaded = store.load_feature_frame(outputs, 20200101, 20200131)
    assert loaded.to_dict("records") == rows.to_dict("records")


def test_target_values_use_one_table_per_target(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    target = "target.ret_open_t1_open_t6_rank_pct.v1"
    rows = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": 20200102, "value": 0.75}]
    )

    assert store.append_target_values(target, rows) == 1
    tables = set(store.fetch_df("SHOW TABLES")["name"])
    assert "tv_target_ret_open_t1_open_t6_rank_pct_v1" in tables
    assert store.load_target_frame(target, 20200101, 20200131).to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, target: 0.75}
    ]


def test_builtin_base_plugins_return_wide_frames(tmp_path: Path) -> None:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "daily",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "pre_close": 10.0,
                    "change": 0.5,
                    "pct_chg": 5.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        ),
    )
    registry = FeatureRegistry()
    register_builtin_features(registry)

    plugin = registry.get_plugin("market.raw_fields.v1")
    result = plugin.func(FeatureCalcContext(raw), [20200102])
    assert result.columns.tolist() == ["ts_code", "trade_date", *plugin.outputs]
    assert "feature_name" not in result
    assert "value" not in result
    assert {item.plugin_id for item in registry.plugins()} == {
        "market.raw_fields.v1",
        "market.ret_5d_rank.v1",
        "limit.high_stop.v1",
        "moneyflow.net_mf_amount_rank.v1",
    }


def test_feature_updater_appends_wide_plugin_batch_atomically(tmp_path: Path) -> None:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "trade_cal",
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": 20200102,
                    "is_open": 1,
                    "pretrade_date": 20191231,
                }
            ]
        ),
    )
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id="test.wide.v1",
        outputs=["test.value.v1"],
        inputs=[],
        lookback_days=0,
    )
    def calculate(_ctx, dates):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": dates[0],
                    "test.value.v1": 0.5,
                }
            ]
        )

    summary = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=object(),
    ).update_to(20200102)

    assert summary["test.wide.v1"].last_success_trade_date == 20200102
    assert features.load_feature_frame(["test.value.v1"], 20200101, 20200131).to_dict(
        "records"
    ) == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "test.value.v1": 0.5}
    ]


def test_target_calculator_returns_value_without_target_name(tmp_path: Path) -> None:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    result = calculate_target(
        TargetCalcContext(raw), "target.ret_open_t1_open_t6_rank_pct.v1", []
    )
    assert result.columns.tolist() == ["ts_code", "trade_date", "value"]


def test_dataset_assembly_joins_wide_features_and_target(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    outputs = ["market.close.v1", "market.amount.v1"]
    for feature_name in outputs:
        store.register_feature_meta(
            FeatureMeta(
                feature_name=feature_name,
                domain="market",
                version="v1",
                kind="raw_field",
                owner_plugin="market.raw_fields.v1",
                input_tables=["raw.daily"],
                lookback_days=0,
                code_hash="abc",
            )
        )
    store.append_plugin_values(
        "market.raw_fields.v1",
        outputs,
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "market.close.v1": 10.5,
                    "market.amount.v1": 1000.0,
                }
            ]
        ),
    )
    target = "target.ret_open_t1_open_t6_rank_pct.v1"
    store.append_target_values(
        target,
        pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": 20200102, "value": 0.75}]
        ),
    )

    dataset = assemble_dataset(
        store,
        DatasetRequest(outputs, target, 20200101, 20200131),
    )
    assert dataset.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "market.close.v1": 10.5,
            "market.amount.v1": 1000.0,
            target: 0.75,
        }
    ]

    fingerprint = prediction_data_fingerprint(
        store,
        outputs,
        target,
        20200101,
        20200131,
    )
    assert [item["feature_name"] for item in fingerprint["features"]] == outputs
    assert [item["row_count"] for item in fingerprint["features"]] == [1, 1]
    assert fingerprint["target"]["row_count"] == 1
