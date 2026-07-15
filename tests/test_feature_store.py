from pathlib import Path

import duckdb
import pandas as pd
import pytest

from aicszl.features.store import FeatureMeta, FeatureStore


def test_feature_store_creates_v0_schema(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)

    tables = store.fetch_df(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    )

    assert tables["table_name"].tolist() == [
        "feature_meta",
        "feature_update_state",
        "feature_values",
        "target_values",
    ]


def test_feature_values_upsert_replaces_existing_value(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    first = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": 20200102,
                "feature_name": "market.close.v1",
                "value": 10.5,
            }
        ]
    )
    replacement = first.copy()
    replacement.loc[0, "value"] = 12.0

    assert store.upsert_feature_values(first) == 1
    assert store.upsert_feature_values(replacement) == 1

    rows = store.fetch_df(
        "SELECT ts_code, trade_date, feature_name, value, feature_version FROM feature_values"
    )
    assert rows.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "feature_name": "market.close.v1",
            "value": 12.0,
            "feature_version": "v1",
        }
    ]


def test_feature_store_records_meta_and_rejects_code_hash_mismatch(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    meta = FeatureMeta(
        feature_name="market.close.v1",
        domain="market",
        version="v1",
        kind="raw_field",
        owner_plugin="market_prices",
        input_tables=["raw.daily"],
        lookback_days=0,
        code_hash="abc123",
        status="active",
        description="Daily close price",
    )

    store.register_feature_meta(meta)
    store.register_feature_meta(meta)

    changed = FeatureMeta(
        feature_name="market.close.v1",
        domain="market",
        version="v1",
        kind="raw_field",
        owner_plugin="market_prices",
        input_tables=["raw.daily"],
        lookback_days=0,
        code_hash="different",
        status="active",
        description="Changed meaning",
    )
    with pytest.raises(ValueError, match="code_hash mismatch"):
        store.register_feature_meta(changed)

    rows = store.fetch_df(
        """
        SELECT feature_name, domain, version, kind, owner_plugin, input_tables,
               lookback_days, code_hash, status, description
        FROM feature_meta
        """
    )
    assert rows.to_dict("records") == [
        {
            "feature_name": "market.close.v1",
            "domain": "market",
            "version": "v1",
            "kind": "raw_field",
            "owner_plugin": "market_prices",
            "input_tables": "raw.daily",
            "lookback_days": 0,
            "code_hash": "abc123",
            "status": "active",
            "description": "Daily close price",
        }
    ]


def test_feature_update_state_records_attempt_success_and_failure(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)

    assert store.get_state("market.close.v1").status == "pending"

    store.mark_attempt("market.close.v1", trade_date=20200102, input_data_max_date=20200102)
    running = store.get_state("market.close.v1")
    assert running.last_attempt_trade_date == 20200102
    assert running.status == "running"
    assert running.input_data_max_date == 20200102

    store.mark_success("market.close.v1", trade_date=20200102, input_data_max_date=20200102, row_count=1)
    success = store.get_state("market.close.v1")
    assert success.last_success_trade_date == 20200102
    assert success.last_attempt_trade_date == 20200102
    assert success.status == "success"
    assert success.row_count == 1
    assert success.error_message == ""

    store.mark_failure("market.close.v1", trade_date=20200103, error_message="boom")
    failed = store.get_state("market.close.v1")
    assert failed.last_success_trade_date == 20200102
    assert failed.last_attempt_trade_date == 20200103
    assert failed.status == "failed"
    assert failed.error_message == "boom"


def test_feature_store_transaction_rolls_back_values_and_state(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    values = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": 20200102,
                "feature_name": "market.close.v1",
                "value": 10.5,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction():
            store.upsert_feature_values(values)
            store.mark_success("market.close.v1", 20200102, 20200102, 1)
            raise RuntimeError("boom")

    assert store.fetch_df("SELECT count(*) AS n FROM feature_values").iloc[0]["n"] == 0
    assert store.get_state("market.close.v1").status == "pending"


def test_feature_store_reports_statuses_and_date_coverage(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    meta = FeatureMeta(
        feature_name="market.close.v1",
        domain="market",
        version="v1",
        kind="raw_field",
        owner_plugin="market.raw_fields.v1",
        input_tables=["raw.daily"],
        lookback_days=0,
        code_hash="abc123",
    )
    store.register_feature_meta(meta)
    store.upsert_feature_values(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "feature_name": "market.close.v1",
                    "value": 10.5,
                }
            ]
        )
    )

    assert store.get_feature_statuses(["market.close.v1", "market.new.v1"]) == {
        "market.close.v1": "active"
    }
    assert store.feature_date_coverage(["market.close.v1", "market.new.v1"]) == {
        "market.close.v1": {20200102},
        "market.new.v1": set(),
    }


def test_feature_store_can_reopen_read_only_without_writes(tmp_path: Path):
    path = tmp_path / "features.duckdb"
    writer = FeatureStore(path, start_date=20200101)
    writer.close()

    reader = FeatureStore(path, start_date=20200101, read_only=True)
    try:
        assert reader.fetch_df("SELECT count(*) AS n FROM feature_values").iloc[0]["n"] == 0
        with pytest.raises(duckdb.Error):
            reader.mark_success("market.close.v1", 20200102, 20200102, 1)
    finally:
        reader.close()
