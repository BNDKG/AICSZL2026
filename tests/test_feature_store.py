from pathlib import Path

import pandas as pd
import pytest

from aicszl.features.store import FeatureMeta, FeatureStore


PLUGIN = "test.wide.v1"
OUTPUTS = ["test.alpha.v1", "test.beta.v1"]


def test_feature_store_creates_wide_schema_without_long_fact_tables(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)

    tables = set(store.fetch_df("SHOW TABLES")["name"])

    assert tables == {"feature_meta", "feature_store_meta", "feature_update_state"}
    assert store.fetch_df("SELECT schema_version FROM feature_store_meta").iloc[0, 0] == 2


def test_feature_store_records_meta_and_rejects_code_hash_mismatch(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    store.register_feature_meta(_meta(OUTPUTS[0], code_hash="hash-v1"))

    with pytest.raises(ValueError, match="code_hash mismatch"):
        store.register_feature_meta(_meta(OUTPUTS[0], code_hash="hash-v2"))

    row = store.fetch_df(
        "SELECT owner_plugin, code_hash, status FROM feature_meta WHERE feature_name = ?",
        [OUTPUTS[0]],
    ).iloc[0]
    assert row.to_dict() == {
        "owner_plugin": PLUGIN,
        "code_hash": "hash-v1",
        "status": "active",
    }


def test_feature_store_transaction_rolls_back_wide_values_and_state(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    for output in OUTPUTS:
        store.register_feature_meta(_meta(output))

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            store.append_plugin_values(PLUGIN, OUTPUTS, _wide_rows([20200102]))
            for output in OUTPUTS:
                store.mark_success(output, 20200102, 20200102, row_count=1)
            raise RuntimeError("rollback")

    assert "fv_test_wide_v1" not in set(store.fetch_df("SHOW TABLES")["name"])
    assert all(store.get_state(output).status == "pending" for output in OUTPUTS)


def test_feature_store_tracks_attempt_success_failure_and_resets_one_plugin(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    for output in OUTPUTS:
        store.register_feature_meta(_meta(output))
    store.append_plugin_values(PLUGIN, OUTPUTS, _wide_rows([20200102]))

    store.mark_attempt(OUTPUTS[0], 20200103, 20200102)
    assert store.get_state(OUTPUTS[0]).status == "running"
    store.mark_success(OUTPUTS[0], 20200103, 20200102, row_count=1)
    assert store.get_state(OUTPUTS[0]).last_success_trade_date == 20200103
    store.mark_failure(OUTPUTS[0], 20200106, "boom")
    failed = store.get_state(OUTPUTS[0])
    assert (failed.status, failed.last_success_trade_date, failed.error_message) == (
        "failed",
        20200103,
        "boom",
    )

    store.reset_feature_plugin(PLUGIN, OUTPUTS)

    assert "fv_test_wide_v1" not in set(store.fetch_df("SHOW TABLES")["name"])
    assert store.get_feature_statuses(OUTPUTS) == {}
    assert all(store.get_state(output).status == "pending" for output in OUTPUTS)


def test_feature_store_reopens_read_only_and_reads_wide_values(tmp_path: Path):
    path = tmp_path / "features.duckdb"
    writer = FeatureStore(path, start_date=20200101)
    for output in OUTPUTS:
        writer.register_feature_meta(_meta(output))
    writer.append_plugin_values(PLUGIN, OUTPUTS, _wide_rows([20200102, 20200103]))
    writer.close()

    reader = FeatureStore(path, start_date=20200101, read_only=True)
    loaded = reader.load_feature_frame(OUTPUTS, 20200101, 20200131)

    assert loaded.to_dict("records") == _wide_rows([20200102, 20200103]).to_dict(
        "records"
    )
    with pytest.raises(Exception):
        reader.fetch_df("DELETE FROM feature_meta")


def _meta(feature_name: str, code_hash: str = "hash-v1") -> FeatureMeta:
    return FeatureMeta(
        feature_name=feature_name,
        domain="test",
        version="v1",
        kind="derived",
        owner_plugin=PLUGIN,
        input_tables=[],
        lookback_days=0,
        code_hash=code_hash,
    )


def _wide_rows(dates: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": date,
                OUTPUTS[0]: float(date),
                OUTPUTS[1]: float(date) + 1.0,
            }
            for date in dates
        ],
        columns=["ts_code", "trade_date", *OUTPUTS],
    )
