from pathlib import Path

import duckdb
import pandas as pd
import pytest

from aicszl.raw.store import RawStore


def test_daily_upsert_replaces_existing_primary_key(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    first = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": 20200102,
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "pre_close": 10.0,
                "change": 0.5,
                "pct_chg": 5.0,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        ]
    )
    replacement = first.copy()
    replacement.loc[0, "close"] = 12.0

    store.upsert("daily", first)
    store.upsert("daily", replacement)

    rows = store.fetch_df("SELECT ts_code, trade_date, close FROM daily")
    assert rows.to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "close": 12.0}
    ]


def test_update_state_records_successful_table_progress(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)

    store.mark_success("daily", 20200102, row_count=1)
    state = store.get_state("daily")

    assert state.table_name == "daily"
    assert state.start_date == 20200101
    assert state.last_success_trade_date == 20200102
    assert state.last_attempt_trade_date == 20200102
    assert state.status == "success"
    assert state.row_count == 1
    assert state.error_message == ""


def test_raw_store_upserts_all_v0_tables(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    samples = {
        "trade_cal": pd.DataFrame(
            [{"cal_date": 20200101, "exchange": "SSE", "is_open": 0, "pretrade_date": 20191231}]
        ),
        "adj_factor": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.25}]
        ),
        "stk_limit": pd.DataFrame(
            [{"trade_date": 20200102, "ts_code": "000001.SZ", "up_limit": 11.55, "down_limit": 9.45}]
        ),
        "moneyflow": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "buy_sm_vol": 1.0,
                    "buy_sm_amount": 2.0,
                    "sell_sm_vol": 3.0,
                    "sell_sm_amount": 4.0,
                    "buy_md_vol": 5.0,
                    "buy_md_amount": 6.0,
                    "sell_md_vol": 7.0,
                    "sell_md_amount": 8.0,
                    "buy_lg_vol": 9.0,
                    "buy_lg_amount": 10.0,
                    "sell_lg_vol": 11.0,
                    "sell_lg_amount": 12.0,
                    "buy_elg_vol": 13.0,
                    "buy_elg_amount": 14.0,
                    "sell_elg_vol": 15.0,
                    "sell_elg_amount": 16.0,
                    "net_mf_vol": 17.0,
                    "net_mf_amount": 18.0,
                }
            ]
        ),
        "daily_basic": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "turnover_rate": 1.0,
                    "volume_ratio": 2.0,
                    "pe": 3.0,
                    "pb": 4.0,
                    "ps_ttm": 5.0,
                    "dv_ttm": 6.0,
                    "circ_mv": 7.0,
                    "total_mv": 8.0,
                }
            ]
        ),
    }

    for table_name, df in samples.items():
        assert store.upsert(table_name, df) == 1
        rows = store.fetch_df(f'SELECT count(*) AS n FROM "{table_name}"')
        assert int(rows.loc[0, "n"]) == 1


def test_raw_store_can_reopen_read_only_without_writes(tmp_path: Path):
    path = tmp_path / "raw.duckdb"
    writer = RawStore(path, start_date=20200101)
    writer.close()

    reader = RawStore(path, start_date=20200101, read_only=True)
    try:
        assert reader.get_state("daily").status == "pending"
        with pytest.raises(duckdb.Error):
            reader.mark_success("daily", 20200102, row_count=1)
    finally:
        reader.close()
