from pathlib import Path

import pandas as pd
import pytest

from aicszl.features.store import FeatureStore
from aicszl.raw.store import RawStore
from aicszl.targets import (
    TargetCalcContext,
    calculate_target,
    get_target_definition,
)


LEGACY = "target.ret_5d_rank_pct.v1"
EXECUTABLE = "target.ret_open_t1_open_t6_rank_pct.v1"


def test_targets_use_separate_physical_wide_tables_and_replace_requested_range(
    tmp_path: Path,
):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    first = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": 20200102, "value": 0.25}]
    )
    replacement = first.assign(value=0.75)

    store.append_target_values(LEGACY, first)
    store.append_target_values(EXECUTABLE, first)
    store.append_target_values(LEGACY, replacement)

    tables = set(store.fetch_df("SHOW TABLES")["name"])
    assert "tv_target_ret_5d_rank_pct_v1" in tables
    assert "tv_target_ret_open_t1_open_t6_rank_pct_v1" in tables
    assert "target_values" not in tables
    assert store.load_target_frame(LEGACY, 20200101, 20200131)[LEGACY].tolist() == [
        0.75
    ]
    assert store.load_target_frame(EXECUTABLE, 20200101, 20200131)[
        EXECUTABLE
    ].tolist() == [0.25]


def test_ret_5d_rank_target_uses_forward_adjusted_close_returns(tmp_path: Path):
    raw = _price_store(tmp_path)

    result = calculate_target(TargetCalcContext(raw), LEGACY, [20200102])

    assert result.sort_values("ts_code").to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "value": 1.0},
        {"ts_code": "000002.SZ", "trade_date": 20200102, "value": 0.5},
    ]


def test_executable_target_uses_next_open_to_sixth_open(tmp_path: Path):
    raw = _price_store(tmp_path)

    result = calculate_target(TargetCalcContext(raw), EXECUTABLE, [20200102])

    assert result.sort_values("ts_code").to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "value": 1.0},
        {"ts_code": "000002.SZ", "trade_date": 20200102, "value": 0.5},
    ]


def test_target_definitions_expose_execution_and_purge_contract():
    legacy = get_target_definition(LEGACY)
    executable = get_target_definition(EXECUTABLE)

    assert (
        legacy.entry_offset,
        legacy.exit_offset,
        legacy.execution_delay,
        legacy.purge_before_predict,
    ) == (0, 5, 0, False)
    assert (
        executable.entry_offset,
        executable.exit_offset,
        executable.execution_delay,
        executable.purge_before_predict,
    ) == (1, 6, 1, True)
    with pytest.raises(ValueError, match="Unknown target"):
        get_target_definition("target.unknown.v1")


def _price_store(tmp_path: Path) -> RawStore:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    dates = [20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110]
    daily_rows = []
    adj_rows = []
    for index, date in enumerate(dates):
        daily_rows.extend(
            [
                _daily("000001.SZ", date, 10.0 + index * 2.0),
                _daily("000002.SZ", date, 20.0 + index * 0.2),
            ]
        )
        adj_rows.extend(
            [
                {"ts_code": "000001.SZ", "trade_date": date, "adj_factor": 1.0},
                {"ts_code": "000002.SZ", "trade_date": date, "adj_factor": 1.0},
            ]
        )
    raw.upsert("daily", pd.DataFrame(daily_rows))
    raw.upsert("adj_factor", pd.DataFrame(adj_rows))
    return raw


def _daily(ts_code: str, trade_date: int, price: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "pre_close": price,
        "change": 0.0,
        "pct_chg": 0.0,
        "vol": 1000.0,
        "amount": 10000.0,
    }
