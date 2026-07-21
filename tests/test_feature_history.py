from pathlib import Path

import pandas as pd

from aicszl.features.history import fetch_bounded_history
from aicszl.raw.store import RawStore


DATES = [20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110, 20200113]


def test_fetch_bounded_history_returns_target_span_and_prior_observations(
    tmp_path: Path,
):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    daily_rows = []
    adjustment_rows = []
    for index, trade_date in enumerate(DATES):
        for code, base in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
            daily_rows.append(_daily_row(code, trade_date, base + index))
            adjustment_rows.append(
                {"ts_code": code, "trade_date": trade_date, "adj_factor": 1.0}
            )
    raw.upsert("daily", pd.DataFrame(daily_rows))
    raw.upsert("adj_factor", pd.DataFrame(adjustment_rows))

    result = fetch_bounded_history(
        raw,
        source_sql="""
            SELECT d.ts_code, d.trade_date, d.close * a.adj_factor AS adj_close
            FROM daily d
            JOIN adj_factor a USING (ts_code, trade_date)
        """,
        columns=["ts_code", "trade_date", "adj_close"],
        dates=[20200109, 20200110],
        lookback_rows=3,
    )

    assert list(result.columns) == ["ts_code", "trade_date", "adj_close"]
    assert result.groupby("ts_code")["trade_date"].apply(list).to_dict() == {
        "000001.SZ": [20200106, 20200107, 20200108, 20200109, 20200110],
        "000002.SZ": [20200106, 20200107, 20200108, 20200109, 20200110],
    }
    assert 20200113 not in set(result["trade_date"])


def test_fetch_bounded_history_returns_typed_empty_frame_for_no_dates(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)

    result = fetch_bounded_history(
        raw,
        source_sql="SELECT ts_code, trade_date, close FROM daily",
        columns=["ts_code", "trade_date", "close"],
        dates=[],
        lookback_rows=5,
    )

    assert result.empty
    assert list(result.columns) == ["ts_code", "trade_date", "close"]


def _daily_row(ts_code: str, trade_date: int, close: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "pre_close": close,
        "change": 0.0,
        "pct_chg": 0.0,
        "vol": 1000.0,
        "amount": 10000.0,
    }
