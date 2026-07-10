from pathlib import Path

import pandas as pd
import pytest
import qlib
from qlib.data import D

from aicszl.backtests.qlib_adapter import export_qlib_provider, run_qlib_topk_backtest


def test_export_qlib_provider_writes_calendar_instruments_and_factor_bins(tmp_path: Path):
    scores = pd.DataFrame(
        [
            _row(20200102, "000001.SZ", 10.0),
            _row(20200103, "000001.SZ", 11.0),
            _row(20200106, "000001.SZ", 12.0),
        ]
    )
    factors = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.0},
            {"ts_code": "000001.SZ", "trade_date": 20200103, "adj_factor": 1.1},
            {"ts_code": "000001.SZ", "trade_date": 20200106, "adj_factor": 1.2},
        ]
    )

    path = export_qlib_provider(scores, factors, tmp_path / "provider")

    assert (path / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() == ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert (path / "instruments" / "all.txt").exists()
    assert (path / "features" / "000001.sz" / "factor.day.bin").exists()

    qlib.init(provider_uri=str(path), region="cn", clear_mem_cache=True)
    loaded = D.features(["000001.SZ"], ["$open", "$factor"], "2020-01-02", "2020-01-06")
    assert loaded["$open"].tolist() == [10.0, 11.0, 12.0]
    assert loaded["$factor"].tolist() == pytest.approx([1.0, 1.1, 1.2])

    metrics, _ = run_qlib_topk_backtest(path, scores, topk=1, n_drop=1, initial_cash=100_000)
    assert "1day" in metrics


def test_export_qlib_provider_marks_non_tradable_rows_as_suspended(tmp_path: Path):
    scores = pd.DataFrame([_row(20200102, "000001.SZ", 10.0)])
    scores.loc[0, "is_tradable"] = False
    factors = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.0}])

    path = export_qlib_provider(scores, factors, tmp_path / "provider")
    qlib.init(provider_uri=str(path), region="cn", clear_mem_cache=True)

    assert pd.isna(D.features(["000001.SZ"], ["$close"], "2020-01-02", "2020-01-02").iloc[0, 0])


def _row(trade_date: int, ts_code: str, close: float) -> dict[str, object]:
    return {"trade_date": trade_date, "ts_code": ts_code, "score": 1.0, "open": close, "high": close, "low": close, "close": close, "vol": 100.0, "amount": 1000.0, "is_tradable": True, "limit_up": close * 1.1, "limit_down": close * 0.9}
