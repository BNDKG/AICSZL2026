from pathlib import Path
import warnings

import pandas as pd
import pytest
import qlib
from qlib.data import D

from aicszl.backtests.base import BacktestRunSettings
from aicszl.backtests.dataset import BacktestDatasetArtifact
from aicszl.backtests import qlib_adapter
from aicszl.backtests.qlib_adapter import (
    LIMIT_PRICE_ATOL,
    QlibBacktestAdapter,
    export_qlib_provider,
    run_qlib_topk_backtest,
)


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
    _, positions = metrics["1day"]
    assert list(positions) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    assert positions[pd.Timestamp("2020-01-02")].get_stock_list() == []
    assert positions[pd.Timestamp("2020-01-03")].get_stock_list() == ["000001.SZ"]


def test_export_qlib_provider_marks_non_tradable_rows_as_suspended(tmp_path: Path):
    scores = pd.DataFrame([_row(20200102, "000001.SZ", 10.0)])
    scores.loc[0, "is_tradable"] = False
    factors = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.0}])

    path = export_qlib_provider(scores, factors, tmp_path / "provider")
    qlib.init(provider_uri=str(path), region="cn", clear_mem_cache=True)

    assert pd.isna(D.features(["000001.SZ"], ["$close"], "2020-01-02", "2020-01-02").iloc[0, 0])


def test_export_qlib_provider_does_not_warn_for_prelisting_calendar_rows(
    tmp_path: Path,
):
    scores = pd.DataFrame(
        [
            _row(20200102, "000001.SZ", 10.0),
            _row(20200103, "000001.SZ", 10.0),
            _row(20200103, "000002.SZ", 20.0),
        ]
    )
    factors = scores[["trade_date", "ts_code"]].assign(adj_factor=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        export_qlib_provider(scores, factors, tmp_path / "provider")


def test_qlib_backtest_blocks_buy_at_numeric_upper_limit(tmp_path: Path):
    scores = _scores_for_directional_limit_test()
    unrestricted = _run_positions(tmp_path / "unrestricted", scores)
    scores.loc[
        (scores["trade_date"] == 20200103) & (scores["ts_code"] == "000001.SZ"),
        "limit_up",
    ] = 10.0

    limited = _run_positions(tmp_path / "limited", scores)

    assert "000001.SZ" in unrestricted[pd.Timestamp("2020-01-03")].get_stock_list()
    assert "000001.SZ" not in limited[pd.Timestamp("2020-01-03")].get_stock_list()


def test_qlib_backtest_blocks_sell_at_numeric_lower_limit(tmp_path: Path):
    scores = _scores_for_directional_limit_test()
    unrestricted = _run_positions(tmp_path / "unrestricted", scores)
    scores.loc[
        (scores["trade_date"] == 20200106) & (scores["ts_code"] == "000001.SZ"),
        "limit_down",
    ] = 10.0

    limited = _run_positions(tmp_path / "limited", scores)

    assert "000001.SZ" not in unrestricted[pd.Timestamp("2020-01-06")].get_stock_list()
    assert "000001.SZ" in limited[pd.Timestamp("2020-01-03")].get_stock_list()
    assert "000001.SZ" in limited[pd.Timestamp("2020-01-06")].get_stock_list()
    assert limited[pd.Timestamp("2020-01-06")].get_stock_amount(
        "000001.SZ"
    ) == limited[pd.Timestamp("2020-01-03")].get_stock_amount("000001.SZ")


def test_qlib_provider_uses_documented_limit_price_tolerance(tmp_path: Path):
    scores = pd.DataFrame([_row(20200102, "000001.SZ", 10.0)])
    scores.loc[0, "limit_up"] = 10.0 + LIMIT_PRICE_ATOL / 2
    factors = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.0}]
    )

    path = export_qlib_provider(scores, factors, tmp_path / "provider")
    qlib.init(provider_uri=str(path), region="cn", clear_mem_cache=True)

    limit_buy = D.features(
        ["000001.SZ"], ["$limit_buy"], "2020-01-02", "2020-01-02"
    ).iloc[0, 0]
    assert limit_buy == 1.0


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ImportError("cannot import qlib"), "pyqlib==0.9.7"),
        (ValueError("invalid provider"), "Qlib backtest failed for dataset example"),
    ],
)
def test_qlib_adapter_wraps_engine_failures_with_actionable_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
):
    dataset_path = tmp_path / "scores.pkl"
    pd.DataFrame([_row(20200102, "000001.SZ", 10.0)]).to_pickle(dataset_path)

    class FakeRawStore:
        def fetch_df(self, query: str) -> pd.DataFrame:
            assert "adj_factor" in query
            return pd.DataFrame(
                [{"ts_code": "000001.SZ", "trade_date": 20200102, "adj_factor": 1.0}]
            )

    monkeypatch.setattr(qlib_adapter, "export_qlib_provider", lambda *args: tmp_path / "provider")

    def fail_backtest(*args, **kwargs):
        raise failure

    monkeypatch.setattr(qlib_adapter, "run_qlib_topk_backtest", fail_backtest)
    adapter = QlibBacktestAdapter(FakeRawStore())
    dataset = BacktestDatasetArtifact("example", dataset_path, rows=1)

    with pytest.raises(RuntimeError, match=message) as error:
        adapter.run(dataset, BacktestRunSettings(topk=1, n_drop=1, initial_cash=100_000))

    assert error.value.__cause__ is failure


def test_qlib_adapter_wraps_provider_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_path = tmp_path / "scores.pkl"
    pd.DataFrame([_row(20200102, "000001.SZ", 10.0)]).to_pickle(dataset_path)

    class FakeRawStore:
        def fetch_df(self, query: str) -> pd.DataFrame:
            return pd.DataFrame()

    failure = ValueError("missing adj_factor")

    def fail_export(*args, **kwargs):
        raise failure

    monkeypatch.setattr(qlib_adapter, "export_qlib_provider", fail_export)
    adapter = QlibBacktestAdapter(FakeRawStore())
    dataset = BacktestDatasetArtifact("example", dataset_path, rows=1)

    with pytest.raises(
        RuntimeError,
        match="Qlib provider export failed for dataset example",
    ) as error:
        adapter.run(dataset, BacktestRunSettings(topk=1, n_drop=1, initial_cash=100_000))

    assert error.value.__cause__ is failure


def _run_positions(tmp_path: Path, scores: pd.DataFrame) -> dict[pd.Timestamp, object]:
    factors = scores[["trade_date", "ts_code"]].assign(adj_factor=1.0)
    provider = export_qlib_provider(scores, factors, tmp_path / "provider")
    metrics, _ = run_qlib_topk_backtest(
        provider,
        scores,
        topk=1,
        n_drop=1,
        initial_cash=100_000,
    )
    return metrics["1day"][1]


def _scores_for_directional_limit_test() -> pd.DataFrame:
    rows = []
    for trade_date, score_a, score_b in [
        (20200102, 2.0, 1.0),
        (20200103, 1.0, 2.0),
        (20200106, 1.0, 2.0),
        (20200107, 1.0, 2.0),
    ]:
        rows.extend(
            [
                _row(trade_date, "000001.SZ", 10.0, score=score_a),
                _row(trade_date, "000002.SZ", 20.0, score=score_b),
            ]
        )
    return pd.DataFrame(rows)


def _row(
    trade_date: int,
    ts_code: str,
    close: float,
    *,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "score": score,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "vol": 100.0,
        "amount": 1000.0,
        "is_tradable": True,
        "limit_up": close * 1.1,
        "limit_down": close * 0.9,
    }
