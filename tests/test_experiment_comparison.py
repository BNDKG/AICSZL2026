from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aicszl.experiments.comparison import build_common_scores, summarize_reports


def _predictions() -> dict[str, pd.DataFrame]:
    five = pd.DataFrame(
        {
            "trade_date": [20240102] * 3 + [20240103] * 3,
            "ts_code": ["A", "B", "C", "A", "B", "C"],
            "score_raw": [0.3, 0.1, 0.2, 0.1, 0.3, 0.2],
            "unrelated": range(6),
        }
    )
    ten = pd.DataFrame(
        {
            "trade_date": [20240102] * 3 + [20240103] * 3,
            "ts_code": ["B", "C", "D", "B", "C", "D"],
            "score_raw": [10.0, 30.0, 20.0, 30.0, 10.0, 20.0],
        }
    )
    return {"5_features": five, "10_features": ten}


def test_common_scores_use_identical_sorted_intersection_and_recomputed_ranks():
    predictions = _predictions()
    originals = {key: value.copy(deep=True) for key, value in predictions.items()}

    scores = build_common_scores(predictions, random_seed=42, topk=2)

    assert list(scores) == ["random_baseline", "5_features", "10_features"]
    expected_keys = [
        (20240102, "B"),
        (20240102, "C"),
        (20240103, "B"),
        (20240103, "C"),
    ]
    for frame in scores.values():
        assert list(
            frame[["trade_date", "ts_code"]].itertuples(index=False, name=None)
        ) == expected_keys
        assert list(frame.columns) == ["trade_date", "ts_code", "score"]
    assert scores["5_features"]["score"].tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])
    assert scores["10_features"]["score"].tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])
    assert scores["random_baseline"]["score"].tolist() == pytest.approx(
        np.random.default_rng(42).random(4)
    )
    for key, original in originals.items():
        pd.testing.assert_frame_equal(predictions[key], original)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frames: {}, "at least one model"),
        (
            lambda frames: {"5_features": frames["5_features"].drop(columns="score_raw")},
            "missing required columns",
        ),
        (
            lambda frames: {
                "5_features": pd.concat(
                    [frames["5_features"], frames["5_features"].iloc[[0]]],
                    ignore_index=True,
                )
            },
            "duplicate keys",
        ),
        (
            lambda frames: {
                "5_features": frames["5_features"].assign(
                    score_raw=lambda frame: frame["score_raw"].mask(frame.index == 0, np.inf)
                )
            },
            "finite",
        ),
        (
            lambda frames: {
                "5_features": frames["5_features"],
                "10_features": frames["10_features"].assign(
                    ts_code=lambda frame: "Z" + frame["ts_code"]
                ),
            },
            "intersection is empty",
        ),
    ],
)
def test_common_scores_reject_invalid_inputs(mutator, message: str):
    with pytest.raises(ValueError, match=message):
        build_common_scores(mutator(_predictions()), random_seed=42, topk=1)


def test_common_scores_reject_dates_with_fewer_than_topk_symbols():
    with pytest.raises(ValueError, match="fewer than topk=3"):
        build_common_scores(_predictions(), random_seed=42, topk=3)


def _report(accounts: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account": accounts,
            "turnover": [0.0, 0.1, 0.2],
            "total_cost": [0.0, 1.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_summarize_reports_uses_documented_account_equity_formulas():
    reports = {
        "random_baseline": _report([100.0, 110.0, 99.0]),
        "5_features": _report([100.0, 105.0, 110.0]),
    }

    metrics, table = summarize_reports(reports)

    random = metrics["random_baseline"]
    daily = pd.Series([0.1, -0.1])
    assert random["report_rows"] == 3
    assert random["start"] == "2024-01-02"
    assert random["end"] == "2024-01-04"
    assert random["net_return"] == pytest.approx(-0.01)
    assert random["annualized_return"] == pytest.approx(0.99 ** (252 / 2) - 1)
    assert random["annualized_volatility"] == pytest.approx(daily.std(ddof=1) * math.sqrt(252))
    assert random["sharpe"] == pytest.approx(0.0)
    assert random["max_drawdown"] == pytest.approx(-0.1)
    assert random["mean_turnover"] == pytest.approx(0.1)
    assert random["total_cost"] == pytest.approx(3.0)
    assert random["nan_cells"] == 0
    assert table["series"].tolist() == ["random_baseline", "5_features"]
    assert list(table.columns) == ["series", *random.keys()]


def test_summarize_reports_uses_null_sharpe_for_zero_volatility():
    metrics, _ = summarize_reports({"flat": _report([100.0, 100.0, 100.0])})
    assert metrics["flat"]["sharpe"] is None


def test_summarize_reports_uses_null_volatility_for_one_return_observation():
    report = _report([100.0, 101.0, 102.0]).iloc[:2]

    metrics, _ = summarize_reports({"short": report})

    assert metrics["short"]["annualized_volatility"] is None
    assert metrics["short"]["sharpe"] is None


def test_summarize_reports_rejects_mismatched_indexes_and_invalid_accounts():
    shifted = _report([100.0, 101.0, 102.0])
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="identical indexes"):
        summarize_reports({"one": _report([100.0, 101.0, 102.0]), "two": shifted})

    invalid = _report([100.0, np.nan, 102.0])
    with pytest.raises(ValueError, match="finite positive"):
        summarize_reports({"invalid": invalid})
