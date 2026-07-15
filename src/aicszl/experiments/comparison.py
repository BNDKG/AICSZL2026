from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


SCORE_COLUMNS = ["trade_date", "ts_code", "score"]
METRIC_COLUMNS = [
    "report_rows",
    "start",
    "end",
    "net_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "mean_turnover",
    "total_cost",
    "nan_cells",
]


def build_common_scores(
    predictions: Mapping[str, pd.DataFrame],
    *,
    random_seed: int,
    topk: int,
) -> dict[str, pd.DataFrame]:
    if not predictions:
        raise ValueError("Common scores require at least one model prediction")
    if topk < 1:
        raise ValueError("topk must be at least 1")

    normalized: dict[str, pd.DataFrame] = {}
    common_keys: pd.DataFrame | None = None
    for label, prediction in predictions.items():
        required = {"trade_date", "ts_code", "score_raw"}
        missing = required.difference(prediction.columns)
        if missing:
            raise ValueError(
                f"Prediction '{label}' is missing required columns: {sorted(missing)}"
            )
        frame = prediction[["trade_date", "ts_code", "score_raw"]].copy()
        frame["trade_date"] = pd.to_numeric(frame["trade_date"], errors="raise").astype(
            "int64"
        )
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["score_raw"] = pd.to_numeric(frame["score_raw"], errors="raise").astype(
            float
        )
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"Prediction '{label}' contains duplicate keys")
        if not np.isfinite(frame["score_raw"].to_numpy()).all():
            raise ValueError(f"Prediction '{label}' score_raw values must be finite")
        keys = frame[["trade_date", "ts_code"]]
        common_keys = (
            keys.drop_duplicates()
            if common_keys is None
            else common_keys.merge(keys, on=["trade_date", "ts_code"], how="inner")
        )
        normalized[label] = frame

    if common_keys is None or common_keys.empty:
        raise ValueError("Prediction common-universe intersection is empty")
    common_keys = common_keys.sort_values(["trade_date", "ts_code"]).reset_index(
        drop=True
    )
    daily_counts = common_keys.groupby("trade_date", sort=True).size()
    insufficient = daily_counts[daily_counts < topk]
    if not insufficient.empty:
        dates = ",".join(str(int(value)) for value in insufficient.index[:5])
        raise ValueError(
            f"Common universe has dates with fewer than topk={topk} symbols: {dates}"
        )

    result: dict[str, pd.DataFrame] = {}
    random_frame = common_keys.copy()
    random_frame["score"] = np.random.default_rng(random_seed).random(len(random_frame))
    result["random_baseline"] = random_frame[SCORE_COLUMNS]

    for label, frame in normalized.items():
        common = common_keys.merge(
            frame,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        common["score"] = common.groupby("trade_date", sort=False)["score_raw"].rank(
            method="average", pct=True
        )
        result[label] = common[SCORE_COLUMNS]
    return result


def summarize_reports(
    reports: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    if not reports:
        raise ValueError("Report summary requires at least one report")
    expected_index: pd.Index | None = None
    metrics: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for label, report in reports.items():
        required = {"account", "turnover", "total_cost"}
        missing = required.difference(report.columns)
        if missing:
            raise ValueError(f"Report '{label}' is missing required columns: {sorted(missing)}")
        if len(report) < 2:
            raise ValueError(f"Report '{label}' must contain at least two rows")
        if expected_index is None:
            expected_index = report.index
        elif not report.index.equals(expected_index):
            raise ValueError("Experiment reports must have identical indexes")

        account = pd.to_numeric(report["account"], errors="raise").astype(float)
        if not np.isfinite(account.to_numpy()).all() or (account <= 0).any():
            raise ValueError(f"Report '{label}' account values must be finite positive numbers")
        turnover = pd.to_numeric(report["turnover"], errors="raise").astype(float)
        total_cost = pd.to_numeric(report["total_cost"], errors="raise").astype(float)
        equity = account / float(account.iloc[0])
        daily_returns = equity.pct_change().dropna()
        periods = len(daily_returns)
        daily_std = float(daily_returns.std(ddof=1))
        volatility = (
            None if not math.isfinite(daily_std) else float(daily_std * math.sqrt(252))
        )
        sharpe = (
            None
            if not math.isfinite(daily_std) or math.isclose(daily_std, 0.0, abs_tol=1e-15)
            else float(daily_returns.mean() / daily_std * math.sqrt(252))
        )
        values: dict[str, object] = {
            "report_rows": int(len(report)),
            "start": str(pd.Timestamp(report.index.min()).date()),
            "end": str(pd.Timestamp(report.index.max()).date()),
            "net_return": float(equity.iloc[-1] - 1.0),
            "annualized_return": float(equity.iloc[-1] ** (252 / periods) - 1.0),
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "mean_turnover": float(turnover.mean()),
            "total_cost": float(total_cost.iloc[-1]),
            "nan_cells": int(report.isna().sum().sum()),
        }
        metrics[label] = values
        rows.append({"series": label, **values})
    return metrics, pd.DataFrame(rows, columns=["series", *METRIC_COLUMNS])
