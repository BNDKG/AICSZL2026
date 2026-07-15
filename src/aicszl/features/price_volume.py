from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from aicszl.raw.store import RawStore

from .registry import FeatureRegistry


PLUGIN_ID = "market.price_volume_pack.v1"
OUTPUTS = [
    "market.ret_20d_rank.v1",
    "market.reversal_1d_rank.v1",
    "risk.volatility_20d_rank.v1",
    "liquidity.amount_ratio_5d_rank.v1",
    "market.close_position_20d_rank.v1",
]


class PriceVolumeCalcContext(Protocol):
    raw_store: RawStore


def register_price_volume_features(registry: FeatureRegistry) -> None:
    registry.feature_plugin(
        plugin_id=PLUGIN_ID,
        outputs=OUTPUTS,
        inputs=["raw.daily", "raw.adj_factor"],
        lookback_days=20,
        kind="derived",
        description="Price momentum, reversal, volatility, volume surge, and range position ranks",
    )(_calc_price_volume_pack)


def _calc_price_volume_pack(
    ctx: PriceVolumeCalcContext,
    dates: list[int],
) -> pd.DataFrame:
    target_dates = sorted({int(date) for date in dates})
    if not target_dates:
        return _empty_feature_frame()
    history = _fetch_price_volume_history(ctx, target_dates)
    if history.empty:
        return _empty_feature_frame()

    ordered = history.sort_values(["ts_code", "trade_date"]).copy()
    ordered["adj_close"] = ordered["close"] * ordered["adj_factor"]
    ordered["adj_high"] = ordered["high"] * ordered["adj_factor"]
    ordered["adj_low"] = ordered["low"] * ordered["adj_factor"]

    grouped = ordered.groupby("ts_code", sort=False)
    ordered["ret_1d"] = grouped["adj_close"].pct_change(periods=1, fill_method=None)
    ordered["ret_20d"] = grouped["adj_close"].pct_change(periods=20, fill_method=None)
    ordered["reversal_1d"] = -ordered["ret_1d"]
    ordered["volatility_20d"] = grouped["ret_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=1)
    )
    previous_amount_mean = grouped["amount"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).mean()
    )
    ordered["amount_ratio_5d"] = ordered["amount"] / previous_amount_mean
    low_20d = grouped["adj_low"].transform(
        lambda values: values.rolling(20, min_periods=20).min()
    )
    high_20d = grouped["adj_high"].transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    ordered["close_position_20d"] = (ordered["adj_close"] - low_20d) / (
        high_20d - low_20d
    )
    ordered = ordered.replace([np.inf, -np.inf], np.nan)
    target = ordered[ordered["trade_date"].isin(target_dates)]

    signal_outputs = [
        ("ret_20d", "market.ret_20d_rank.v1"),
        ("reversal_1d", "market.reversal_1d_rank.v1"),
        ("volatility_20d", "risk.volatility_20d_rank.v1"),
        ("amount_ratio_5d", "liquidity.amount_ratio_5d_rank.v1"),
        ("close_position_20d", "market.close_position_20d_rank.v1"),
    ]
    frames: list[pd.DataFrame] = []
    for signal, feature_name in signal_outputs:
        frame = target[["ts_code", "trade_date", signal]].dropna(subset=[signal]).copy()
        if frame.empty:
            continue
        frame["value"] = frame.groupby("trade_date")[signal].rank(
            method="average",
            pct=True,
        )
        frame["feature_name"] = feature_name
        frames.append(frame[["ts_code", "trade_date", "feature_name", "value"]])
    if not frames:
        return _empty_feature_frame()
    return pd.concat(frames, ignore_index=True)


def _fetch_price_volume_history(
    ctx: PriceVolumeCalcContext,
    dates: list[int],
) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "close", "high", "low", "amount", "adj_factor"]
        )
    min_date = min(int(date) for date in dates)
    max_date = max(int(date) for date in dates)
    return ctx.raw_store.fetch_df(
        """
        WITH joined AS (
            SELECT
                d.ts_code,
                d.trade_date,
                d.close,
                d.high,
                d.low,
                d.amount,
                a.adj_factor
            FROM daily d
            JOIN adj_factor a
              ON d.ts_code = a.ts_code
             AND d.trade_date = a.trade_date
            WHERE d.trade_date <= ?
        ),
        numbered AS (
            SELECT *, row_number() OVER (
                PARTITION BY ts_code ORDER BY trade_date
            ) AS observation_number
            FROM joined
        ),
        target_starts AS (
            SELECT ts_code, min(observation_number) AS first_target_observation
            FROM numbered
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY ts_code
        )
        SELECT
            n.ts_code,
            n.trade_date,
            n.close,
            n.high,
            n.low,
            n.amount,
            n.adj_factor
        FROM numbered n
        JOIN target_starts t USING (ts_code)
        WHERE n.observation_number >= t.first_target_observation - 20
        ORDER BY n.ts_code, n.trade_date
        """,
        [max_date, min_date, max_date],
    )


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", "feature_name", "value"])
