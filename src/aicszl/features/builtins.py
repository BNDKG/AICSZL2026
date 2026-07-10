from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aicszl.raw.store import RawStore

from .registry import FeatureRegistry


@dataclass(frozen=True)
class FeatureCalcContext:
    raw_store: RawStore


def register_builtin_features(registry: FeatureRegistry) -> None:
    registry.feature_plugin(
        outputs=["market.close.v1", "market.amount.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
        kind="raw_field",
        description="Daily close and amount raw fields",
    )(_calc_market_raw_fields)
    registry.feature_plugin(
        outputs=["market.ret_5d_rank.v1"],
        inputs=["raw.daily", "raw.adj_factor"],
        lookback_days=5,
        kind="derived",
        description="Past 5-trading-day adjusted return cross-sectional percentile rank",
    )(_calc_market_ret_5d_rank)
    registry.feature_plugin(
        outputs=["limit.high_stop.v1"],
        inputs=["raw.daily", "raw.stk_limit"],
        lookback_days=0,
        kind="derived",
        description="Whether close price reaches the up-limit price",
    )(_calc_limit_high_stop)
    registry.feature_plugin(
        outputs=["moneyflow.net_mf_amount_rank.v1"],
        inputs=["raw.moneyflow"],
        lookback_days=0,
        kind="derived",
        description="Net moneyflow amount cross-sectional percentile rank",
    )(_calc_moneyflow_net_mf_amount_rank)


def _calc_market_raw_fields(ctx: FeatureCalcContext, dates: list[int]) -> pd.DataFrame:
    daily = _fetch_daily(ctx, dates, columns=["ts_code", "trade_date", "close", "amount"])
    close = _feature_frame(daily, "market.close.v1", "close")
    amount = _feature_frame(daily, "market.amount.v1", "amount")
    return pd.concat([close, amount], ignore_index=True)


def _calc_market_ret_5d_rank(ctx: FeatureCalcContext, dates: list[int]) -> pd.DataFrame:
    if not dates:
        return _empty_feature_frame()
    target_dates = sorted(int(date) for date in dates)
    max_date = max(target_dates)
    prices = ctx.raw_store.fetch_df(
        """
        SELECT d.ts_code, d.trade_date, d.close * a.adj_factor AS adj_close
        FROM daily d
        JOIN adj_factor a
          ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.trade_date <= ?
        ORDER BY d.ts_code, d.trade_date
        """,
        [max_date],
    )
    if prices.empty:
        return _empty_feature_frame()

    frames: list[pd.DataFrame] = []
    for ts_code, group in prices.groupby("ts_code"):
        ordered = group.sort_values("trade_date").copy()
        ordered["past_adj_close"] = ordered["adj_close"].shift(5)
        ordered["ret_5d"] = ordered["adj_close"] / ordered["past_adj_close"] - 1.0
        frames.append(ordered[["ts_code", "trade_date", "ret_5d"]])
    if not frames:
        return _empty_feature_frame()

    returns = pd.concat(frames, ignore_index=True)
    returns = returns[returns["trade_date"].isin(target_dates)].dropna(subset=["ret_5d"])
    if returns.empty:
        return _empty_feature_frame()

    returns["value"] = returns.groupby("trade_date")["ret_5d"].rank(method="average", pct=True)
    returns["feature_name"] = "market.ret_5d_rank.v1"
    return returns[["ts_code", "trade_date", "feature_name", "value"]].reset_index(drop=True)


def _calc_limit_high_stop(ctx: FeatureCalcContext, dates: list[int]) -> pd.DataFrame:
    if not dates:
        return _empty_feature_frame()
    joined = _fetch_limit_join(ctx, dates)
    if joined.empty:
        return _empty_feature_frame()
    result = joined[["ts_code", "trade_date"]].copy()
    result["feature_name"] = "limit.high_stop.v1"
    result["value"] = (joined["close"] >= joined["up_limit"]).astype(float)
    return result


def _calc_moneyflow_net_mf_amount_rank(ctx: FeatureCalcContext, dates: list[int]) -> pd.DataFrame:
    moneyflow = _fetch_table_for_dates(
        ctx,
        table_name="moneyflow",
        columns=["ts_code", "trade_date", "net_mf_amount"],
        dates=dates,
    )
    if moneyflow.empty:
        return _empty_feature_frame()
    moneyflow["value"] = moneyflow.groupby("trade_date")["net_mf_amount"].rank(method="average", pct=True)
    moneyflow["feature_name"] = "moneyflow.net_mf_amount_rank.v1"
    return moneyflow[["ts_code", "trade_date", "feature_name", "value"]].reset_index(drop=True)


def _fetch_daily(ctx: FeatureCalcContext, dates: list[int], columns: list[str]) -> pd.DataFrame:
    return _fetch_table_for_dates(ctx, table_name="daily", columns=columns, dates=dates)


def _fetch_limit_join(ctx: FeatureCalcContext, dates: list[int]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close", "up_limit"])
    placeholders = ", ".join("?" for _ in dates)
    return ctx.raw_store.fetch_df(
        f"""
        SELECT d.ts_code, d.trade_date, d.close, s.up_limit
        FROM daily d
        JOIN stk_limit s
          ON d.ts_code = s.ts_code AND d.trade_date = s.trade_date
        WHERE d.trade_date IN ({placeholders})
        """,
        [int(date) for date in dates],
    )


def _fetch_table_for_dates(
    ctx: FeatureCalcContext,
    table_name: str,
    columns: list[str],
    dates: list[int],
) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=columns)
    placeholders = ", ".join("?" for _ in dates)
    select_columns = ", ".join(columns)
    return ctx.raw_store.fetch_df(
        f"""
        SELECT {select_columns}
        FROM {table_name}
        WHERE trade_date IN ({placeholders})
        """,
        [int(date) for date in dates],
    )


def _feature_frame(df: pd.DataFrame, feature_name: str, value_column: str) -> pd.DataFrame:
    if df.empty:
        return _empty_feature_frame()
    result = df[["ts_code", "trade_date", value_column]].copy()
    result["feature_name"] = feature_name
    result = result.rename(columns={value_column: "value"})
    return result[["ts_code", "trade_date", "feature_name", "value"]]


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", "feature_name", "value"])
