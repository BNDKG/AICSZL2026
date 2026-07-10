from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aicszl.raw.store import RawStore


@dataclass(frozen=True)
class TargetCalcContext:
    raw_store: RawStore


def calc_ret_5d_rank_pct(ctx: TargetCalcContext, dates: list[int]) -> pd.DataFrame:
    if not dates:
        return _empty_target_frame()
    target_dates = sorted(int(date) for date in dates)
    min_date = min(target_dates)
    prices = ctx.raw_store.fetch_df(
        """
        SELECT d.ts_code, d.trade_date, d.close * a.adj_factor AS adj_close
        FROM daily d
        JOIN adj_factor a
          ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.trade_date >= ?
        ORDER BY d.ts_code, d.trade_date
        """,
        [min_date],
    )
    if prices.empty:
        return _empty_target_frame()

    frames: list[pd.DataFrame] = []
    for ts_code, group in prices.groupby("ts_code"):
        ordered = group.sort_values("trade_date").copy()
        ordered["future_adj_close"] = ordered["adj_close"].shift(-5)
        ordered["ret_5d"] = ordered["future_adj_close"] / ordered["adj_close"] - 1.0
        frames.append(ordered[["ts_code", "trade_date", "ret_5d"]])
    if not frames:
        return _empty_target_frame()

    returns = pd.concat(frames, ignore_index=True)
    returns = returns[returns["trade_date"].isin(target_dates)].dropna(subset=["ret_5d"])
    if returns.empty:
        return _empty_target_frame()

    returns["value"] = returns.groupby("trade_date")["ret_5d"].rank(method="average", pct=True)
    returns["target_name"] = "target.ret_5d_rank_pct.v1"
    return returns[["ts_code", "trade_date", "target_name", "value"]].reset_index(drop=True)


def _empty_target_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", "target_name", "value"])
