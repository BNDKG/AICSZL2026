from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aicszl.raw.store import RawStore


LEGACY_CLOSE_5D_TARGET = "target.ret_5d_rank_pct.v1"
EXECUTABLE_OPEN_5D_TARGET = "target.ret_open_t1_open_t6_rank_pct.v1"


@dataclass(frozen=True)
class TargetCalcContext:
    raw_store: RawStore


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    entry_offset: int
    exit_offset: int
    execution_delay: int
    holding_days: int
    purge_before_predict: bool


_TARGET_DEFINITIONS = {
    LEGACY_CLOSE_5D_TARGET: TargetDefinition(
        name=LEGACY_CLOSE_5D_TARGET,
        entry_offset=0,
        exit_offset=5,
        execution_delay=0,
        holding_days=5,
        purge_before_predict=False,
    ),
    EXECUTABLE_OPEN_5D_TARGET: TargetDefinition(
        name=EXECUTABLE_OPEN_5D_TARGET,
        entry_offset=1,
        exit_offset=6,
        execution_delay=1,
        holding_days=5,
        purge_before_predict=True,
    ),
}


def get_target_definition(name: str) -> TargetDefinition:
    try:
        return _TARGET_DEFINITIONS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown target: {name}") from exc


def calculate_target(
    ctx: TargetCalcContext,
    target_name: str,
    dates: list[int],
) -> pd.DataFrame:
    definition = get_target_definition(target_name)
    if definition.name == LEGACY_CLOSE_5D_TARGET:
        return calc_ret_5d_rank_pct(ctx, dates)
    return calc_ret_open_t1_open_t6_rank_pct(ctx, dates)


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
    return returns[["ts_code", "trade_date", "value"]].reset_index(drop=True)


def calc_ret_open_t1_open_t6_rank_pct(
    ctx: TargetCalcContext,
    dates: list[int],
) -> pd.DataFrame:
    if not dates:
        return _empty_target_frame()
    target_dates = sorted(int(date) for date in dates)
    prices = ctx.raw_store.fetch_df(
        """
        SELECT d.ts_code, d.trade_date, d.open * a.adj_factor AS adj_open
        FROM daily d
        JOIN adj_factor a
          ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.trade_date >= ?
        ORDER BY d.ts_code, d.trade_date
        """,
        [min(target_dates)],
    )
    if prices.empty:
        return _empty_target_frame()

    frames: list[pd.DataFrame] = []
    for _, group in prices.groupby("ts_code"):
        ordered = group.sort_values("trade_date").copy()
        ordered["entry_adj_open"] = ordered["adj_open"].shift(-1)
        ordered["exit_adj_open"] = ordered["adj_open"].shift(-6)
        ordered["ret_open_5d"] = (
            ordered["exit_adj_open"] / ordered["entry_adj_open"] - 1.0
        )
        frames.append(ordered[["ts_code", "trade_date", "ret_open_5d"]])
    returns = pd.concat(frames, ignore_index=True)
    returns = returns[returns["trade_date"].isin(target_dates)].dropna(
        subset=["ret_open_5d"]
    )
    if returns.empty:
        return _empty_target_frame()

    returns["value"] = returns.groupby("trade_date")["ret_open_5d"].rank(
        method="average",
        pct=True,
    )
    return returns[["ts_code", "trade_date", "value"]].reset_index(drop=True)


def _empty_target_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", "value"])
