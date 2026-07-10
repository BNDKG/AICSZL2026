from pathlib import Path

import pandas as pd

from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.features.registry import FeatureRegistry
from aicszl.raw.store import RawStore


def test_builtin_market_raw_field_features_read_daily_table(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "daily",
        pd.DataFrame(
            [
                _daily_row("000001.SZ", 20200102, close=10.5, amount=10000.0),
                _daily_row("000002.SZ", 20200102, close=20.5, amount=20000.0),
            ]
        ),
    )
    registry = FeatureRegistry()
    register_builtin_features(registry)

    plugin = registry.get("market.close.v1")
    result = plugin.func(FeatureCalcContext(raw), [20200102])

    assert result.sort_values(["ts_code", "feature_name"]).to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "feature_name": "market.amount.v1", "value": 10000.0},
        {"ts_code": "000001.SZ", "trade_date": 20200102, "feature_name": "market.close.v1", "value": 10.5},
        {"ts_code": "000002.SZ", "trade_date": 20200102, "feature_name": "market.amount.v1", "value": 20000.0},
        {"ts_code": "000002.SZ", "trade_date": 20200102, "feature_name": "market.close.v1", "value": 20.5},
    ]


def test_builtin_market_ret_5d_rank_uses_past_adjusted_prices_only(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    rows = []
    adj_rows = []
    for trade_date, close_a, close_b in [
        (20200102, 10.0, 20.0),
        (20200103, 11.0, 20.5),
        (20200106, 12.0, 21.0),
        (20200107, 13.0, 21.5),
        (20200108, 14.0, 22.0),
        (20200109, 15.0, 22.5),
        (20200110, 100.0, 1.0),
    ]:
        rows.extend(
            [
                _daily_row("000001.SZ", trade_date, close=close_a, amount=1000.0),
                _daily_row("000002.SZ", trade_date, close=close_b, amount=1000.0),
            ]
        )
        adj_rows.extend(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.0},
                {"ts_code": "000002.SZ", "trade_date": trade_date, "adj_factor": 1.0},
            ]
        )
    raw.upsert("daily", pd.DataFrame(rows))
    raw.upsert("adj_factor", pd.DataFrame(adj_rows))
    registry = FeatureRegistry()
    register_builtin_features(registry)

    plugin = registry.get("market.ret_5d_rank.v1")
    result = plugin.func(FeatureCalcContext(raw), [20200109])

    assert result.sort_values("ts_code").to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200109, "feature_name": "market.ret_5d_rank.v1", "value": 1.0},
        {"ts_code": "000002.SZ", "trade_date": 20200109, "feature_name": "market.ret_5d_rank.v1", "value": 0.5},
    ]


def test_builtin_limit_and_moneyflow_features(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "daily",
        pd.DataFrame(
            [
                _daily_row("000001.SZ", 20200102, close=11.0, amount=1000.0),
                _daily_row("000002.SZ", 20200102, close=19.8, amount=1000.0),
            ]
        ),
    )
    raw.upsert(
        "stk_limit",
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": 20200102, "up_limit": 11.0, "down_limit": 9.0},
                {"ts_code": "000002.SZ", "trade_date": 20200102, "up_limit": 22.0, "down_limit": 18.0},
            ]
        ),
    )
    raw.upsert(
        "moneyflow",
        pd.DataFrame(
            [
                _moneyflow_row("000001.SZ", 20200102, net_mf_amount=10.0),
                _moneyflow_row("000002.SZ", 20200102, net_mf_amount=30.0),
            ]
        ),
    )
    registry = FeatureRegistry()
    register_builtin_features(registry)

    limit = registry.get("limit.high_stop.v1").func(FeatureCalcContext(raw), [20200102])
    moneyflow = registry.get("moneyflow.net_mf_amount_rank.v1").func(FeatureCalcContext(raw), [20200102])

    assert limit.sort_values("ts_code").to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20200102, "feature_name": "limit.high_stop.v1", "value": 1.0},
        {"ts_code": "000002.SZ", "trade_date": 20200102, "feature_name": "limit.high_stop.v1", "value": 0.0},
    ]
    assert moneyflow.sort_values("ts_code").to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "feature_name": "moneyflow.net_mf_amount_rank.v1",
            "value": 0.5,
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "feature_name": "moneyflow.net_mf_amount_rank.v1",
            "value": 1.0,
        },
    ]


def _daily_row(ts_code: str, trade_date: int, close: float, amount: float) -> dict[str, object]:
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
        "amount": amount,
    }


def _moneyflow_row(ts_code: str, trade_date: int, net_mf_amount: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "buy_sm_vol": 0.0,
        "buy_sm_amount": 0.0,
        "sell_sm_vol": 0.0,
        "sell_sm_amount": 0.0,
        "buy_md_vol": 0.0,
        "buy_md_amount": 0.0,
        "sell_md_vol": 0.0,
        "sell_md_amount": 0.0,
        "buy_lg_vol": 0.0,
        "buy_lg_amount": 0.0,
        "sell_lg_vol": 0.0,
        "sell_lg_amount": 0.0,
        "buy_elg_vol": 0.0,
        "buy_elg_amount": 0.0,
        "sell_elg_vol": 0.0,
        "sell_elg_amount": 0.0,
        "net_mf_vol": 0.0,
        "net_mf_amount": net_mf_amount,
    }
