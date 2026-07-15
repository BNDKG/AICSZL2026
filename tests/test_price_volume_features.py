from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.features.price_volume import _fetch_price_volume_history
from aicszl.features.registry import FeatureRegistry
from aicszl.features.store import FeatureStore
from aicszl.features.updater import FeatureUpdater
from aicszl.raw.store import RawStore


OUTPUTS = [
    "market.ret_20d_rank.v1",
    "market.reversal_1d_rank.v1",
    "risk.volatility_20d_rank.v1",
    "liquidity.amount_ratio_5d_rank.v1",
    "market.close_position_20d_rank.v1",
]


def test_price_volume_plugin_registration():
    registry = FeatureRegistry()
    register_builtin_features(registry)

    plugin = registry.get_plugin("market.price_volume_pack.v1")

    assert plugin.inputs == ["raw.daily", "raw.adj_factor"]
    assert plugin.lookback_days == 20
    assert plugin.outputs == OUTPUTS


def test_price_volume_features_have_expected_cross_sectional_direction(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    dates = list(range(20200101, 20200122))
    close_a = [100.0 + index + (4.0 if index % 2 else -4.0) for index in range(20)] + [130.0]
    close_b = [200.0 - index for index in range(21)]
    amount_a = [10.0] * 20 + [100.0]
    amount_b = [20.0] * 21
    _write_price_history(raw, dates, close_a, close_b, amount_a, amount_b)

    registry = FeatureRegistry()
    register_builtin_features(registry)
    result = registry.get_plugin("market.price_volume_pack.v1").func(
        FeatureCalcContext(raw),
        [dates[-1]],
    )

    assert result["trade_date"].unique().tolist() == [dates[-1]]
    assert set(result["feature_name"]) == set(OUTPUTS)
    wide = result.pivot(index="ts_code", columns="feature_name", values="value")
    for feature_name in [
        "market.ret_20d_rank.v1",
        "risk.volatility_20d_rank.v1",
        "liquidity.amount_ratio_5d_rank.v1",
        "market.close_position_20d_rank.v1",
    ]:
        assert wide.loc["000001.SZ", feature_name] > wide.loc["000002.SZ", feature_name]
    assert (
        wide.loc["000002.SZ", "market.reversal_1d_rank.v1"]
        > wide.loc["000001.SZ", "market.reversal_1d_rank.v1"]
    )
    assert np.isfinite(result["value"]).all()


def test_price_volume_features_ignore_future_rows(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    dates = list(range(20200101, 20200122))
    close_a = [100.0 + index for index in range(21)]
    close_b = [200.0 - index for index in range(21)]
    _write_price_history(raw, dates, close_a, close_b, [10.0] * 21, [20.0] * 21)
    registry = FeatureRegistry()
    register_builtin_features(registry)
    plugin = registry.get_plugin("market.price_volume_pack.v1")

    before = plugin.func(FeatureCalcContext(raw), [dates[-1]]).sort_values(
        ["ts_code", "feature_name"]
    ).reset_index(drop=True)
    _append_prices(raw, 20200122, {"000001.SZ": 10000.0, "000002.SZ": 1.0})
    after = plugin.func(FeatureCalcContext(raw), [dates[-1]]).sort_values(
        ["ts_code", "feature_name"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(before, after)


def test_price_volume_history_loader_returns_only_twenty_prior_rows_per_stock(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    dates = list(range(20200101, 20200131))
    _write_price_history(
        raw,
        dates,
        [100.0 + index for index in range(30)],
        [200.0 - index for index in range(30)],
        [10.0] * 30,
        [20.0] * 30,
    )

    history = _fetch_price_volume_history(FeatureCalcContext(raw), [dates[-1]])

    assert history.groupby("ts_code").size().to_dict() == {
        "000001.SZ": 21,
        "000002.SZ": 21,
    }
    assert history.groupby("ts_code")["trade_date"].min().to_dict() == {
        "000001.SZ": dates[-21],
        "000002.SZ": dates[-21],
    }


def test_selected_price_volume_plugin_advances_all_five_outputs_atomically(tmp_path: Path):
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    dates = list(range(20200101, 20200122))
    _write_price_history(
        raw,
        dates,
        [100.0 + index for index in range(21)],
        [200.0 - index for index in range(21)],
        [10.0] * 20 + [100.0],
        [20.0] * 21,
    )
    raw.upsert(
        "trade_cal",
        pd.DataFrame(
            [
                {
                    "cal_date": date,
                    "exchange": "SSE",
                    "is_open": 1,
                    "pretrade_date": dates[index - 1] if index else 20191231,
                }
                for index, date in enumerate(dates)
            ]
        ),
    )
    raw.mark_success("daily", dates[-1], row_count=42)
    raw.mark_success("adj_factor", dates[-1], row_count=42)
    registry = FeatureRegistry()
    register_builtin_features(registry)

    summary = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=FeatureCalcContext(raw),
        plugin_ids=["market.price_volume_pack.v1"],
        batch_days=20,
    ).update_to(dates[-1])

    assert summary["market.price_volume_pack.v1"].last_success_trade_date == dates[-1]
    assert {
        features.get_state(output).last_success_trade_date for output in OUTPUTS
    } == {dates[-1]}
    assert features.get_state("market.close.v1").status == "pending"


def _write_price_history(
    raw: RawStore,
    dates: list[int],
    close_a: list[float],
    close_b: list[float],
    amount_a: list[float],
    amount_b: list[float],
) -> None:
    daily_rows: list[dict[str, object]] = []
    adj_rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates):
        for ts_code, closes, amounts in [
            ("000001.SZ", close_a, amount_a),
            ("000002.SZ", close_b, amount_b),
        ]:
            close = closes[index]
            daily_rows.append(_daily_row(ts_code, trade_date, close, amounts[index]))
            adj_rows.append(
                {"ts_code": ts_code, "trade_date": trade_date, "adj_factor": 1.0}
            )
    raw.upsert("daily", pd.DataFrame(daily_rows))
    raw.upsert("adj_factor", pd.DataFrame(adj_rows))


def _append_prices(raw: RawStore, trade_date: int, closes: dict[str, float]) -> None:
    raw.upsert(
        "daily",
        pd.DataFrame(
            [_daily_row(ts_code, trade_date, close, 10.0) for ts_code, close in closes.items()]
        ),
    )
    raw.upsert(
        "adj_factor",
        pd.DataFrame(
            [
                {"ts_code": ts_code, "trade_date": trade_date, "adj_factor": 1.0}
                for ts_code in closes
            ]
        ),
    )


def _daily_row(ts_code: str, trade_date: int, close: float, amount: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "pre_close": close,
        "change": 0.0,
        "pct_chg": 0.0,
        "vol": 1000.0,
        "amount": amount,
    }
