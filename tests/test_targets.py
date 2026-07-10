from pathlib import Path

import pandas as pd

from aicszl.features.store import FeatureStore
from aicszl.raw.store import RawStore
from aicszl.targets.builtins import TargetCalcContext, calc_ret_5d_rank_pct


def test_target_values_are_stored_separately_from_feature_values(tmp_path: Path):
    store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    first = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": 20200102,
                "target_name": "target.ret_5d_rank_pct.v1",
                "value": 0.25,
            }
        ]
    )
    replacement = first.copy()
    replacement.loc[0, "value"] = 0.75

    assert store.upsert_target_values(first) == 1
    assert store.upsert_target_values(replacement) == 1

    targets = store.fetch_df("SELECT ts_code, trade_date, target_name, value FROM target_values")
    features = store.fetch_df("SELECT count(*) AS n FROM feature_values")
    assert targets.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "target_name": "target.ret_5d_rank_pct.v1",
            "value": 0.75,
        }
    ]
    assert int(features.loc[0, "n"]) == 0


def test_ret_5d_rank_pct_target_uses_forward_adjusted_returns(tmp_path: Path):
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
    ]:
        rows.extend(
            [
                _daily_row("000001.SZ", trade_date, close=close_a),
                _daily_row("000002.SZ", trade_date, close=close_b),
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

    result = calc_ret_5d_rank_pct(TargetCalcContext(raw), [20200102])

    assert result.sort_values("ts_code").to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "target_name": "target.ret_5d_rank_pct.v1",
            "value": 1.0,
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "target_name": "target.ret_5d_rank_pct.v1",
            "value": 0.5,
        },
    ]


def _daily_row(ts_code: str, trade_date: int, close: float) -> dict[str, object]:
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
        "amount": 1000.0,
    }
