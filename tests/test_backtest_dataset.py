from pathlib import Path

import pandas as pd
import pytest

from aicszl.backtests.dataset import SCORE_DATASET_COLUMNS, build_score_dataset
from aicszl.raw import RawStore


def test_build_score_dataset_joins_market_and_limit_data(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    try:
        _seed_daily(store)
        _seed_limits(store)
        blend_path = tmp_path / "blend.pkl"
        pd.DataFrame(
            [
                _blend("000001.SZ", 20200102, 0.9),
                _blend("000002.SZ", 20200102, 0.1),
                _blend("000003.SZ", 20200102, 0.5),
            ]
        ).to_pickle(blend_path)

        artifact = build_score_dataset(store, blend_path, tmp_path / "artifacts" / "backtests")

        result = pd.read_pickle(artifact.dataset_path)
        assert result.columns.tolist() == SCORE_DATASET_COLUMNS
        assert artifact.rows == 2
        first, second = result.to_dict("records")
        assert first == {
            "trade_date": 20200102,
            "ts_code": "000001.SZ",
            "score": 0.9,
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "vol": 100.0,
            "amount": 1000.0,
            "is_tradable": True,
            "limit_up": 11.0,
            "limit_down": 9.0,
        }
        assert second["ts_code"] == "000002.SZ"
        assert second["is_tradable"] is False
        assert pd.isna(second["limit_up"])
        assert pd.isna(second["limit_down"])
    finally:
        store.close()


def test_build_score_dataset_rejects_blend_without_score(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    try:
        blend_path = tmp_path / "blend.pkl"
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": 20200102}]).to_pickle(blend_path)

        with pytest.raises(ValueError, match="score_raw_blend"):
            build_score_dataset(store, blend_path, tmp_path / "artifacts" / "backtests")
    finally:
        store.close()


def test_build_score_dataset_rejects_empty_market_join(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    try:
        _seed_daily(store)
        blend_path = tmp_path / "blend.pkl"
        pd.DataFrame([_blend("000003.SZ", 20200102, 0.9)]).to_pickle(blend_path)

        with pytest.raises(ValueError, match="no matching daily market data"):
            build_score_dataset(store, blend_path, tmp_path / "artifacts" / "backtests")
    finally:
        store.close()


def test_build_score_dataset_limits_raw_queries_to_blend_date_range(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    calls: list[tuple[str, list[object]]] = []

    class RecordingStore:
        def fetch_df(self, sql: str, params: list[object] | None = None):
            calls.append((sql, list(params or [])))
            return store.fetch_df(sql, params)

    try:
        _seed_daily(store)
        _seed_limits(store)
        blend_path = tmp_path / "blend.pkl"
        pd.DataFrame([_blend("000001.SZ", 20200102, 0.9)]).to_pickle(blend_path)

        build_score_dataset(
            RecordingStore(), blend_path, tmp_path / "artifacts" / "backtests"
        )

        assert len(calls) == 2
        assert all("trade_date BETWEEN ? AND ?" in sql for sql, _ in calls)
        assert all(params == [20200102, 20200102] for _, params in calls)
    finally:
        store.close()


def _seed_daily(store: RawStore) -> None:
    store.upsert(
        "daily",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "pre_close": 10.0,
                    "change": 0.5,
                    "pct_chg": 5.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": 20200102,
                    "open": 20.0,
                    "high": 20.5,
                    "low": 19.5,
                    "close": 20.0,
                    "pre_close": 20.0,
                    "change": 0.0,
                    "pct_chg": 0.0,
                    "vol": 0.0,
                    "amount": 0.0,
                },
            ]
        ),
    )


def _seed_limits(store: RawStore) -> None:
    store.upsert(
        "stk_limit",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": 20200102,
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                }
            ]
        ),
    )


def _blend(ts_code: str, trade_date: int, score: float) -> dict[str, object]:
    return {"ts_code": ts_code, "trade_date": trade_date, "score_raw_blend": score}
