from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aicszl.raw import RawStore


SCORE_DATASET_COLUMNS = [
    "trade_date",
    "ts_code",
    "score",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "is_tradable",
    "limit_up",
    "limit_down",
]
_BLEND_COLUMNS = {"ts_code", "trade_date", "score_raw_blend"}


@dataclass(frozen=True)
class BacktestDatasetArtifact:
    dataset_id: str
    dataset_path: Path
    rows: int


def build_score_dataset(
    store: RawStore,
    blend_path: str | Path,
    output_dir: str | Path,
) -> BacktestDatasetArtifact:
    source_path = Path(blend_path)
    blend = pd.read_pickle(source_path)
    _require_columns(blend, _BLEND_COLUMNS)
    if blend.empty:
        raise ValueError("Blend must not be empty")
    start_date = int(blend["trade_date"].min())
    end_date = int(blend["trade_date"].max())
    market = store.fetch_df(
        """
        SELECT ts_code, trade_date, open, high, low, close, vol, amount
        FROM daily
        WHERE trade_date BETWEEN ? AND ?
        """,
        [start_date, end_date],
    )
    result = blend.merge(market, on=["ts_code", "trade_date"], how="inner")
    if result.empty:
        raise ValueError("Blend has no matching daily market data")
    limits = store.fetch_df(
        """
        SELECT ts_code, trade_date, up_limit, down_limit
        FROM stk_limit
        WHERE trade_date BETWEEN ? AND ?
        """,
        [start_date, end_date],
    )
    result = result.merge(limits, on=["ts_code", "trade_date"], how="left")
    result["score"] = result["score_raw_blend"]
    result["is_tradable"] = result[["open", "high", "low", "close", "vol"]].notna().all(
        axis=1
    ) & result["vol"].gt(0)
    result = result.rename(columns={"up_limit": "limit_up", "down_limit": "limit_down"})
    result = result[SCORE_DATASET_COLUMNS].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    dataset_id = f"{source_path.stem}__{_file_hash(source_path)}"
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = destination_dir / f"{dataset_id}.pkl"
    result.to_pickle(dataset_path)
    return BacktestDatasetArtifact(dataset_id=dataset_id, dataset_path=dataset_path, rows=int(len(result)))


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Blend is missing required columns: {', '.join(missing)}")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]
