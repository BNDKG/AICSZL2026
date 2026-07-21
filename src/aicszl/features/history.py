from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aicszl.raw.store import RawStore


def fetch_bounded_history(
    raw_store: RawStore,
    *,
    source_sql: str,
    columns: list[str],
    dates: list[int],
    lookback_rows: int,
    source_params: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Load target dates plus at most N prior observations per participating stock."""
    _validate_columns(columns)
    target_dates = sorted({int(value) for value in dates})
    if not target_dates:
        return pd.DataFrame(columns=columns)

    selected = ", ".join(f"s.{_qid(column)}" for column in columns)
    projected = ", ".join(f"n.{_qid(column)}" for column in columns)
    placeholders = ", ".join("?" for _ in target_dates)
    frame = raw_store.fetch_df(
        f"""
        WITH source_data AS (
            {source_sql}
        ),
        numbered AS (
            SELECT {selected},
                   row_number() OVER (
                       PARTITION BY s.ts_code ORDER BY s.trade_date
                   ) AS observation_number
            FROM source_data s
            WHERE s.trade_date <= ?
        ),
        first_target AS (
            SELECT ts_code, min(observation_number) AS first_target_observation
            FROM numbered
            WHERE trade_date IN ({placeholders})
            GROUP BY ts_code
        )
        SELECT {projected}
        FROM numbered n
        JOIN first_target t USING (ts_code)
        WHERE n.observation_number >= t.first_target_observation - ?
        ORDER BY n.ts_code, n.trade_date
        """,
        [
            *(source_params or []),
            target_dates[-1],
            *target_dates,
            max(0, int(lookback_rows)),
        ],
    )
    if list(frame.columns) != columns:
        raise ValueError(f"Bounded history columns must be exactly {columns}")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Bounded history contains duplicate (ts_code, trade_date) keys")
    return frame


def _validate_columns(columns: list[str]) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError("Bounded history columns must not contain duplicates")
    missing = [name for name in ("ts_code", "trade_date") if name not in columns]
    if missing:
        raise ValueError(f"Bounded history columns missing keys: {missing}")


def _qid(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
