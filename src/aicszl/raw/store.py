from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .schemas import RAW_TABLES


@dataclass(frozen=True)
class RawUpdateState:
    table_name: str
    start_date: int
    last_success_trade_date: int | None
    last_attempt_trade_date: int | None
    status: str
    row_count: int
    error_message: str


class RawStore:
    def __init__(self, db_path: str | Path, start_date: int, read_only: bool = False):
        self.db_path = Path(db_path)
        self.start_date = int(start_date)
        self.read_only = bool(read_only)
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        if not self.read_only:
            self._init_state_table()

    def upsert(self, table_name: str, df: pd.DataFrame) -> int:
        spec = self._spec(table_name)
        self._ensure_raw_table(table_name)
        if df.empty:
            return 0

        missing = [name for name in spec.column_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {table_name}: {missing}")

        normalized = df[spec.column_names].copy()
        self.conn.register("raw_upsert_df", normalized)
        try:
            columns = ", ".join(_quote(name) for name in spec.column_names)
            self.conn.execute(
                f"INSERT OR REPLACE INTO {_quote(table_name)} ({columns}) "
                f"SELECT {columns} FROM raw_upsert_df"
            )
        finally:
            self.conn.unregister("raw_upsert_df")
        return int(len(normalized))

    def fetch_df(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        return self.conn.execute(sql, params or []).fetchdf()

    def get_state(self, table_name: str) -> RawUpdateState:
        row = self.conn.execute(
            """
            SELECT table_name, start_date, last_success_trade_date,
                   last_attempt_trade_date, status, row_count, error_message
            FROM raw_update_state
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        if row is None:
            return RawUpdateState(table_name, self.start_date, None, None, "pending", 0, "")
        return RawUpdateState(
            table_name=str(row[0]),
            start_date=int(row[1]),
            last_success_trade_date=_none_or_int(row[2]),
            last_attempt_trade_date=_none_or_int(row[3]),
            status=str(row[4]),
            row_count=int(row[5]),
            error_message=str(row[6] or ""),
        )

    def mark_attempt(self, table_name: str, trade_date: int) -> None:
        state = self.get_state(table_name)
        self._replace_state(
            table_name=table_name,
            last_success_trade_date=state.last_success_trade_date,
            last_attempt_trade_date=trade_date,
            status="running",
            row_count=state.row_count,
            error_message="",
        )

    def mark_success(self, table_name: str, trade_date: int, row_count: int) -> None:
        self._replace_state(
            table_name=table_name,
            last_success_trade_date=trade_date,
            last_attempt_trade_date=trade_date,
            status="success",
            row_count=row_count,
            error_message="",
        )

    def mark_failure(self, table_name: str, trade_date: int, error_message: str) -> None:
        state = self.get_state(table_name)
        self._replace_state(
            table_name=table_name,
            last_success_trade_date=state.last_success_trade_date,
            last_attempt_trade_date=trade_date,
            status="failed",
            row_count=state.row_count,
            error_message=error_message,
        )

    def close(self) -> None:
        self.conn.close()

    def _replace_state(
        self,
        table_name: str,
        last_success_trade_date: int | None,
        last_attempt_trade_date: int | None,
        status: str,
        row_count: int,
        error_message: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO raw_update_state
            VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                table_name,
                self.start_date,
                last_success_trade_date,
                last_attempt_trade_date,
                status,
                int(row_count),
                error_message,
            ],
        )

    def _init_state_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_update_state (
                table_name VARCHAR PRIMARY KEY,
                start_date INTEGER NOT NULL,
                last_success_trade_date INTEGER,
                last_attempt_trade_date INTEGER,
                status VARCHAR NOT NULL,
                row_count INTEGER NOT NULL,
                error_message VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )

    def _ensure_raw_table(self, table_name: str) -> None:
        spec = self._spec(table_name)
        columns = ", ".join(f"{_quote(name)} {dtype}" for name, dtype in spec.columns)
        primary_key = ", ".join(_quote(name) for name in spec.primary_keys)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_quote(table_name)} "
            f"({columns}, PRIMARY KEY ({primary_key}))"
        )

    def _spec(self, table_name: str):
        if table_name not in RAW_TABLES:
            raise KeyError(f"Unknown raw table: {table_name}")
        return RAW_TABLES[table_name]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _none_or_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
