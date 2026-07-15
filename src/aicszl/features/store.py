from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class FeatureMeta:
    feature_name: str
    domain: str
    version: str
    kind: str
    owner_plugin: str
    input_tables: list[str]
    lookback_days: int
    code_hash: str
    status: str = "active"
    description: str = ""


@dataclass(frozen=True)
class FeatureUpdateState:
    feature_name: str
    start_date: int
    last_success_trade_date: int | None
    last_attempt_trade_date: int | None
    status: str
    input_data_max_date: int | None
    row_count: int
    error_message: str


class FeatureStore:
    def __init__(self, db_path: str | Path, start_date: int, read_only: bool = False):
        self.db_path = Path(db_path)
        self.start_date = int(start_date)
        self.read_only = bool(read_only)
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        if not self.read_only:
            self._init_tables()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def upsert_feature_values(self, df: pd.DataFrame) -> int:
        columns = ["ts_code", "trade_date", "feature_name", "value"]
        if df.empty:
            return 0
        missing = [name for name in columns if name not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for feature_values: {missing}")

        normalized = df[columns].copy()
        normalized["feature_version"] = normalized["feature_name"].map(_feature_version)
        self.conn.register("feature_values_upsert_df", normalized)
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO feature_values (
                    ts_code, trade_date, feature_name, value, feature_version, updated_at
                )
                SELECT
                    ts_code, trade_date, feature_name, value, feature_version, current_timestamp
                FROM feature_values_upsert_df
                """
            )
        finally:
            self.conn.unregister("feature_values_upsert_df")
        return int(len(normalized))

    def upsert_target_values(self, df: pd.DataFrame) -> int:
        columns = ["ts_code", "trade_date", "target_name", "value"]
        if df.empty:
            return 0
        missing = [name for name in columns if name not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for target_values: {missing}")

        normalized = df[columns].copy()
        self.conn.register("target_values_upsert_df", normalized)
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO target_values (
                    ts_code, trade_date, target_name, value, updated_at
                )
                SELECT
                    ts_code, trade_date, target_name, value, current_timestamp
                FROM target_values_upsert_df
                """
            )
        finally:
            self.conn.unregister("target_values_upsert_df")
        return int(len(normalized))

    def register_feature_meta(self, meta: FeatureMeta) -> None:
        existing = self.conn.execute(
            """
            SELECT code_hash
            FROM feature_meta
            WHERE feature_name = ?
            """,
            [meta.feature_name],
        ).fetchone()
        if existing is not None and str(existing[0]) != meta.code_hash:
            raise ValueError(
                f"feature {meta.feature_name} code_hash mismatch: "
                f"stored={existing[0]} current={meta.code_hash}"
            )

        existing_created_at = self.conn.execute(
            """
            SELECT created_at
            FROM feature_meta
            WHERE feature_name = ?
            """,
            [meta.feature_name],
        ).fetchone()
        created_at_sql = "?" if existing_created_at is not None else "current_timestamp"
        params: list[object] = [
            meta.feature_name,
            meta.domain,
            meta.version,
            meta.kind,
            meta.owner_plugin,
            ",".join(meta.input_tables),
            int(meta.lookback_days),
            meta.code_hash,
            meta.status,
            meta.description,
        ]
        if existing_created_at is not None:
            params.append(existing_created_at[0])
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO feature_meta (
                feature_name, domain, version, kind, owner_plugin, input_tables,
                lookback_days, code_hash, status, description, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {created_at_sql}, current_timestamp)
            """,
            params,
        )

    def get_feature_statuses(self, feature_names: list[str]) -> dict[str, str]:
        if not feature_names:
            return {}
        placeholders = ", ".join("?" for _ in feature_names)
        rows = self.conn.execute(
            f"""
            SELECT feature_name, status
            FROM feature_meta
            WHERE feature_name IN ({placeholders})
            """,
            list(feature_names),
        ).fetchall()
        return {str(feature_name): str(status) for feature_name, status in rows}

    def feature_date_coverage(self, feature_names: list[str]) -> dict[str, set[int]]:
        coverage = {feature_name: set() for feature_name in feature_names}
        if not feature_names:
            return coverage
        placeholders = ", ".join("?" for _ in feature_names)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT feature_name, trade_date
            FROM feature_values
            WHERE feature_name IN ({placeholders})
            """,
            list(feature_names),
        ).fetchall()
        for feature_name, trade_date in rows:
            coverage[str(feature_name)].add(int(trade_date))
        return coverage

    def get_state(self, feature_name: str) -> FeatureUpdateState:
        row = self.conn.execute(
            """
            SELECT feature_name, start_date, last_success_trade_date,
                   last_attempt_trade_date, status, input_data_max_date,
                   row_count, error_message
            FROM feature_update_state
            WHERE feature_name = ?
            """,
            [feature_name],
        ).fetchone()
        if row is None:
            return FeatureUpdateState(feature_name, self.start_date, None, None, "pending", None, 0, "")
        return FeatureUpdateState(
            feature_name=str(row[0]),
            start_date=int(row[1]),
            last_success_trade_date=_none_or_int(row[2]),
            last_attempt_trade_date=_none_or_int(row[3]),
            status=str(row[4]),
            input_data_max_date=_none_or_int(row[5]),
            row_count=int(row[6]),
            error_message=str(row[7] or ""),
        )

    def mark_attempt(self, feature_name: str, trade_date: int, input_data_max_date: int | None) -> None:
        state = self.get_state(feature_name)
        self._replace_state(
            feature_name=feature_name,
            last_success_trade_date=state.last_success_trade_date,
            last_attempt_trade_date=trade_date,
            status="running",
            input_data_max_date=input_data_max_date,
            row_count=state.row_count,
            error_message="",
        )

    def mark_success(
        self,
        feature_name: str,
        trade_date: int,
        input_data_max_date: int | None,
        row_count: int,
    ) -> None:
        self._replace_state(
            feature_name=feature_name,
            last_success_trade_date=trade_date,
            last_attempt_trade_date=trade_date,
            status="success",
            input_data_max_date=input_data_max_date,
            row_count=row_count,
            error_message="",
        )

    def mark_failure(self, feature_name: str, trade_date: int, error_message: str) -> None:
        state = self.get_state(feature_name)
        self._replace_state(
            feature_name=feature_name,
            last_success_trade_date=state.last_success_trade_date,
            last_attempt_trade_date=trade_date,
            status="failed",
            input_data_max_date=state.input_data_max_date,
            row_count=state.row_count,
            error_message=error_message,
        )

    def fetch_df(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        return self.conn.execute(sql, params or []).fetchdf()

    def close(self) -> None:
        self.conn.close()

    def _replace_state(
        self,
        feature_name: str,
        last_success_trade_date: int | None,
        last_attempt_trade_date: int | None,
        status: str,
        input_data_max_date: int | None,
        row_count: int,
        error_message: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO feature_update_state
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                feature_name,
                self.start_date,
                last_success_trade_date,
                last_attempt_trade_date,
                status,
                input_data_max_date,
                int(row_count),
                error_message,
            ],
        )

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_values (
                ts_code VARCHAR NOT NULL,
                trade_date INTEGER NOT NULL,
                feature_name VARCHAR NOT NULL,
                value DOUBLE,
                feature_version VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (ts_code, trade_date, feature_name)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS target_values (
                ts_code VARCHAR NOT NULL,
                trade_date INTEGER NOT NULL,
                target_name VARCHAR NOT NULL,
                value DOUBLE,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (ts_code, trade_date, target_name)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_meta (
                feature_name VARCHAR PRIMARY KEY,
                domain VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                kind VARCHAR NOT NULL,
                owner_plugin VARCHAR NOT NULL,
                input_tables VARCHAR NOT NULL,
                lookback_days INTEGER NOT NULL,
                code_hash VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_update_state (
                feature_name VARCHAR PRIMARY KEY,
                start_date INTEGER NOT NULL,
                last_success_trade_date INTEGER,
                last_attempt_trade_date INTEGER,
                status VARCHAR NOT NULL,
                input_data_max_date INTEGER,
                row_count INTEGER NOT NULL,
                error_message VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )


def _feature_version(feature_name: str) -> str:
    parts = feature_name.split(".")
    if len(parts) != 3:
        raise ValueError(f"Feature name must use domain.name.version: {feature_name}")
    return parts[2]


def _none_or_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
