from __future__ import annotations

import re
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

    def append_plugin_values(
        self,
        plugin_id: str,
        outputs: list[str],
        frame: pd.DataFrame,
    ) -> int:
        expected = ["ts_code", "trade_date", *outputs]
        if list(frame.columns) != expected:
            raise ValueError(f"Plugin {plugin_id} columns must be exactly {expected}")
        if frame.empty:
            return 0
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError(f"Plugin {plugin_id} contains duplicate (ts_code, trade_date) keys")
        table = feature_table_name(plugin_id)
        column_sql = ", ".join(f"{_qid(output)} DOUBLE" for output in outputs)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_qid(table)} ("
            "ts_code VARCHAR NOT NULL, trade_date INTEGER NOT NULL, "
            f"{column_sql})"
        )
        ordered = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.conn.register("plugin_values_df", ordered)
        try:
            overlap = self.conn.execute(
                f"SELECT count(*) FROM {_qid(table)} existing "
                "JOIN plugin_values_df incoming USING (ts_code, trade_date)"
            ).fetchone()[0]
            if overlap:
                raise ValueError(f"Plugin {plugin_id} write overlaps {overlap} existing rows")
            columns = ", ".join(_qid(column) for column in expected)
            self.conn.execute(
                f"INSERT INTO {_qid(table)} ({columns}) "
                f"SELECT {columns} FROM plugin_values_df"
            )
        finally:
            self.conn.unregister("plugin_values_df")
        return int(len(ordered))

    def append_target_values(self, target_name: str, frame: pd.DataFrame) -> int:
        expected = ["ts_code", "trade_date", "value"]
        if list(frame.columns) != expected:
            raise ValueError(f"Target {target_name} columns must be exactly {expected}")
        if frame.empty:
            return 0
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError(f"Target {target_name} contains duplicate (ts_code, trade_date) keys")
        table = target_table_name(target_name)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_qid(table)} ("
            "ts_code VARCHAR NOT NULL, trade_date INTEGER NOT NULL, value DOUBLE)"
        )
        ordered = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.conn.register("target_values_df", ordered)
        try:
            self.conn.execute(
                f"DELETE FROM {_qid(table)} WHERE trade_date BETWEEN ? AND ?",
                [int(ordered["trade_date"].min()), int(ordered["trade_date"].max())],
            )
            self.conn.execute(
                f"INSERT INTO {_qid(table)} (ts_code, trade_date, value) "
                "SELECT ts_code, trade_date, value FROM target_values_df"
            )
        finally:
            self.conn.unregister("target_values_df")
        return int(len(ordered))

    def load_feature_frame(
        self,
        features: list[str],
        start_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        if not features:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        locations = self._feature_locations(features)
        plugins: list[str] = []
        for feature in features:
            plugin_id = locations[feature]
            if plugin_id not in plugins:
                plugins.append(plugin_id)
        aliases = {plugin_id: f"p{index}" for index, plugin_id in enumerate(plugins)}
        first = plugins[0]
        select = [f"{aliases[first]}.ts_code", f"{aliases[first]}.trade_date"]
        select.extend(
            f"{aliases[locations[feature]]}.{_qid(feature)} AS {_qid(feature)}"
            for feature in features
        )
        joins = " ".join(
            f"JOIN {_qid(feature_table_name(plugin_id))} {aliases[plugin_id]} "
            "USING (ts_code, trade_date)"
            for plugin_id in plugins[1:]
        )
        non_null = " AND ".join(
            f"{aliases[locations[feature]]}.{_qid(feature)} IS NOT NULL"
            for feature in features
        )
        sql = (
            f"SELECT {', '.join(select)} FROM {_qid(feature_table_name(first))} {aliases[first]} "
            f"{joins} WHERE {aliases[first]}.trade_date BETWEEN ? AND ? AND {non_null} "
            f"ORDER BY {aliases[first]}.trade_date, {aliases[first]}.ts_code"
        )
        return self.fetch_df(sql, [int(start_date), int(end_date)])

    def load_target_frame(
        self,
        target_name: str,
        start_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        table = target_table_name(target_name)
        if not self._table_exists(table):
            return pd.DataFrame(columns=["ts_code", "trade_date", target_name])
        return self.fetch_df(
            f"SELECT ts_code, trade_date, value AS {_qid(target_name)} "
            f"FROM {_qid(table)} WHERE trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date, ts_code",
            [int(start_date), int(end_date)],
        )

    def feature_available_dates(
        self,
        features: list[str],
        start_date: int,
        end_date: int,
    ) -> list[int]:
        if not features:
            return []
        locations = self._feature_locations(features)
        plugins: list[str] = []
        for feature in features:
            if locations[feature] not in plugins:
                plugins.append(locations[feature])
        aliases = {plugin: f"p{index}" for index, plugin in enumerate(plugins)}
        first = plugins[0]
        joins = " ".join(
            f"JOIN {_qid(feature_table_name(plugin))} {aliases[plugin]} USING (ts_code, trade_date)"
            for plugin in plugins[1:]
        )
        non_null = " AND ".join(
            f"{aliases[locations[feature]]}.{_qid(feature)} IS NOT NULL"
            for feature in features
        )
        rows = self.fetch_df(
            f"SELECT DISTINCT {aliases[first]}.trade_date "
            f"FROM {_qid(feature_table_name(first))} {aliases[first]} {joins} "
            f"WHERE {aliases[first]}.trade_date BETWEEN ? AND ? AND {non_null} "
            f"ORDER BY {aliases[first]}.trade_date",
            [int(start_date), int(end_date)],
        )
        return [int(value) for value in rows["trade_date"].tolist()]

    def feature_value_aggregates(
        self,
        features: list[str],
        start_date: int,
        end_date: int,
    ) -> dict[str, dict[str, int | str]]:
        locations = self._feature_locations(features) if features else {}
        result: dict[str, dict[str, int | str]] = {}
        for feature in features:
            table = feature_table_name(locations[feature])
            row = self.conn.execute(
                f"""
                SELECT count({_qid(feature)}),
                       CAST(bit_xor(hash(ts_code, trade_date, ?, {_qid(feature)})) AS VARCHAR),
                       CAST(sum(CAST(hash(ts_code, trade_date, ?, {_qid(feature)}) AS HUGEINT)) AS VARCHAR)
                FROM {_qid(table)}
                WHERE trade_date BETWEEN ? AND ? AND {_qid(feature)} IS NOT NULL
                """,
                [feature, feature, int(start_date), int(end_date)],
            ).fetchone()
            result[feature] = {
                "row_count": int(row[0]),
                "hash_xor": str(row[1] or "0"),
                "hash_sum": str(row[2] or "0"),
            }
        return result

    def feature_coverage(
        self,
        features: list[str],
        start_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        locations = self._feature_locations(features) if features else {}
        rows: list[dict[str, object]] = []
        for feature in features:
            table = feature_table_name(locations[feature])
            row = self.conn.execute(
                f"""
                SELECT count({_qid(feature)}) AS row_count,
                       count({_qid(feature)}) FILTER (WHERE isfinite({_qid(feature)})) AS finite_count,
                       min(trade_date) FILTER (WHERE {_qid(feature)} IS NOT NULL) AS min_date,
                       max(trade_date) FILTER (WHERE {_qid(feature)} IS NOT NULL) AS max_date,
                       count(DISTINCT trade_date) FILTER (WHERE {_qid(feature)} IS NOT NULL) AS date_count
                FROM {_qid(table)}
                WHERE trade_date BETWEEN ? AND ?
                """,
                [int(start_date), int(end_date)],
            ).fetchone()
            rows.append(
                {
                    "feature_name": feature,
                    "row_count": int(row[0]),
                    "finite_count": int(row[1]),
                    "min_date": None if row[2] is None else int(row[2]),
                    "max_date": None if row[3] is None else int(row[3]),
                    "date_count": int(row[4]),
                }
            )
        return pd.DataFrame(rows)

    def target_coverage(
        self,
        target_name: str,
        start_date: int,
        end_date: int,
    ) -> dict[str, int]:
        table = target_table_name(target_name)
        if not self._table_exists(table):
            return {"row_count": 0, "date_count": 0}
        row = self.conn.execute(
            f"""
            SELECT count(*) FILTER (WHERE isfinite(value)),
                   count(DISTINCT trade_date) FILTER (WHERE isfinite(value))
            FROM {_qid(table)}
            WHERE trade_date BETWEEN ? AND ?
            """,
            [int(start_date), int(end_date)],
        ).fetchone()
        return {"row_count": int(row[0]), "date_count": int(row[1])}

    def target_value_aggregate(
        self,
        target_name: str,
        start_date: int,
        end_date: int,
    ) -> dict[str, int | str]:
        table = target_table_name(target_name)
        if not self._table_exists(table):
            return {"row_count": 0, "hash_xor": "0", "hash_sum": "0"}
        row = self.conn.execute(
            f"""
            SELECT count(value),
                   CAST(bit_xor(hash(ts_code, trade_date, ?, value)) AS VARCHAR),
                   CAST(sum(CAST(hash(ts_code, trade_date, ?, value) AS HUGEINT)) AS VARCHAR)
            FROM {_qid(table)}
            WHERE trade_date BETWEEN ? AND ? AND value IS NOT NULL
            """,
            [target_name, target_name, int(start_date), int(end_date)],
        ).fetchone()
        return {
            "row_count": int(row[0]),
            "hash_xor": str(row[1] or "0"),
            "hash_sum": str(row[2] or "0"),
        }

    def register_feature_meta(self, meta: FeatureMeta) -> None:
        existing = self.conn.execute(
            "SELECT code_hash FROM feature_meta WHERE feature_name = ?",
            [meta.feature_name],
        ).fetchone()
        if existing is not None and str(existing[0]) != meta.code_hash:
            raise ValueError(
                f"feature {meta.feature_name} code_hash mismatch: "
                f"stored={existing[0]} current={meta.code_hash}"
            )
        created = self.conn.execute(
            "SELECT created_at FROM feature_meta WHERE feature_name = ?",
            [meta.feature_name],
        ).fetchone()
        created_at = created[0] if created is not None else None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO feature_meta (
                feature_name, domain, version, kind, owner_plugin, input_tables,
                lookback_days, code_hash, status, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, coalesce(?, current_timestamp), current_timestamp)
            """,
            [
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
                created_at,
            ],
        )

    def get_feature_statuses(self, feature_names: list[str]) -> dict[str, str]:
        if not feature_names:
            return {}
        placeholders = ", ".join("?" for _ in feature_names)
        rows = self.conn.execute(
            f"SELECT feature_name, status FROM feature_meta WHERE feature_name IN ({placeholders})",
            list(feature_names),
        ).fetchall()
        return {str(name): str(status) for name, status in rows}

    def get_feature_code_hashes(self, feature_names: list[str]) -> dict[str, str]:
        if not feature_names:
            return {}
        placeholders = ", ".join("?" for _ in feature_names)
        rows = self.conn.execute(
            f"SELECT feature_name, code_hash FROM feature_meta WHERE feature_name IN ({placeholders})",
            list(feature_names),
        ).fetchall()
        return {str(name): str(code_hash) for name, code_hash in rows}

    def reset_feature_plugin(self, plugin_id: str, feature_names: list[str]) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {_qid(feature_table_name(plugin_id))}")
        if not feature_names:
            return
        placeholders = ", ".join("?" for _ in feature_names)
        params = list(feature_names)
        self.conn.execute(
            f"DELETE FROM feature_update_state WHERE feature_name IN ({placeholders})",
            params,
        )
        self.conn.execute(
            f"DELETE FROM feature_meta WHERE feature_name IN ({placeholders})",
            params,
        )

    def feature_date_coverage(self, feature_names: list[str]) -> dict[str, set[int]]:
        coverage = {feature: set() for feature in feature_names}
        if not feature_names:
            return coverage
        locations = self._feature_locations(feature_names)
        for feature in feature_names:
            rows = self.conn.execute(
                f"SELECT trade_date FROM {_qid(feature_table_name(locations[feature]))} "
                f"WHERE {_qid(feature)} IS NOT NULL"
            ).fetchall()
            coverage[feature] = {int(row[0]) for row in rows}
        return coverage

    def get_state(self, feature_name: str) -> FeatureUpdateState:
        row = self.conn.execute(
            """
            SELECT feature_name, start_date, last_success_trade_date,
                   last_attempt_trade_date, status, input_data_max_date,
                   row_count, error_message
            FROM feature_update_state WHERE feature_name = ?
            """,
            [feature_name],
        ).fetchone()
        if row is None:
            return FeatureUpdateState(feature_name, self.start_date, None, None, "pending", None, 0, "")
        return FeatureUpdateState(
            str(row[0]), int(row[1]), _none_or_int(row[2]), _none_or_int(row[3]),
            str(row[4]), _none_or_int(row[5]), int(row[6]), str(row[7] or ""),
        )

    def mark_attempt(self, feature_name: str, trade_date: int, input_data_max_date: int | None) -> None:
        state = self.get_state(feature_name)
        self._replace_state(feature_name, state.last_success_trade_date, trade_date, "running", input_data_max_date, state.row_count, "")

    def mark_success(self, feature_name: str, trade_date: int, input_data_max_date: int | None, row_count: int) -> None:
        self._replace_state(feature_name, trade_date, trade_date, "success", input_data_max_date, row_count, "")

    def mark_failure(self, feature_name: str, trade_date: int, error_message: str) -> None:
        state = self.get_state(feature_name)
        self._replace_state(feature_name, state.last_success_trade_date, trade_date, "failed", state.input_data_max_date, state.row_count, error_message)

    def fetch_df(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        return self.conn.execute(sql, params or []).fetchdf()

    def close(self) -> None:
        self.conn.close()

    def _feature_locations(self, features: list[str]) -> dict[str, str]:
        placeholders = ", ".join("?" for _ in features)
        rows = self.conn.execute(
            f"SELECT feature_name, owner_plugin FROM feature_meta WHERE feature_name IN ({placeholders})",
            list(features),
        ).fetchall()
        result = {str(feature): str(plugin) for feature, plugin in rows}
        missing = [feature for feature in features if feature not in result]
        if missing:
            raise ValueError(f"Unknown features: {missing}")
        for feature, plugin in result.items():
            if not self._table_exists(feature_table_name(plugin)):
                raise ValueError(f"Feature table is missing for {feature}: {feature_table_name(plugin)}")
        return result

    def _table_exists(self, table_name: str) -> bool:
        return self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()[0] == 1

    def _replace_state(self, feature_name, last_success, last_attempt, status, input_max, row_count, error) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_update_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
            [feature_name, self.start_date, last_success, last_attempt, status, input_max, int(row_count), error],
        )

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_store_meta (
                schema_version INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        if self.conn.execute("SELECT count(*) FROM feature_store_meta").fetchone()[0] == 0:
            self.conn.execute("INSERT INTO feature_store_meta VALUES (2, current_timestamp)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_meta (
                feature_name VARCHAR PRIMARY KEY, domain VARCHAR NOT NULL,
                version VARCHAR NOT NULL, kind VARCHAR NOT NULL,
                owner_plugin VARCHAR NOT NULL, input_tables VARCHAR NOT NULL,
                lookback_days INTEGER NOT NULL, code_hash VARCHAR NOT NULL,
                status VARCHAR NOT NULL, description VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_update_state (
                feature_name VARCHAR PRIMARY KEY, start_date INTEGER NOT NULL,
                last_success_trade_date INTEGER, last_attempt_trade_date INTEGER,
                status VARCHAR NOT NULL, input_data_max_date INTEGER,
                row_count INTEGER NOT NULL, error_message VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )


def feature_table_name(plugin_id: str) -> str:
    return _physical_name("fv", plugin_id)


def target_table_name(target_name: str) -> str:
    return _physical_name("tv", target_name)


def _physical_name(prefix: str, semantic_name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.]+", semantic_name) is None:
        raise ValueError(f"Unsafe semantic name: {semantic_name}")
    return f"{prefix}_{semantic_name.replace('.', '_')}"


def _qid(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _none_or_int(value: object) -> int | None:
    return None if value is None else int(value)
