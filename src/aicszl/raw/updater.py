from __future__ import annotations

from collections.abc import Callable
from time import perf_counter, sleep
from typing import Protocol

import pandas as pd

from .store import RawStore, RawUpdateState


class RawDataClient(Protocol):
    def trade_dates(self, start_date: int, end_date: int) -> list[int]:
        ...

    def fetch_table(self, table_name: str, trade_date: int, end_date: int | None = None) -> pd.DataFrame:
        ...


class RawUpdater:
    def __init__(
        self,
        store: RawStore,
        client: RawDataClient,
        tables: list[str],
        batch_days: int = 20,
        max_retries: int = 3,
        retry_sleep_seconds: float = 1.0,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ):
        self.store = store
        self.client = client
        self.tables = list(tables)
        self.batch_days = max(1, int(batch_days))
        self.max_retries = max(0, int(max_retries))
        self.retry_sleep_seconds = max(0.0, float(retry_sleep_seconds))
        self.on_progress = on_progress

    def update_to(self, target_date: int) -> dict[str, RawUpdateState]:
        summary: dict[str, RawUpdateState] = {}
        for table_name in self.tables:
            state = self.store.get_state(table_name)
            start_date = state.last_success_trade_date + 1 if state.last_success_trade_date else state.start_date
            if table_name == "trade_cal":
                summary[table_name] = self._update_trade_cal(start_date, int(target_date))
                continue
            batch: list[tuple[int, pd.DataFrame]] = []
            for trade_date in self.client.trade_dates(start_date, int(target_date)):
                self.store.mark_attempt(table_name, trade_date)
                try:
                    fetch_started = perf_counter()
                    df = self._fetch_with_retries(table_name, trade_date)
                    fetch_ms = int((perf_counter() - fetch_started) * 1000)
                    self._emit(
                        {
                            "event": "fetch",
                            "table": table_name,
                            "trade_date": trade_date,
                            "rows": len(df),
                            "fetch_ms": fetch_ms,
                        }
                    )
                    batch.append((trade_date, df))
                    if len(batch) >= self.batch_days:
                        self._commit_batch(table_name, batch)
                        batch = []
                except Exception as exc:
                    self.store.mark_failure(table_name, trade_date, str(exc))
                    self._emit(
                        {
                            "event": "failed",
                            "table": table_name,
                            "trade_date": trade_date,
                            "error": str(exc),
                        }
                    )
                    raise RuntimeError(
                        f"raw update failed table={table_name} trade_date={trade_date}: {exc}"
                    ) from exc
            if batch:
                self._commit_batch(table_name, batch)
            summary[table_name] = self.store.get_state(table_name)
        return summary

    def _commit_batch(self, table_name: str, batch: list[tuple[int, pd.DataFrame]]) -> None:
        frames = [df for _, df in batch]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        commit_started = perf_counter()
        row_count = self.store.upsert(table_name, merged)
        last_trade_date = batch[-1][0]
        self.store.mark_success(table_name, last_trade_date, row_count=row_count)
        commit_ms = int((perf_counter() - commit_started) * 1000)
        self._emit(
            {
                "event": "commit",
                "table": table_name,
                "start_trade_date": batch[0][0],
                "end_trade_date": last_trade_date,
                "rows": row_count,
                "dates": len(batch),
                "commit_ms": commit_ms,
            }
        )

    def _update_trade_cal(self, start_date: int, target_date: int) -> RawUpdateState:
        table_name = "trade_cal"
        if start_date > target_date:
            return self.store.get_state(table_name)
        self.store.mark_attempt(table_name, target_date)
        try:
            fetch_started = perf_counter()
            df = self._fetch_with_retries(table_name, start_date, target_date)
            fetch_ms = int((perf_counter() - fetch_started) * 1000)
            self._emit(
                {
                    "event": "fetch",
                    "table": table_name,
                    "trade_date": target_date,
                    "rows": len(df),
                    "fetch_ms": fetch_ms,
                }
            )
            commit_started = perf_counter()
            row_count = self.store.upsert(table_name, df)
            self.store.mark_success(table_name, target_date, row_count=row_count)
            commit_ms = int((perf_counter() - commit_started) * 1000)
            self._emit(
                {
                    "event": "commit",
                    "table": table_name,
                    "start_trade_date": start_date,
                    "end_trade_date": target_date,
                    "rows": row_count,
                    "dates": 1,
                    "commit_ms": commit_ms,
                }
            )
        except Exception as exc:
            self.store.mark_failure(table_name, target_date, str(exc))
            self._emit(
                {
                    "event": "failed",
                    "table": table_name,
                    "trade_date": target_date,
                    "error": str(exc),
                }
            )
            raise RuntimeError(
                f"raw update failed table={table_name} trade_date={target_date}: {exc}"
            ) from exc
        return self.store.get_state(table_name)

    def _fetch_with_retries(
        self,
        table_name: str,
        trade_date: int,
        end_date: int | None = None,
    ) -> pd.DataFrame:
        attempt = 0
        while True:
            try:
                if end_date is None:
                    return self.client.fetch_table(table_name, trade_date)
                return self.client.fetch_table(table_name, trade_date, end_date)
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self._emit(
                    {
                        "event": "retry",
                        "table": table_name,
                        "trade_date": trade_date,
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error": str(exc),
                        "sleep_seconds": self.retry_sleep_seconds,
                    }
                )
                if self.retry_sleep_seconds > 0:
                    sleep(self.retry_sleep_seconds)

    def _emit(self, event: dict[str, object]) -> None:
        if self.on_progress is not None:
            self.on_progress(event)
