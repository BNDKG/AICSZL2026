from pathlib import Path

import pandas as pd
import pytest

from aicszl.raw.store import RawStore
from aicszl.raw.updater import RawUpdater


class FakeTushareClient:
    def __init__(self, fail_on: int | None = None):
        self.fail_on = fail_on
        self.fetch_calls: list[int] = []

    def trade_dates(self, start_date: int, end_date: int) -> list[int]:
        dates = [20200102, 20200103, 20200106]
        return [date for date in dates if start_date <= date <= end_date]

    def fetch_table(self, table_name: str, trade_date: int) -> pd.DataFrame:
        assert table_name == "daily"
        self.fetch_calls.append(trade_date)
        if trade_date == self.fail_on:
            raise RuntimeError(f"boom on {trade_date}")
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "pre_close": 10.0,
                    "change": 0.5,
                    "pct_chg": 5.0,
                    "vol": 1000.0,
                    "amount": 10000.0,
                }
            ]
        )


class FlakyTushareClient(FakeTushareClient):
    def __init__(self, fail_first_on: int):
        super().__init__()
        self.fail_first_on = fail_first_on
        self.failures: set[int] = set()

    def fetch_table(self, table_name: str, trade_date: int) -> pd.DataFrame:
        self.fetch_calls.append(trade_date)
        if trade_date == self.fail_first_on and trade_date not in self.failures:
            self.failures.add(trade_date)
            raise RuntimeError(f"temporary boom on {trade_date}")
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "pre_close": 10.0,
                    "change": 0.5,
                    "pct_chg": 5.0,
                    "vol": 1000.0,
                    "amount": 10000.0,
                }
            ]
        )


class MultiTableFakeClient:
    def __init__(self):
        self.calls: list[tuple[str, int, int | None]] = []

    def trade_dates(self, start_date: int, end_date: int) -> list[int]:
        return [date for date in [20200102] if start_date <= date <= end_date]

    def fetch_table(self, table_name: str, trade_date: int, end_date: int | None = None) -> pd.DataFrame:
        self.calls.append((table_name, trade_date, end_date))
        if table_name == "trade_cal":
            return pd.DataFrame(
                [
                    {"cal_date": 20200101, "exchange": "SSE", "is_open": 0, "pretrade_date": 20191231},
                    {"cal_date": 20200102, "exchange": "SSE", "is_open": 1, "pretrade_date": 20191231},
                ]
            )
        if table_name == "adj_factor":
            return pd.DataFrame(
                [{"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.25}]
            )
        if table_name == "daily_basic":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
                        "turnover_rate": 1.0,
                        "volume_ratio": 2.0,
                        "pe": 3.0,
                        "pb": 4.0,
                        "ps_ttm": 5.0,
                        "dv_ttm": 6.0,
                        "circ_mv": 7.0,
                        "total_mv": 8.0,
                    }
                ]
            )
        raise AssertionError(table_name)


def test_raw_updater_commits_each_trade_date_and_advances_watermark(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    client = FakeTushareClient()
    updater = RawUpdater(store=store, client=client, tables=["daily"], batch_days=1, max_retries=0)

    summary = updater.update_to(20200106)

    assert client.fetch_calls == [20200102, 20200103, 20200106]
    assert summary["daily"].last_success_trade_date == 20200106
    assert store.get_state("daily").last_success_trade_date == 20200106
    rows = store.fetch_df("SELECT count(*) AS n FROM daily")
    assert int(rows.loc[0, "n"]) == 3


def test_raw_updater_keeps_successful_progress_when_later_date_fails(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    client = FakeTushareClient(fail_on=20200103)
    updater = RawUpdater(store=store, client=client, tables=["daily"], batch_days=1, max_retries=0)

    with pytest.raises(RuntimeError, match="boom"):
        updater.update_to(20200106)

    assert client.fetch_calls == [20200102, 20200103]
    state = store.get_state("daily")
    assert state.last_success_trade_date == 20200102
    assert state.last_attempt_trade_date == 20200103
    assert state.status == "failed"
    rows = store.fetch_df("SELECT trade_date FROM daily ORDER BY trade_date")
    assert rows["trade_date"].tolist() == [20200102]


def test_raw_updater_retries_transient_fetch_failure(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    events = []
    client = FlakyTushareClient(fail_first_on=20200103)
    updater = RawUpdater(
        store=store,
        client=client,
        tables=["daily"],
        batch_days=1,
        max_retries=2,
        retry_sleep_seconds=0,
        on_progress=events.append,
    )

    summary = updater.update_to(20200103)

    assert client.fetch_calls == [20200102, 20200103, 20200103]
    assert summary["daily"].last_success_trade_date == 20200103
    rows = store.fetch_df("SELECT trade_date FROM daily ORDER BY trade_date")
    assert rows["trade_date"].tolist() == [20200102, 20200103]
    assert any(
        event["event"] == "retry"
        and event["table"] == "daily"
        and event["trade_date"] == 20200103
        and event["attempt"] == 1
        for event in events
    )


def test_raw_updater_updates_trade_calendar_as_range(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    client = MultiTableFakeClient()
    updater = RawUpdater(store=store, client=client, tables=["trade_cal"])

    summary = updater.update_to(20200102)

    assert client.calls == [("trade_cal", 20200101, 20200102)]
    assert summary["trade_cal"].last_success_trade_date == 20200102
    rows = store.fetch_df('SELECT cal_date, is_open FROM "trade_cal" ORDER BY cal_date')
    assert rows.to_dict("records") == [
        {"cal_date": 20200101, "is_open": 0},
        {"cal_date": 20200102, "is_open": 1},
    ]


def test_raw_updater_updates_multiple_tables_with_independent_state(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    client = MultiTableFakeClient()
    updater = RawUpdater(store=store, client=client, tables=["adj_factor", "daily_basic"])

    summary = updater.update_to(20200102)

    assert [call[0] for call in client.calls] == ["adj_factor", "daily_basic"]
    assert summary["adj_factor"].last_success_trade_date == 20200102
    assert summary["daily_basic"].last_success_trade_date == 20200102
    assert int(store.fetch_df("SELECT count(*) AS n FROM adj_factor").loc[0, "n"]) == 1
    assert int(store.fetch_df("SELECT count(*) AS n FROM daily_basic").loc[0, "n"]) == 1


def test_raw_updater_emits_progress_events_for_success_and_failure(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    events = []
    client = FakeTushareClient(fail_on=20200103)
    updater = RawUpdater(
        store=store,
        client=client,
        tables=["daily"],
        batch_days=1,
        max_retries=0,
        on_progress=events.append,
    )

    with pytest.raises(RuntimeError):
        updater.update_to(20200106)

    assert [event["event"] for event in events] == ["fetch", "commit", "failed"]
    assert events[0]["table"] == "daily"
    assert events[0]["trade_date"] == 20200102
    assert events[0]["rows"] == 1
    assert "fetch_ms" in events[0]
    assert events[1]["start_trade_date"] == 20200102
    assert events[1]["end_trade_date"] == 20200102
    assert events[1]["rows"] == 1
    assert "commit_ms" in events[1]
    assert events[2] == {
        "event": "failed",
        "table": "daily",
        "trade_date": 20200103,
        "error": "boom on 20200103",
    }


def test_raw_updater_commits_by_batch_not_each_trade_date(tmp_path: Path):
    store = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    client = FakeTushareClient(fail_on=20200106)
    updater = RawUpdater(store=store, client=client, tables=["daily"], batch_days=2, max_retries=0)

    with pytest.raises(RuntimeError):
        updater.update_to(20200106)

    state = store.get_state("daily")
    assert state.last_success_trade_date == 20200103
    assert state.last_attempt_trade_date == 20200106
    assert state.status == "failed"
    rows = store.fetch_df("SELECT trade_date FROM daily ORDER BY trade_date")
    assert rows["trade_date"].tolist() == [20200102, 20200103]
