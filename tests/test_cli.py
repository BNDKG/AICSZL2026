from aicszl.cli import main


def test_cli_help_exits_successfully(capsys):
    exit_code = main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "AICSZL2026 research workflow" in captured.out
    assert "raw" in captured.out
    assert "feature" in captured.out
    assert "train" in captured.out


def test_raw_update_dry_run_reports_target_date(capsys):
    exit_code = main(["raw", "update", "--to", "20260708", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "raw update dry-run" in captured.out
    assert "20260708" in captured.out


def test_raw_update_uses_config_and_prints_summary(monkeypatch, capsys, tmp_path):
    calls = {}

    class FakeSettings:
        class project:
            start_date = 20200101

        class paths:
            raw_db = tmp_path / "raw.duckdb"

        class tushare:
            token_file = tmp_path / "token.txt"

    class FakeStore:
        def __init__(self, db_path, start_date):
            calls["store"] = (db_path, start_date)

    class FakeClient:
        @classmethod
        def from_token_file(cls, token_file):
            calls["token_file"] = token_file
            return cls()

    class FakeState:
        last_success_trade_date = 20260708
        row_count = 42
        status = "success"

    class FakeUpdater:
        def __init__(
            self,
            store,
            client,
            tables,
            batch_days,
            max_retries,
            retry_sleep_seconds,
            on_progress=None,
        ):
            calls["tables"] = tables
            calls["batch_days"] = batch_days
            calls["max_retries"] = max_retries
            calls["retry_sleep_seconds"] = retry_sleep_seconds
            self.on_progress = on_progress

        def update_to(self, target_date):
            calls["target_date"] = target_date
            self.on_progress(
                {
                    "event": "fetch",
                    "table": "daily",
                    "trade_date": 20260708,
                    "rows": 42,
                    "fetch_ms": 123,
                }
            )
            self.on_progress(
                {
                    "event": "commit",
                    "table": "daily",
                    "start_trade_date": 20260708,
                    "end_trade_date": 20260708,
                    "rows": 42,
                    "dates": 1,
                    "commit_ms": 45,
                }
            )
            return {"daily": FakeState()}

    monkeypatch.setattr("aicszl.cli.load_settings", lambda path: FakeSettings(), raising=False)
    monkeypatch.setattr("aicszl.cli.RawStore", FakeStore, raising=False)
    monkeypatch.setattr("aicszl.cli.TushareRawClient", FakeClient, raising=False)
    monkeypatch.setattr("aicszl.cli.RawUpdater", FakeUpdater, raising=False)

    exit_code = main(
        [
            "raw",
            "update",
            "--to",
            "20260708",
            "--config",
            str(tmp_path / "settings.yaml"),
            "--tables",
            "daily",
            "--batch-days",
            "30",
            "--retries",
            "5",
            "--retry-sleep-ms",
            "250",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["store"] == (tmp_path / "raw.duckdb", 20200101)
    assert calls["token_file"] == tmp_path / "token.txt"
    assert calls["tables"] == ["daily"]
    assert calls["batch_days"] == 30
    assert calls["max_retries"] == 5
    assert calls["retry_sleep_seconds"] == 0.25
    assert calls["target_date"] == 20260708
    assert "[raw] fetch daily 20260708 rows=42 fetch_ms=123" in captured.out
    assert "[raw] commit daily 20260708-20260708 dates=1 rows=42 commit_ms=45" in captured.out
    assert "daily success 20260708 rows=42" in captured.out


def test_raw_update_prints_friendly_error(monkeypatch, capsys, tmp_path):
    class FakeSettings:
        class project:
            start_date = 20200101

        class paths:
            raw_db = tmp_path / "raw.duckdb"

        class tushare:
            token_file = tmp_path / "token.txt"

    class FakeStore:
        def __init__(self, db_path, start_date):
            pass

        def close(self):
            pass

    class FakeClient:
        @classmethod
        def from_token_file(cls, token_file):
            return cls()

    class FakeUpdater:
        def __init__(
            self,
            store,
            client,
            tables,
            batch_days,
            max_retries,
            retry_sleep_seconds,
            on_progress=None,
        ):
            pass

        def update_to(self, target_date):
            raise RuntimeError("raw update failed table=moneyflow trade_date=20200102: boom")

    monkeypatch.setattr("aicszl.cli.load_settings", lambda path: FakeSettings(), raising=False)
    monkeypatch.setattr("aicszl.cli.RawStore", FakeStore, raising=False)
    monkeypatch.setattr("aicszl.cli.TushareRawClient", FakeClient, raising=False)
    monkeypatch.setattr("aicszl.cli.RawUpdater", FakeUpdater, raising=False)

    exit_code = main(["raw", "update", "--to", "20200102", "--tables", "moneyflow"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR: raw update failed table=moneyflow trade_date=20200102: boom" in captured.err
