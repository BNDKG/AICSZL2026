from pathlib import Path
from types import SimpleNamespace
import tomllib

from aicszl.cli import main


def test_cli_help_exits_successfully(capsys):
    exit_code = main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "AICSZL2026 research workflow" in captured.out
    assert "raw" in captured.out
    assert "feature" in captured.out
    assert "train" in captured.out


def test_project_declares_aicszl_console_entry_point():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["aicszl"] == "aicszl.cli:main"


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


def test_feature_update_selects_plugins_and_prints_summary(monkeypatch, capsys, tmp_path):
    calls = _install_fake_feature_workflow(monkeypatch, tmp_path)

    exit_code = main(
        [
            "feature",
            "update",
            "--to",
            "20260710",
            "--plugins",
            "market.raw_fields.v1,limit.high_stop.v1",
            "--batch-days",
            "10",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["plugin_ids"] == ["market.raw_fields.v1", "limit.high_stop.v1"]
    assert calls["batch_days"] == 10
    assert calls["target_date"] == 20260710
    assert calls["read_only"] == [False, False]
    assert "market.raw_fields.v1 success 20260710 rows=42" in captured.out


def test_feature_update_dry_run_uses_read_only_planner(monkeypatch, capsys, tmp_path):
    calls = _install_fake_feature_workflow(monkeypatch, tmp_path)

    exit_code = main(
        [
            "feature",
            "update",
            "--to",
            "20260710",
            "--plugins",
            "market.raw_fields.v1",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["read_only"] == [True, True]
    assert calls["planned_target_date"] == 20260710
    assert "market.raw_fields.v1 pending 20250102-20260710 trade_days=370" in captured.out
    assert "target_date" not in calls


def test_feature_list_prints_plugin_outputs_and_watermark(monkeypatch, capsys, tmp_path):
    calls = _install_fake_feature_workflow(monkeypatch, tmp_path)

    exit_code = main(["feature", "list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["read_only"] == [True]
    assert "market.raw_fields.v1" in captured.out
    assert "market.close.v1,market.amount.v1" in captured.out
    assert "watermark=20241231" in captured.out


def _install_fake_feature_workflow(monkeypatch, tmp_path):
    calls = {}

    class FakeSettings:
        class project:
            start_date = 20200101

        class paths:
            raw_db = tmp_path / "raw.duckdb"
            feature_db = tmp_path / "features.duckdb"

    class FakeRawStore:
        def __init__(self, db_path, start_date, read_only=False):
            calls.setdefault("read_only", []).append(read_only)

        def close(self):
            calls["raw_closed"] = True

    class FakeFeatureStore:
        def __init__(self, db_path, start_date, read_only=False):
            calls.setdefault("read_only", []).append(read_only)

        def get_state(self, feature_name):
            return SimpleNamespace(last_success_trade_date=20241231, status="success")

        def get_feature_statuses(self, feature_names):
            return {name: "active" for name in feature_names}

        def close(self):
            calls["feature_closed"] = True

    plugin = SimpleNamespace(
        plugin_id="market.raw_fields.v1",
        outputs=["market.close.v1", "market.amount.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
    )

    class FakeRegistry:
        def plugins(self):
            return [plugin]

        def get_plugin(self, plugin_id):
            if plugin_id != plugin.plugin_id:
                raise KeyError(plugin_id)
            return plugin

    class FakeUpdater:
        def __init__(
            self,
            *,
            raw_store,
            feature_store,
            registry,
            calc_context,
            plugin_ids=None,
            batch_days=20,
            on_progress=None,
        ):
            calls["plugin_ids"] = plugin_ids
            calls["batch_days"] = batch_days

        def plan_to(self, target_date):
            calls["planned_target_date"] = target_date
            return [
                SimpleNamespace(
                    plugin_id="market.raw_fields.v1",
                    status="pending",
                    start_trade_date=20250102,
                    target_trade_date=20260710,
                    trade_days=370,
                )
            ]

        def update_to(self, target_date):
            calls["target_date"] = target_date
            return {
                "market.raw_fields.v1": SimpleNamespace(
                    status="success",
                    last_success_trade_date=20260710,
                    row_count=42,
                )
            }

    monkeypatch.setattr("aicszl.cli.load_settings", lambda path: FakeSettings())
    monkeypatch.setattr("aicszl.cli.RawStore", FakeRawStore)
    monkeypatch.setattr("aicszl.cli.FeatureStore", FakeFeatureStore, raising=False)
    monkeypatch.setattr("aicszl.cli.FeatureRegistry", FakeRegistry, raising=False)
    monkeypatch.setattr("aicszl.cli.FeatureUpdater", FakeUpdater, raising=False)
    monkeypatch.setattr("aicszl.cli.FeatureCalcContext", lambda raw_store: object(), raising=False)
    monkeypatch.setattr("aicszl.cli.register_builtin_features", lambda registry: None, raising=False)
    return calls
