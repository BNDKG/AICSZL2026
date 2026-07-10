from pathlib import Path

from aicszl.backtests.base import BacktestAdapter, BacktestRunArtifact, BacktestRunSettings
from aicszl.backtests.dataset import BacktestDatasetArtifact
from aicszl.backtests.qlib_adapter import QlibBacktestAdapter


def test_adapter_boundary_is_runtime_checkable_and_engine_neutral(tmp_path: Path):
    class FakeAdapter:
        def run(
            self,
            dataset: BacktestDatasetArtifact,
            settings: BacktestRunSettings,
        ) -> BacktestRunArtifact:
            return BacktestRunArtifact(
                engine="fake",
                report_path=tmp_path / "report.pkl",
                positions_path=tmp_path / "positions.pkl",
            )

    dataset = BacktestDatasetArtifact("scores", tmp_path / "scores.pkl", rows=2)
    settings = BacktestRunSettings(topk=1, n_drop=1, initial_cash=100_000)

    assert isinstance(FakeAdapter(), BacktestAdapter)
    assert settings.benchmark is None
    assert FakeAdapter().run(dataset, settings).engine == "fake"


def test_qlib_adapter_uses_the_engine_neutral_boundary():
    assert isinstance(QlibBacktestAdapter(raw_store=object()), BacktestAdapter)


def test_backtest_settings_rejects_invalid_topk_and_drop():
    for kwargs in [{"topk": 0, "n_drop": 1}, {"topk": 1, "n_drop": 0}, {"topk": 1, "n_drop": 2}]:
        try:
            BacktestRunSettings(initial_cash=100_000, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid settings to fail: {kwargs}")
