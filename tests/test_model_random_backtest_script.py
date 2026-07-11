from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = Path("scripts/run_model_random_backtest.py").resolve()


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_model_random_backtest_for_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(final_account: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account": [1_000_000.0, final_account],
            "turnover": [0.0, 0.1],
            "total_cost": [0.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def _install_fast_dependencies(monkeypatch, module, tmp_path: Path):
    scores = pd.DataFrame(
        {
            "trade_date": [20240102, 20240102, 20240103, 20240103],
            "ts_code": ["000001.SZ", "000002.SZ"] * 2,
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    dataset_path = tmp_path / "scores.pkl"
    scores.to_pickle(dataset_path)
    provider_path = tmp_path / "provider"
    provider_path.mkdir()

    class FakeRawStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_df(self, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [20240102],
                    "adj_factor": [1.0],
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(
        module,
        "load_settings",
        lambda _path: SimpleNamespace(
            paths=SimpleNamespace(raw_db=tmp_path / "raw.duckdb"),
            project=SimpleNamespace(start_date=20200101),
        ),
    )
    monkeypatch.setattr(module, "RawStore", FakeRawStore)
    monkeypatch.setattr(
        module,
        "build_score_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(dataset_path=dataset_path),
    )
    monkeypatch.setattr(
        module, "export_qlib_provider", lambda *_args, **_kwargs: provider_path
    )

    calls = []

    def fake_backtest(provider, run_scores, **settings):
        calls.append((provider, run_scores.copy(), settings))
        final_account = 1_100_000.0 if len(calls) == 1 else 900_000.0
        report = _report(final_account)
        positions = {date: {} for date in report.index}
        return {"1day": (report, positions)}, None

    monkeypatch.setattr(module, "run_qlib_topk_backtest", fake_backtest)
    return scores, provider_path, calls


def test_runner_publishes_complete_artifact_set_and_prints_paths(
    monkeypatch, tmp_path: Path, capsys
):
    module = _load_script()
    original_scores, provider_path, calls = _install_fast_dependencies(
        monkeypatch, module, tmp_path
    )
    output_dir = tmp_path / "output"

    result = module.main(
        [
            "--blend-path",
            str(tmp_path / "blend.pkl"),
            "--output-dir",
            str(output_dir),
            "--random-seed",
            "42",
        ]
    )

    assert result == 0
    expected_paths = [
        output_dir / "model" / "report.pkl",
        output_dir / "model" / "positions.pkl",
        output_dir / "random_seed_42" / "report.pkl",
        output_dir / "random_seed_42" / "positions.pkl",
        output_dir / "metrics.json",
        output_dir / "equity_curve.png",
        output_dir / "completion.json",
    ]
    assert all(path.is_file() for path in expected_paths)
    assert json.loads((output_dir / "completion.json").read_text("utf-8"))[
        "status"
    ] == "complete"
    stdout = capsys.readouterr().out
    for path in expected_paths[:-1]:
        assert str(path.resolve()) in stdout

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == provider_path
    assert calls[0][2] == calls[1][2]
    pd.testing.assert_frame_equal(calls[0][1], original_scores)
    assert calls[1][1]["score"].tolist() == pytest.approx(
        [0.7739560486, 0.4388784398, 0.8585979199, 0.6973680291]
    )


def test_runner_does_not_publish_partial_results_when_plotting_fails(
    monkeypatch, tmp_path: Path
):
    module = _load_script()
    _install_fast_dependencies(monkeypatch, module, tmp_path)
    output_dir = tmp_path / "output"
    old_report_path = output_dir / "model" / "report.pkl"
    old_report_path.parent.mkdir(parents=True)
    pd.DataFrame({"sentinel": [1]}).to_pickle(old_report_path)
    output_dir.joinpath("completion.json").write_text(
        '{"status":"complete"}', encoding="utf-8"
    )
    monkeypatch.setattr(
        module,
        "save_equity_curve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        module.main(
            [
                "--blend-path",
                str(tmp_path / "blend.pkl"),
                "--output-dir",
                str(output_dir),
            ]
        )

    pd.testing.assert_frame_equal(
        pd.read_pickle(old_report_path), pd.DataFrame({"sentinel": [1]})
    )
    assert not output_dir.joinpath("completion.json").exists()
