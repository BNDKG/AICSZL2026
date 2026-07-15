from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/run_experiment.py").resolve()


def _load_script():
    spec = importlib.util.spec_from_file_location("run_experiment_for_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_builds_request_and_prints_dry_run_contract(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    captured = []

    def fake_run(request):
        captured.append(request)
        request.on_progress("preflight complete")
        return module.ExperimentRunResult(
            dry_run=True,
            run_dir=None,
            manifest_path=None,
            effective_train_range=(20200102, 20231229),
            effective_predict_range=(20240102, 20260701),
            required_plugins=("market.raw_fields.v1", "market.price_volume_pack.v1"),
        )

    monkeypatch.setattr(module, "run_experiment", fake_run)
    experiment = tmp_path / "experiment.yaml"

    result = module.main(
        [
            "--experiment",
            str(experiment),
            "--config",
            "custom-settings.yaml",
            "--feature-groups",
            "custom-features.yaml",
            "--dry-run",
        ]
    )

    assert result == 0
    request = captured[0]
    assert request.experiment_path == experiment
    assert request.settings_path == Path("custom-settings.yaml")
    assert request.feature_groups_path == Path("custom-features.yaml")
    assert request.dry_run is True
    stdout = capsys.readouterr().out
    assert "preflight complete" in stdout
    assert "train=20200102-20231229" in stdout
    assert "predict=20240102-20260701" in stdout
    assert "market.price_volume_pack.v1" in stdout


def test_script_prints_completed_absolute_artifacts(monkeypatch, tmp_path: Path, capsys):
    module = _load_script()
    run_dir = (tmp_path / "run").resolve()
    curve = run_dir / "equity_curve.png"
    metrics = run_dir / "metrics.json"
    manifest = run_dir / "run_manifest.json"
    monkeypatch.setattr(
        module,
        "run_experiment",
        lambda _request: module.ExperimentRunResult(
            dry_run=False,
            run_dir=run_dir,
            manifest_path=manifest,
            effective_train_range=(20200102, 20231229),
            effective_predict_range=(20240102, 20260701),
            required_plugins=("market.raw_fields.v1",),
            equity_curve_path=curve,
            metrics_json_path=metrics,
            metrics_csv_path=run_dir / "metrics.csv",
            common_rows=123,
            common_dates=456,
        ),
    )

    assert module.main(["--experiment", "experiment.yaml"]) == 0
    stdout = capsys.readouterr().out
    for path in [run_dir, curve, metrics, manifest]:
        assert str(path) in stdout
    assert "common_rows=123 common_dates=456" in stdout


def test_script_returns_nonzero_and_does_not_print_success_on_error(
    monkeypatch, capsys
):
    module = _load_script()
    monkeypatch.setattr(
        module,
        "run_experiment",
        lambda _request: (_ for _ in ()).throw(RuntimeError("feature failed")),
    )

    result = module.main(["--experiment", "experiment.yaml"])

    assert result == 1
    output = capsys.readouterr()
    assert "feature failed" in output.err
    assert "run complete" not in output.out


def test_script_passes_resume_directory(monkeypatch, tmp_path: Path):
    module = _load_script()
    captured = []
    monkeypatch.setattr(
        module,
        "run_experiment",
        lambda request: captured.append(request)
        or module.ExperimentRunResult(
            dry_run=True,
            run_dir=None,
            manifest_path=None,
            effective_train_range=(1, 2),
            effective_predict_range=(3, 4),
            required_plugins=("plugin",),
        ),
    )
    run_dir = tmp_path / "run"

    assert (
        module.main(
            [
                "--experiment",
                "experiment.yaml",
                "--resume",
                str(run_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured[0].resume_dir == run_dir
