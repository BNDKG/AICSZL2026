from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aicszl.artifact_cache import (
    MODEL_CACHE_SCHEMA_VERSION,
    PREDICTION_CACHE_SCHEMA_VERSION,
)
from aicszl.experiments.runner import ExperimentRunRequest, run_experiment


EXPERIMENT_YAML = """\
name: runner_test
data:
  feature_cutoff: 20240110
train:
  start: 20200101
  end: 20200103
  target: target.ret_5d_rank_pct.v1
predict:
  start: 20240101
  end: 20240104
models:
  - label: 5_features
    feature_group: base_v1
  - label: alternate_model
    feature_group: alternate_v1
model_params:
  n_estimators: 50
  learning_rate: 0.1
  min_data_in_leaf: 1
  verbose: -1
backtest:
  topk: 2
  n_drop: 1
  initial_cash: 1000000
  random_seed: 42
"""

EXECUTABLE_EXPERIMENT_YAML = EXPERIMENT_YAML.replace(
    "name: runner_test",
    "name: runner_executable_test",
).replace(
    "  start: 20200101\n  end: 20200103\n  target: target.ret_5d_rank_pct.v1",
    "  start: 20231220\n  end: 20240101\n  target: target.ret_open_t1_open_t6_rank_pct.v1",
).replace("  n_drop: 1", "  n_drop: 2")

MODEL_CACHE_KEY = "a" * 64
PREDICTION_CACHE_KEY = "b" * 64


def test_experiment_package_exports_runner_interface():
    from aicszl.experiments import (
        ExperimentRunRequest as ExportedRequest,
        run_experiment as exported_runner,
    )

    assert ExportedRequest is ExperimentRunRequest
    assert exported_runner is run_experiment


def test_equity_curve_title_uses_resolved_feature_counts():
    import aicszl.experiments.runner as module
    from aicszl.experiments.config import ResolvedModel

    models = (
        ResolvedModel(
            "current",
            "current_group",
            ("f1", "f2"),
        ),
        ResolvedModel("new", "new_group", ("f1", "f2", "f3")),
    )

    assert module._equity_curve_title(models) == (
        "Random baseline vs 2 features vs 3 features net equity"
    )


def _request(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    experiment_yaml: str = EXPERIMENT_YAML,
) -> ExperimentRunRequest:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(experiment_yaml, encoding="utf-8")
    feature_groups = tmp_path / "features.yaml"
    feature_groups.write_text(
        "feature_groups:\n"
        "  base_v1:\n"
        "    features:\n"
        "      - market.close.v1\n"
        "  alternate_v1:\n"
        "    features:\n"
        "      - market.amount.v1\n",
        encoding="utf-8",
    )
    return ExperimentRunRequest(
        experiment_path=experiment,
        settings_path=tmp_path / "settings.yaml",
        feature_groups_path=feature_groups,
        dry_run=dry_run,
    )


def _settings(tmp_path: Path):
    return SimpleNamespace(
        project=SimpleNamespace(start_date=20200101),
        paths=SimpleNamespace(
            raw_db=tmp_path / "raw.duckdb",
            feature_db=tmp_path / "features.duckdb",
            artifacts_dir=tmp_path / "artifacts",
        ),
    )


class FakeRawStore:
    instances: list["FakeRawStore"] = []

    def __init__(self, *_args, **_kwargs):
        self.closed = False
        self.calendar = [
            20200102,
            20200103,
            20231221,
            20231222,
            20231225,
            20231226,
            20231227,
            20231228,
            20231229,
            20240102,
            20240103,
            20240104,
            20240105,
            20240108,
            20240109,
            20240110,
        ]
        self.instances.append(self)

    def fetch_df(self, sql, params=None):
        params = params or []
        if "FROM trade_cal" in sql and "BETWEEN" in sql:
            start, end = map(int, params[:2])
            return pd.DataFrame(
                {"cal_date": [date for date in self.calendar if start <= date <= end]}
            )
        if "FROM trade_cal" in sql and "cal_date >" in sql:
            start = int(params[0])
            return pd.DataFrame(
                {"cal_date": [date for date in self.calendar if date > start][:5]}
            )
        if "FROM adj_factor" in sql:
            rows = [
                (code, date, 1.0)
                for date in [20240102, 20240103, 20240104, 20240105]
                for code in ["B", "C"]
            ]
            return pd.DataFrame(rows, columns=["ts_code", "trade_date", "adj_factor"])
        raise AssertionError(f"Unexpected raw query: {sql}")

    def get_state(self, table_name):
        return SimpleNamespace(
            table_name=table_name,
            last_success_trade_date=20240110,
            status="success",
        )

    def close(self):
        self.closed = True


class FakeFeatureStore:
    instances: list["FakeFeatureStore"] = []

    def __init__(self, *_args, **_kwargs):
        self.closed = False
        self.target_dates: list[int] = []
        self.instances.append(self)

    def get_state(self, feature_name):
        return SimpleNamespace(
            feature_name=feature_name,
            last_success_trade_date=20240110,
            status="success",
        )

    def fetch_df(self, sql, params=None):
        if "FROM feature_meta" in sql:
            features = list(params or [])
            return pd.DataFrame(
                {
                    "feature_name": features,
                    "code_hash": [f"hash:{feature}" for feature in features],
                }
            )
        raise AssertionError(f"Unexpected feature query: {sql}")

    def feature_value_aggregates(self, features, _start_date, _end_date):
        return {
            feature: {
                "row_count": 4,
                "hash_xor": f"xor:{feature}",
                "hash_sum": f"sum:{feature}",
            }
            for feature in features
        }

    def target_value_aggregate(self, _target, _start_date, _end_date):
        return {"row_count": 4, "hash_xor": "target-xor", "hash_sum": "target-sum"}

    def feature_coverage(self, features, _start_date, _end_date):
        return pd.DataFrame(
            {
                "feature_name": features,
                "row_count": [100] * len(features),
                "finite_count": [100] * len(features),
                "min_date": [20200102] * len(features),
                "max_date": [20240110] * len(features),
                "date_count": [9] * len(features),
            }
        )

    def target_coverage(self, _target, _start_date, _end_date):
        date_count = len(self.target_dates)
        return {"row_count": date_count * 2, "date_count": date_count}

    def append_target_values(self, _target, frame):
        self.target_dates = sorted(frame["trade_date"].unique().tolist())
        return len(frame)

    def close(self):
        self.closed = True


def _install_fast_pipeline(monkeypatch, tmp_path: Path, *, fail_second_train=False):
    import aicszl.experiments.runner as module

    events: list[str] = []
    training_jobs = []
    prediction_requests = []
    backtest_calls = []
    cache_calls = {"train": [], "predict": []}
    FakeRawStore.instances.clear()
    FakeFeatureStore.instances.clear()

    monkeypatch.setattr(module, "load_settings", lambda _path: _settings(tmp_path))
    monkeypatch.setattr(module, "RawStore", FakeRawStore)
    monkeypatch.setattr(module, "FeatureStore", FakeFeatureStore)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.9.7")

    class FakeUpdater:
        def __init__(self, **kwargs):
            assert kwargs["plugin_ids"] == ["market.raw_fields.v1"]

        def update_to(self, cutoff):
            assert cutoff == 20240110
            events.append("features")
            return {}

    monkeypatch.setattr(module, "FeatureUpdater", FakeUpdater)

    def fake_target(_context, target_name, dates):
        events.append("targets")
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "trade_date": date,
                    "target_name": target_name,
                    "value": value,
                }
                for date in dates
                for code, value in [("B", 0.25), ("C", 0.75)]
            ]
        )

    monkeypatch.setattr(module, "calculate_target", fake_target)

    def fake_cached_train(_store, job, cache_root, run_output_dir, legacy_root):
        training_jobs.append(job)
        events.append(f"train:{job.x_group}")
        cache_calls["train"].append(
            {
                "cache_root": Path(cache_root),
                "run_output_dir": Path(run_output_dir),
                "legacy_root": legacy_root,
            }
        )
        if fail_second_train and len(training_jobs) == 2:
            raise RuntimeError("second model failed")
        output = Path(run_output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.pkl"
        meta_path = output / "model.meta.json"
        model_path.write_bytes(job.x_group.encode())
        meta_path.write_text(json.dumps({"job": job.name}), encoding="utf-8")
        return SimpleNamespace(
            artifact_hash=MODEL_CACHE_KEY,
            cache_hit=True,
            cache_key=MODEL_CACHE_KEY,
            cache_source=str(Path(cache_root) / MODEL_CACHE_KEY),
            model_path=model_path,
            meta_path=meta_path,
            train_rows=4,
        )

    monkeypatch.setattr(module, "get_or_train_cached_model", fake_cached_train)

    def fake_cached_predict(
        _store,
        request,
        model_cache_key,
        cache_root,
        run_output_dir,
        legacy_prediction_dir,
    ):
        prediction_requests.append(request)
        label = Path(run_output_dir).name
        events.append(f"predict:{label}")
        cache_calls["predict"].append(
            {
                "model_cache_key": model_cache_key,
                "cache_root": Path(cache_root),
                "run_output_dir": Path(run_output_dir),
                "legacy_prediction_dir": legacy_prediction_dir,
            }
        )
        frame = pd.DataFrame(
            {
                "trade_date": [20240102] * 3 + [20240103] * 3 + [20240104] * 3,
                "ts_code": (["A", "B", "C"] if label == "5_features" else ["B", "C", "D"])
                * 3,
                "score_raw": [0.1, 0.2, 0.3] * 3,
            }
        )
        output = Path(run_output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "prediction.pkl"
        frame.to_pickle(path)
        return SimpleNamespace(
            prediction_id=f"prediction-{label}",
            prediction_path=path,
            rows=len(frame),
            cache_hit=True,
            cache_mode="extend",
            cache_key=PREDICTION_CACHE_KEY,
            cache_source=str(
                Path(cache_root) / model_cache_key / f"{PREDICTION_CACHE_KEY}.pkl"
            ),
            reused_rows=3,
            generated_rows=6,
            generated_range=(20240103, 20240104),
        )

    monkeypatch.setattr(module, "get_or_predict_cached", fake_cached_predict)

    def fake_build_dataset(_raw_store, blend_path, output_dir):
        events.append("provider_dataset")
        blend = pd.read_pickle(blend_path)
        market = blend.rename(columns={"score_raw_blend": "score"}).copy()
        for column, value in {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "vol": 100.0,
            "amount": 1000.0,
            "is_tradable": True,
            "limit_up": 11.0,
            "limit_down": 9.0,
        }.items():
            market[column] = value
        path = Path(output_dir) / "provider_dataset.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        market.to_pickle(path)
        return SimpleNamespace(dataset_path=path, rows=len(market))

    monkeypatch.setattr(module, "build_score_dataset", fake_build_dataset)

    def fake_export(_scores, _factors, target_dir):
        events.append("provider")
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "provider.bin").write_bytes(b"provider")
        return path

    monkeypatch.setattr(module, "export_qlib_provider", fake_export)

    def fake_backtest(provider, scores, **settings):
        label = ["random_baseline", "5_features", "alternate_model"][len(backtest_calls)]
        events.append(f"backtest:{label}")
        backtest_calls.append((provider, scores.copy(), settings))
        index = pd.to_datetime(["2024-01-02", "2024-01-03"])
        report = pd.DataFrame(
            {
                "account": [1_000_000.0, 1_010_000.0],
                "turnover": [0.0, 0.1],
                "total_cost": [0.0, 10.0],
            },
            index=index,
        )
        return {"1day": (report, {date: {} for date in index})}, None

    monkeypatch.setattr(module, "run_qlib_topk_backtest", fake_backtest)

    def fake_plot(reports, output_path, **_kwargs):
        events.append("publish")
        path = Path(output_path)
        path.write_bytes(b"\x89PNG\r\n\x1a\nplot")
        return path

    monkeypatch.setattr(module, "save_equity_curve", fake_plot)
    return events, training_jobs, prediction_requests, backtest_calls, cache_calls


def test_dry_run_resolves_contract_without_writable_or_expensive_stages(
    monkeypatch, tmp_path: Path
):
    import aicszl.experiments.runner as module

    monkeypatch.setattr(module, "load_settings", lambda _path: _settings(tmp_path))
    monkeypatch.setattr(module, "RawStore", FakeRawStore)
    monkeypatch.setattr(
        module,
        "FeatureStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run opened writable feature store")
        ),
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.9.7")

    result = run_experiment(_request(tmp_path, dry_run=True))

    assert result.dry_run is True
    assert result.run_dir is None
    assert result.effective_train_range == (20200102, 20200103)
    assert result.effective_predict_range == (20240102, 20240104)
    assert result.required_plugins == ("market.raw_fields.v1",)
    assert FakeRawStore.instances[-1].closed is True
    assert not (tmp_path / "artifacts").exists()


def test_executable_dry_run_purges_train_dates_and_resolves_next_open_range(
    monkeypatch, tmp_path: Path
):
    import aicszl.experiments.runner as module

    monkeypatch.setattr(module, "load_settings", lambda _path: _settings(tmp_path))
    monkeypatch.setattr(module, "RawStore", FakeRawStore)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.9.7")

    result = run_experiment(
        _request(
            tmp_path,
            dry_run=True,
            experiment_yaml=EXECUTABLE_EXPERIMENT_YAML,
        )
    )

    assert result.effective_train_range == (20231221, 20231221)
    assert result.effective_predict_range == (20240102, 20240104)


def test_executable_dry_run_trims_signal_end_to_last_available_next_open(
    monkeypatch, tmp_path: Path
):
    import aicszl.experiments.runner as module

    monkeypatch.setattr(module, "load_settings", lambda _path: _settings(tmp_path))
    monkeypatch.setattr(module, "RawStore", FakeRawStore)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.9.7")

    experiment_yaml = EXECUTABLE_EXPERIMENT_YAML.replace(
        "  end: 20240104\nmodels:",
        "  end: 20240110\nmodels:",
    )


def _manifest_artifact_paths(manifest, stage: str) -> list[Path]:
    return [
        manifest.run_dir / artifact["path"]
        for artifact in manifest.data["stages"][stage]["artifacts"]
    ]
    result = run_experiment(
        _request(tmp_path, dry_run=True, experiment_yaml=experiment_yaml)
    )

    assert result.effective_predict_range == (20240102, 20240109)


def test_runner_executes_fair_sequential_pipeline_and_completes_manifest(
    monkeypatch, tmp_path: Path
):
    events, jobs, prediction_requests, backtests, cache_calls = _install_fast_pipeline(
        monkeypatch, tmp_path
    )

    result = run_experiment(_request(tmp_path))

    assert result.dry_run is False
    assert result.run_dir is not None
    assert result.equity_curve_path is not None and result.equity_curve_path.is_file()
    assert result.metrics_json_path is not None and result.metrics_json_path.is_file()
    assert result.common_rows == 6
    assert result.common_dates == 3
    assert events == [
        "features",
        "targets",
        "train:base_v1",
        "train:alternate_v1",
        "predict:5_features",
        "predict:alternate_model",
        "provider_dataset",
        "provider",
        "backtest:random_baseline",
        "backtest:5_features",
        "backtest:alternate_model",
        "publish",
    ]
    assert len(jobs) == 2
    assert [call["cache_root"] for call in cache_calls["train"]] == [
        tmp_path / "artifacts" / "cache" / "models",
        tmp_path / "artifacts" / "cache" / "models",
    ]
    assert [call["legacy_root"] for call in cache_calls["train"]] == [None, None]
    assert jobs[0].train_range == jobs[1].train_range == (20200102, 20200103)
    assert jobs[0].target == jobs[1].target == "target.ret_5d_rank_pct.v1"
    assert jobs[0].filters == jobs[1].filters == ["market.amount.v1 > 0"]
    assert jobs[0].model_params == jobs[1].model_params
    assert all(request.start_date == 20240102 for request in prediction_requests)
    assert all(request.end_date == 20240104 for request in prediction_requests)
    assert [call["model_cache_key"] for call in cache_calls["predict"]] == [
        MODEL_CACHE_KEY,
        MODEL_CACHE_KEY,
    ]
    assert [call["cache_root"] for call in cache_calls["predict"]] == [
        tmp_path / "artifacts" / "cache" / "predictions",
        tmp_path / "artifacts" / "cache" / "predictions",
    ]
    assert len(backtests) == 3
    assert backtests[0][0] == backtests[1][0] == backtests[2][0]
    assert backtests[0][2] == backtests[1][2] == backtests[2][2]
    keys = backtests[0][1][["trade_date", "ts_code"]]
    for _, scores, _ in backtests[1:]:
        pd.testing.assert_frame_equal(keys, scores[["trade_date", "ts_code"]])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert list(manifest["stages"])[-1] == "publish"
    train_stage = manifest["stages"]["train:5_features"]
    train_meta = train_stage["metadata"]
    assert train_meta["cache_hit"] is True
    assert train_meta["cache_schema_version"] == MODEL_CACHE_SCHEMA_VERSION
    assert train_meta["cache_key"] == MODEL_CACHE_KEY
    assert Path(train_meta["cache_source"]) == (
        tmp_path / "artifacts" / "cache" / "models" / MODEL_CACHE_KEY
    )
    assert {artifact["path"] for artifact in train_stage["artifacts"]} == {
        "models/5_features/model.pkl",
        "models/5_features/model.meta.json",
    }
    predict_stage = manifest["stages"]["predict:5_features"]
    predict_meta = predict_stage["metadata"]
    assert predict_meta["cache_hit"] is True
    assert predict_meta["cache_schema_version"] == PREDICTION_CACHE_SCHEMA_VERSION
    assert predict_meta["cache_mode"] == "extend"
    assert predict_meta["cache_key"] == PREDICTION_CACHE_KEY
    assert Path(predict_meta["cache_source"]) == (
        tmp_path
        / "artifacts"
        / "cache"
        / "predictions"
        / MODEL_CACHE_KEY
        / f"{PREDICTION_CACHE_KEY}.pkl"
    )
    assert predict_meta["reused_rows"] > 0
    assert predict_meta["generated_rows"] > 0
    assert predict_meta["generated_range"] == [20240103, 20240104]
    assert [artifact["path"] for artifact in predict_stage["artifacts"]] == [
        "predictions/5_features/prediction.pkl"
    ]
    assert FakeRawStore.instances[-1].closed is True
    assert FakeFeatureStore.instances[-1].closed is True


def test_runner_executes_new_target_on_next_open_dates(monkeypatch, tmp_path: Path):
    _, jobs, _, backtests, _ = _install_fast_pipeline(monkeypatch, tmp_path)

    result = run_experiment(
        _request(tmp_path, experiment_yaml=EXECUTABLE_EXPERIMENT_YAML)
    )

    assert [job.train_range for job in jobs] == [
        (20231221, 20231221),
        (20231221, 20231221),
    ]
    assert {job.target for job in jobs} == {
        "target.ret_open_t1_open_t6_rank_pct.v1"
    }
    for _, scores, settings in backtests:
        assert sorted(scores["trade_date"].unique()) == [
            20240103,
            20240104,
            20240105,
        ]
        assert settings["n_drop"] == 2

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "execution_scores" in manifest["stages"]
    timing = manifest["resolved_contract"]["timing"]
    assert timing["execution_delay"] == 1
    assert timing["execution_range"] == [20240103, 20240105]
    metrics = json.loads(result.metrics_json_path.read_text(encoding="utf-8"))
    assert metrics["contract"]["timing"] == timing


def test_runner_records_failure_and_stops_before_predictions(monkeypatch, tmp_path: Path):
    events, _, _, _, _ = _install_fast_pipeline(
        monkeypatch, tmp_path, fail_second_train=True
    )

    with pytest.raises(RuntimeError, match="second model failed"):
        run_experiment(_request(tmp_path))

    assert events == [
        "features",
        "targets",
        "train:base_v1",
        "train:alternate_v1",
    ]
    manifests = list((tmp_path / "artifacts").rglob("run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "train:alternate_model"
    assert "train:5_features" in manifest["stages"]
    assert "train:alternate_model" not in manifest["stages"]


def test_runner_resume_skips_valid_completed_stages(monkeypatch, tmp_path: Path):
    _install_fast_pipeline(monkeypatch, tmp_path, fail_second_train=True)
    with pytest.raises(RuntimeError, match="second model failed"):
        run_experiment(_request(tmp_path))
    run_dir = next((tmp_path / "artifacts").rglob("run_manifest.json")).parent

    events, jobs, _, _, _ = _install_fast_pipeline(monkeypatch, tmp_path)
    original = _request(tmp_path)
    resumed_request = ExperimentRunRequest(
        experiment_path=original.experiment_path,
        settings_path=original.settings_path,
        feature_groups_path=original.feature_groups_path,
        resume_dir=run_dir,
    )
    result = run_experiment(resumed_request)

    assert result.manifest_path is not None
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert [job.x_group for job in jobs] == ["alternate_v1"]
    assert events[:3] == [
        "train:alternate_v1",
        "predict:5_features",
        "predict:alternate_model",
    ]


@pytest.mark.parametrize(
    "historical_cache_key",
    [None, "semantic-64-without-schema"],
    ids=["short-key", "unmarked-64-key"],
)
def test_runner_resume_uses_untrusted_model_locally_without_global_adoption(
    monkeypatch, tmp_path: Path, historical_cache_key: str | None
):
    import aicszl.experiments.runner as module
    from aicszl.artifact_cache import build_training_contract
    from aicszl.predictions import PredictionArtifact
    from aicszl.experiments.manifest import RunManifest, record_stage

    _, initial_jobs, _, _, _ = _install_fast_pipeline(
        monkeypatch, tmp_path, fail_second_train=True
    )
    with pytest.raises(RuntimeError, match="second model failed"):
        run_experiment(_request(tmp_path))
    run_dir = next((tmp_path / "artifacts").rglob("run_manifest.json")).parent
    old_manifest = RunManifest.load(run_dir)
    old_model, old_meta = [
        run_dir / artifact["path"]
        for artifact in old_manifest.data["stages"]["train:5_features"]["artifacts"]
    ]
    if old_model.suffix != ".pkl":
        old_model, old_meta = old_meta, old_model
    job = initial_jobs[0]
    store = FakeFeatureStore.instances[-1]
    contract = build_training_contract(store, job)
    recorded_artifact_hash = (
        contract.cache_key if historical_cache_key is not None else "deadbeef"
    )
    old_meta.write_text(
        json.dumps(
            {
                "artifact_hash": recorded_artifact_hash,
                "job": {
                    "name": job.name,
                    "x_group": job.x_group,
                    "features": job.features,
                    "target": job.target,
                    "train_range": list(job.train_range),
                    "filters": job.filters,
                    "model": job.model,
                    "model_params": job.model_params,
                },
                "feature_code_hashes": contract.feature_code_hashes,
                "train_rows": 4,
                "model_path": str(old_model),
            }
        ),
        encoding="utf-8",
    )
    record_stage(
        old_manifest,
        "train:5_features",
        artifacts=[old_model, old_meta],
        metadata={
            "artifact_hash": recorded_artifact_hash,
            **(
                {"cache_key": contract.cache_key}
                if historical_cache_key is not None
                else {}
            ),
            "train_rows": 4,
            "feature_group": job.x_group,
            "features": job.features,
        },
    )

    events, _, _, _, cache_calls = _install_fast_pipeline(monkeypatch, tmp_path)
    direct_prediction_calls = []

    def direct_predict(_store, prediction_request, output_dir):
        direct_prediction_calls.append(prediction_request)
        events.append("predict:5_features")
        frame = pd.DataFrame(
            {
                "trade_date": [20240102] * 3 + [20240103] * 3 + [20240104] * 3,
                "ts_code": ["A", "B", "C"] * 3,
                "score_raw": [0.1, 0.2, 0.3] * 3,
            }
        )
        path = Path(output_dir) / "legacy-prediction.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
        return PredictionArtifact("legacy-prediction", path, len(frame))

    monkeypatch.setattr(module, "predict_from_artifact", direct_predict, raising=False)
    original = _request(tmp_path)
    result = run_experiment(
        ExperimentRunRequest(
            experiment_path=original.experiment_path,
            settings_path=original.settings_path,
            feature_groups_path=original.feature_groups_path,
            resume_dir=run_dir,
        )
    )

    assert len(direct_prediction_calls) == 1
    assert direct_prediction_calls[0].model_path == old_model
    assert [call["run_output_dir"].name for call in cache_calls["train"]] == [
        "alternate_model"
    ]
    assert [call["run_output_dir"].name for call in cache_calls["predict"]] == [
        "alternate_model"
    ]
    assert not (tmp_path / "artifacts" / "cache" / "models" / contract.cache_key).exists()
    assert not (tmp_path / "artifacts" / "cache" / "predictions").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    legacy_train = manifest["stages"]["train:5_features"]["metadata"]
    assert legacy_train["cache_hit"] is False
    assert legacy_train["cache_mode"] == "unavailable"
    assert legacy_train["cache_key"] is None
    assert legacy_train["cache_unavailable_reason"] == (
        "pre_cache_model_missing_original_data_fingerprint"
    )
    legacy_predict = manifest["stages"]["predict:5_features"]["metadata"]
    assert legacy_predict["cache_hit"] is False
    assert legacy_predict["cache_mode"] == "unavailable"
    assert legacy_predict["cache_key"] is None
    assert legacy_predict["cache_unavailable_reason"] == (
        "pre_cache_model_missing_original_data_fingerprint"
    )
    assert events[:3] == [
        "train:alternate_v1",
        "predict:5_features",
        "predict:alternate_model",
    ]


def test_resume_upgrades_completed_untrusted_prediction_stage_audit(
    monkeypatch, tmp_path: Path
):
    import aicszl.experiments.runner as module
    from aicszl.experiments.manifest import RunManifest, record_stage

    original_build_common_scores = module.build_common_scores
    _install_fast_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "build_common_scores",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("scores failed")),
    )
    with pytest.raises(RuntimeError, match="scores failed"):
        run_experiment(_request(tmp_path))

    run_dir = next((tmp_path / "artifacts").rglob("run_manifest.json")).parent
    manifest = RunManifest.load(run_dir)
    train_stage = "train:5_features"
    train_paths = _manifest_artifact_paths(manifest, train_stage)
    train_metadata = dict(manifest.data["stages"][train_stage]["metadata"])
    train_metadata.pop("cache_schema_version", None)
    record_stage(
        manifest,
        train_stage,
        artifacts=train_paths,
        metadata=train_metadata,
    )

    prediction_stage = "predict:5_features"
    prediction_paths = _manifest_artifact_paths(manifest, prediction_stage)
    original_prediction_bytes = prediction_paths[0].read_bytes()
    old_prediction_metadata = manifest.data["stages"][prediction_stage]["metadata"]
    record_stage(
        manifest,
        prediction_stage,
        artifacts=prediction_paths,
        metadata={
            "prediction_id": old_prediction_metadata["prediction_id"],
            "rows": old_prediction_metadata["rows"],
        },
    )

    monkeypatch.setattr(module, "build_common_scores", original_build_common_scores)
    _, _, _, _, cache_calls = _install_fast_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "predict_from_artifact",
        lambda *_args, **_kwargs: pytest.fail("completed prediction stage was regenerated"),
    )
    original = _request(tmp_path)
    result = run_experiment(
        ExperimentRunRequest(
            experiment_path=original.experiment_path,
            settings_path=original.settings_path,
            feature_groups_path=original.feature_groups_path,
            resume_dir=run_dir,
        )
    )

    assert cache_calls["predict"] == []
    assert prediction_paths[0].read_bytes() == original_prediction_bytes
    completed = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    upgraded = completed["stages"][prediction_stage]["metadata"]
    assert upgraded["cache_hit"] is False
    assert upgraded["cache_mode"] == "unavailable"
    assert upgraded["cache_key"] is None
    assert upgraded["cache_source"] is None
    assert upgraded["cache_unavailable_reason"] == (
        "pre_cache_model_missing_original_data_fingerprint"
    )
