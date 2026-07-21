import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from aicszl.artifact_cache import (
    build_training_contract,
    get_or_predict_cached,
    get_or_train_cached_model,
    prediction_data_fingerprint,
    semantic_training_payload,
)
from aicszl.features.store import FeatureMeta, FeatureStore
from aicszl.models.training import ModelArtifact, TrainingJob
from aicszl.predictions.runner import PredictionArtifact, PredictionRequest


MODEL_CACHE_KEY = "a" * 64
LEGACY_ARTIFACT_HASH = "legacy-model-artifact-hash"


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    feature_store = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    _seed_values(feature_store)
    _seed_meta(feature_store)
    return feature_store


@pytest.fixture
def job() -> TrainingJob:
    return TrainingJob(
        name="baseline_run",
        x_group="base_v1",
        features=["market.close.v1", "market.amount.v1"],
        target="target.ret_5d_rank_pct.v1",
        train_range=(20200102, 20200103),
        filters=["market.amount.v1 > 500"],
        model_params={"n_estimators": 3, "learning_rate": 0.1},
    )


def test_semantic_training_payload_excludes_aliases_and_normalizes_inputs(job: TrainingJob):
    payload = semantic_training_payload(job)

    assert "name" not in payload
    assert "x_group" not in payload
    assert payload == {
        "features": ["market.close.v1", "market.amount.v1"],
        "target": "target.ret_5d_rank_pct.v1",
        "train_range": {
            "start_date": 20200102,
            "end_date": 20200103,
            "inclusive": True,
        },
        "filters": ["market.amount.v1 > 500"],
        "model": "lgbm_regressor_v1",
        "model_params": {"learning_rate": 0.1, "n_estimators": 3},
    }


def test_training_contract_ignores_name_x_group_and_parameter_mapping_order(
    store: FeatureStore,
    job: TrainingJob,
):
    first = build_training_contract(store, job)
    alias = build_training_contract(
        store,
        replace(
            job,
            name="another_run",
            x_group="alias",
            model_params={"learning_rate": 0.1, "n_estimators": 3},
        ),
    )

    assert first.cache_key == alias.cache_key
    assert first.payload == alias.payload


def test_training_contract_payload_is_the_complete_canonical_cache_input(
    store: FeatureStore,
    job: TrainingJob,
):
    contract = build_training_contract(store, job)

    assert contract.payload == {
        "training": semantic_training_payload(job),
        "feature_code_hashes": contract.feature_code_hashes,
        "data_fingerprint": contract.data_fingerprint,
    }


@pytest.mark.parametrize(
    "changed_job",
    [
        lambda value: replace(value, features=list(reversed(value.features))),
        lambda value: replace(value, target="target.ret_10d_rank_pct.v1"),
        lambda value: replace(value, train_range=(20200102, 20200104)),
        lambda value: replace(value, filters=["market.amount.v1 > 1000"]),
        lambda value: replace(value, model_params={"n_estimators": 4, "learning_rate": 0.1}),
    ],
    ids=["ordered-features", "target", "train-range", "filter", "model-parameters"],
)
def test_training_contract_key_changes_with_each_semantic_job_input(
    store: FeatureStore,
    job: TrainingJob,
    changed_job,
):
    first = build_training_contract(store, job)

    changed = build_training_contract(store, changed_job(job))

    assert changed.cache_key != first.cache_key


def test_training_contract_key_changes_with_feature_code_hash(store: FeatureStore, job: TrainingJob):
    first = build_training_contract(store, job)
    store.conn.execute(
        "UPDATE feature_meta SET code_hash = ? WHERE feature_name = ?",
        ["close-hash-v2", "market.close.v1"],
    )

    changed = build_training_contract(store, job)

    assert changed.cache_key != first.cache_key


def test_training_contract_key_changes_with_selected_feature_value(store: FeatureStore, job: TrainingJob):
    first = build_training_contract(store, job)
    store.conn.execute(
        'UPDATE fv_market_raw_fields_v1 SET "market.close.v1" = 99.0 '
        "WHERE ts_code = '000001.SZ' AND trade_date = 20200102"
    )

    changed = build_training_contract(store, job)

    assert changed.cache_key != first.cache_key


def test_training_contract_key_changes_with_selected_target_value(store: FeatureStore, job: TrainingJob):
    first = build_training_contract(store, job)
    store.conn.execute(
        "UPDATE tv_target_ret_5d_rank_pct_v1 SET value = 0.9 "
        "WHERE ts_code = '000001.SZ' AND trade_date = 20200102"
    )

    changed = build_training_contract(store, job)

    assert changed.cache_key != first.cache_key


def test_prediction_fingerprint_has_row_counts_and_string_hash_aggregates(
    store: FeatureStore,
    job: TrainingJob,
):
    fingerprint = prediction_data_fingerprint(
        store,
        job.features,
        job.target,
        job.train_range[0],
        job.train_range[1],
    )

    assert fingerprint["range"] == {
        "start_date": 20200102,
        "end_date": 20200103,
        "inclusive": True,
    }
    assert [row["feature_name"] for row in fingerprint["features"]] == job.features
    assert [row["row_count"] for row in fingerprint["features"]] == [4, 4]
    assert all(isinstance(row["hash_xor"], str) for row in fingerprint["features"])
    assert all(isinstance(row["hash_sum"], str) for row in fingerprint["features"])
    assert fingerprint["target"]["target_name"] == job.target
    assert fingerprint["target"]["row_count"] == 4
    assert isinstance(fingerprint["target"]["hash_xor"], str)
    assert isinstance(fingerprint["target"]["hash_sum"], str)


def test_prediction_fingerprint_ignores_unselected_and_out_of_range_values(
    store: FeatureStore,
    job: TrainingJob,
):
    first = prediction_data_fingerprint(
        store,
        job.features,
        job.target,
        job.train_range[0],
        job.train_range[1],
    )
    store.append_plugin_values(
        "market.unselected.v1",
        ["market.unselected.v1"],
        pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": 20200102, "market.unselected.v1": 123.0}]
        ),
    )
    store.append_plugin_values(
        "market.raw_fields.v1",
        ["market.close.v1", "market.amount.v1"],
        pd.DataFrame(
            [{
                "ts_code": "000001.SZ",
                "trade_date": 20200104,
                "market.close.v1": 456.0,
                "market.amount.v1": 1000.0,
            }]
        ),
    )
    store.append_target_values(
        job.target,
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": 20200104, "value": 0.8}]),
    )

    changed = prediction_data_fingerprint(
        store,
        job.features,
        job.target,
        job.train_range[0],
        job.train_range[1],
    )

    assert changed == first


def test_cached_model_miss_trains_once_and_alias_hit_skips_trainer(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    calls: list[str] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        calls.append(training_job.name)
        return _write_synthetic_model(training_store, training_job, Path(output_dir), b"trained-model")

    first = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-1",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )
    second = get_or_train_cached_model(
        store,
        replace(job, name="aliased_run", x_group="alias"),
        tmp_path / "cache",
        tmp_path / "run-2",
        tmp_path / "legacy",
        trainer=lambda *_args, **_kwargs: pytest.fail("cache hit called trainer"),
    )

    assert calls == [job.name]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert first.model_path.read_bytes() == b"trained-model"
    assert second.model_path.read_bytes() == b"trained-model"
    assert first.meta_path.is_file()
    assert second.meta_path.is_file()
    cache_metadata = json.loads(
        (tmp_path / "cache" / first.cache_key / "model.meta.json").read_text(
            encoding="utf-8"
        )
    )
    alias_metadata = json.loads(second.meta_path.read_text(encoding="utf-8"))
    assert cache_metadata["job"]["name"] == job.name
    assert cache_metadata["job"]["x_group"] == job.x_group
    assert alias_metadata["job"]["name"] == "aliased_run"
    assert alias_metadata["job"]["x_group"] == "alias"
    assert alias_metadata["model_path"] == str(second.model_path.resolve())
    assert alias_metadata["cache_provenance"]["source_job"] == cache_metadata["job"]


def test_concurrent_same_key_requests_train_once_and_receive_one_valid_artifact(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    database_path = store.db_path
    store.conn.close()
    stores = [
        FeatureStore(database_path, start_date=20200101, read_only=True),
        FeatureStore(database_path, start_date=20200101, read_only=True),
    ]
    jobs = [job, replace(job, name="concurrent_alias", x_group="alias")]
    start = threading.Barrier(2)
    call_lock = threading.Lock()
    calls: list[int] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        with call_lock:
            calls.append(len(calls) + 1)
            invocation = calls[-1]
        time.sleep(0.2)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            f"trained-{invocation}".encode(),
        )
    def request(index: int):
        start.wait()
        return get_or_train_cached_model(
            stores[index],
            jobs[index],
            tmp_path / "cache",
            tmp_path / f"run-{index}",
            tmp_path / "legacy",
            trainer=recording_trainer,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(request, range(2)))
    finally:
        for read_store in stores:
            read_store.conn.close()

    assert calls == [1]
    assert {result.cache_key for result in results} == {results[0].cache_key}
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert results[0].model_path.read_bytes() == results[1].model_path.read_bytes() == b"trained-1"
    metadata = [json.loads(result.meta_path.read_text(encoding="utf-8")) for result in results]
    assert {item["job"]["name"] for item in metadata} == {
        "baseline_run",
        "concurrent_alias",
    }
    assert {item["job"]["x_group"] for item in metadata} == {"base_v1", "alias"}
    assert {item["model_path"] for item in metadata} == {
        str(result.model_path.resolve()) for result in results
    }
    assert metadata[0]["cache_provenance"]["source_job"] == (
        metadata[1]["cache_provenance"]["source_job"]
    )


def test_corrupt_cached_model_is_rebuilt(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    calls: list[bytes] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        payload = f"trained-{len(calls) + 1}".encode()
        calls.append(payload)
        return _write_synthetic_model(training_store, training_job, Path(output_dir), payload)

    first = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-1",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )
    cache_model = tmp_path / "cache" / first.cache_key / "model.pkl"
    cache_model.write_bytes(b"corrupt")

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-2",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )

    assert calls == [b"trained-1", b"trained-2"]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"trained-2"


@pytest.mark.parametrize(
    "manifest_change",
    [
        lambda manifest: manifest.update(source="unknown"),
        lambda manifest: manifest.update(source="legacy"),
    ],
    ids=["unknown-source", "legacy-without-provenance"],
)
def test_cache_manifest_with_invalid_provenance_is_rebuilt(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
    manifest_change,
):
    first = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-1",
        tmp_path / "legacy",
        trainer=lambda training_store, training_job, output_dir: _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"first-model",
        ),
    )
    manifest_path = tmp_path / "cache" / first.cache_key / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_change(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    calls: list[str] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"rebuilt-model",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-2",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"rebuilt-model"


def test_existing_legacy_model_cache_without_original_data_fingerprint_is_rebuilt(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    first = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-1",
        None,
        trainer=lambda training_store, training_job, output_dir: _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"unsafe-legacy-cache",
        ),
    )
    manifest_path = tmp_path / "cache" / first.cache_key / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "legacy"
    manifest["legacy_source"] = {
        "model_path": str((tmp_path / "old-model.pkl").resolve()),
        "meta_path": str((tmp_path / "old-model.meta.json").resolve()),
    }
    manifest.pop("original_data_fingerprint", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[str] = []

    def recording_trainer(training_store, training_job, output_dir):
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"safe-rebuild",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run-2",
        None,
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"safe-rebuild"


def test_matching_legacy_model_without_original_data_fingerprint_is_retrained(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    legacy_model, legacy_meta = _write_legacy_model(store, job, tmp_path / "legacy", b"legacy-model")

    calls: list[str] = []

    def recording_trainer(training_store, training_job, output_dir):
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"safe-trained-model",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )

    assert legacy_model.is_file() and legacy_meta.is_file()
    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"safe-trained-model"


@pytest.mark.parametrize("conflict", ["model-content", "train-rows"])
def test_conflicting_legacy_models_are_rejected_and_trained_once(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
    conflict: str,
):
    legacy_root = tmp_path / "legacy"
    _write_synthetic_model(
        store,
        replace(job, name="old_a", x_group="old_group_a"),
        legacy_root / "run-a" / "models" / "old_a",
        b"legacy-a",
    )
    second = _write_synthetic_model(
        store,
        replace(job, name="old_b", x_group="old_group_b"),
        legacy_root / "run-b" / "models" / "old_b",
        b"legacy-b" if conflict == "model-content" else b"legacy-a",
    )
    if conflict == "train-rows":
        metadata = json.loads(second.meta_path.read_text(encoding="utf-8"))
        metadata["train_rows"] = 5
        second.meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls: list[str] = []

    def recording_trainer(training_store, training_job, output_dir):
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"fresh-model",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        legacy_root,
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"fresh-model"


def test_narrow_legacy_root_excludes_recorded_canonical_run_model(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    run_dir = tmp_path / "managed-run"
    canonical = _write_synthetic_model(
        store,
        job,
        run_dir / "models" / "current_label",
        b"canonical-run-model",
    )
    _write_model_stage_manifest(run_dir, canonical, cache_key=canonical.artifact_hash)
    calls: list[str] = []

    def recording_trainer(training_store, training_job, output_dir):
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"fresh-model",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        canonical.meta_path.parent,
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"fresh-model"


def test_narrow_legacy_root_excludes_unrecorded_managed_run_model(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    run_dir = tmp_path / "managed-run"
    unrecorded = _write_synthetic_model(
        store,
        job,
        run_dir / "models" / "current_label",
        b"in-progress-model",
    )
    run_dir.joinpath("run_manifest.json").write_text(
        json.dumps({"stages": {}}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def recording_trainer(training_store, training_job, output_dir):
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"fresh-model",
        )

    rebuilt = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        unrecorded.meta_path.parent,
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert rebuilt.cache_hit is False
    assert rebuilt.model_path.read_bytes() == b"fresh-model"


def test_legacy_model_with_mismatched_feature_hashes_is_rejected(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    _, legacy_meta = _write_legacy_model(store, job, tmp_path / "legacy", b"stale-legacy")
    metadata = json.loads(legacy_meta.read_text(encoding="utf-8"))
    metadata["feature_code_hashes"]["market.close.v1"] = "stale-close-hash"
    legacy_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    calls: list[str] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        calls.append(training_job.name)
        return _write_synthetic_model(training_store, training_job, Path(output_dir), b"fresh-model")

    result = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert result.cache_hit is False
    assert result.model_path.read_bytes() == b"fresh-model"


def test_legacy_model_with_noninclusive_training_contract_is_rejected(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
):
    _, legacy_meta = _write_legacy_model(store, job, tmp_path / "legacy", b"wrong-range")
    metadata = json.loads(legacy_meta.read_text(encoding="utf-8"))
    metadata["job"]["train_range"] = {
        "start_date": job.train_range[0],
        "end_date": job.train_range[1],
        "inclusive": False,
    }
    legacy_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        tmp_path / "legacy",
        trainer=lambda training_store, training_job, output_dir: _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"fresh-model",
        ),
    )

    assert result.cache_hit is False
    assert result.model_path.read_bytes() == b"fresh-model"


@pytest.mark.parametrize("invalid_train_rows", ["invalid", -1])
def test_legacy_model_with_invalid_train_rows_is_rejected(
    store: FeatureStore,
    job: TrainingJob,
    tmp_path: Path,
    invalid_train_rows: object,
):
    _, legacy_meta = _write_legacy_model(store, job, tmp_path / "legacy", b"invalid-legacy")
    metadata = json.loads(legacy_meta.read_text(encoding="utf-8"))
    metadata["train_rows"] = invalid_train_rows
    legacy_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    calls: list[str] = []

    def recording_trainer(
        training_store: FeatureStore,
        training_job: TrainingJob,
        output_dir: str | Path,
    ) -> ModelArtifact:
        calls.append(training_job.name)
        return _write_synthetic_model(
            training_store,
            training_job,
            Path(output_dir),
            b"fresh-model",
        )

    result = get_or_train_cached_model(
        store,
        job,
        tmp_path / "cache",
        tmp_path / "run",
        tmp_path / "legacy",
        trainer=recording_trainer,
    )

    assert calls == [job.name]
    assert result.cache_hit is False
    assert result.model_path.read_bytes() == b"fresh-model"


def test_prediction_cache_miss_then_exact_hit_skips_predictor(
    store: FeatureStore,
    tmp_path: Path,
):
    request = _prediction_request(tmp_path, 20200102, 20200103)
    predictor = _recording_predictor()

    missed = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-1",
        None,
        predictor=predictor,
    )
    exact = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-2",
        None,
        predictor=lambda *_args, **_kwargs: pytest.fail("exact hit called predictor"),
    )

    assert predictor.ranges == [(20200102, 20200103)]
    assert missed.cache_hit is False
    assert missed.cache_mode == "miss"
    assert missed.reused_rows == 0
    assert missed.generated_rows == 4
    assert exact.cache_hit is True
    assert exact.cache_mode == "exact"
    assert exact.reused_rows == 4
    assert exact.generated_rows == 0
    pd.testing.assert_frame_equal(
        pd.read_pickle(exact.prediction_path),
        pd.read_pickle(missed.prediction_path),
    )


def test_prediction_cache_slices_valid_superset_without_predicting(
    store: FeatureStore,
    tmp_path: Path,
):
    full_request = _prediction_request(tmp_path, 20200102, 20200106)
    cached = get_or_predict_cached(
        store,
        full_request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "full-run",
        None,
        predictor=_recording_predictor(),
    )

    sliced = get_or_predict_cached(
        store,
        _prediction_request(tmp_path, 20200103, 20200103),
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "slice-run",
        None,
        predictor=lambda *_args, **_kwargs: pytest.fail("slice hit called predictor"),
    )

    expected = pd.read_pickle(cached.prediction_path).query("trade_date == 20200103").reset_index(
        drop=True
    )
    assert sliced.cache_hit is True
    assert sliced.cache_mode == "slice"
    assert sliced.reused_rows == len(expected)
    assert sliced.generated_rows == 0
    pd.testing.assert_frame_equal(pd.read_pickle(sliced.prediction_path), expected)


def test_prediction_cache_extends_only_from_first_available_date_and_preserves_prefix(
    store: FeatureStore,
    tmp_path: Path,
):
    prefix_request = _prediction_request(tmp_path, 20200102, 20200103)
    prefix = get_or_predict_cached(
        store,
        prefix_request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "prefix-run",
        None,
        predictor=_recording_predictor(),
    )
    prefix_frame = pd.read_pickle(prefix.prediction_path)
    predictor = _recording_predictor()

    extended = get_or_predict_cached(
        store,
        _prediction_request(tmp_path, 20200102, 20200106),
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "extended-run",
        None,
        predictor=predictor,
    )
    extended_frame = pd.read_pickle(extended.prediction_path)

    assert predictor.ranges == [(20200106, 20200106)]
    assert extended.cache_hit is True
    assert extended.cache_mode == "extend"
    assert extended.reused_rows == len(prefix_frame)
    assert extended.generated_rows == 2
    assert extended.generated_range == (20200106, 20200106)
    pd.testing.assert_frame_equal(
        extended_frame[extended_frame.trade_date <= 20200103].reset_index(drop=True),
        prefix_frame.reset_index(drop=True),
    )


def test_changed_prefix_data_forces_full_prediction_instead_of_extension(
    store: FeatureStore,
    tmp_path: Path,
):
    get_or_predict_cached(
        store,
        _prediction_request(tmp_path, 20200102, 20200103),
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "prefix-run",
        None,
        predictor=_recording_predictor(),
    )
    store.conn.execute(
        'UPDATE fv_market_raw_fields_v1 SET "market.close.v1" = 99.0 '
        "WHERE ts_code = '000001.SZ' AND trade_date = 20200102"
    )
    predictor = _recording_predictor()

    result = get_or_predict_cached(
        store,
        _prediction_request(tmp_path, 20200102, 20200106),
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "changed-run",
        None,
        predictor=predictor,
    )

    assert predictor.ranges == [(20200102, 20200106)]
    assert result.cache_hit is False
    assert result.cache_mode == "miss"
    assert result.reused_rows == 0
    assert result.generated_rows == 6


def test_corrupt_prediction_cache_falls_back_to_full_prediction(
    store: FeatureStore,
    tmp_path: Path,
):
    request = _prediction_request(tmp_path, 20200102, 20200103)
    first = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "first-run",
        None,
        predictor=_recording_predictor(),
    )
    Path(first.cache_source).write_bytes(b"corrupt-pickle")
    predictor = _recording_predictor()

    rebuilt = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "rebuilt-run",
        None,
        predictor=predictor,
    )

    assert predictor.ranges == [(20200102, 20200103)]
    assert rebuilt.cache_hit is False
    assert rebuilt.cache_mode == "miss"
    assert len(pd.read_pickle(rebuilt.prediction_path)) == 4


def test_legacy_prediction_without_original_data_fingerprint_is_regenerated(
    store: FeatureStore,
    tmp_path: Path,
):
    request = _prediction_request(tmp_path, 20200102, 20200103)
    legacy_dir = tmp_path / "legacy-run" / "predictions" / "current-model"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / "legacy.pkl"
    _prediction_frame(20200102, 20200103).to_pickle(legacy_path)

    predictor = _recording_predictor()
    regenerated = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "adopted-run",
        legacy_dir,
        predictor=predictor,
    )

    assert legacy_path.is_file()
    assert predictor.ranges == [(20200102, 20200103)]
    assert regenerated.cache_hit is False
    assert regenerated.cache_mode == "miss"


def test_existing_legacy_prediction_cache_without_original_fingerprint_is_rebuilt(
    store: FeatureStore,
    tmp_path: Path,
):
    request = _prediction_request(tmp_path, 20200102, 20200103)
    first = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-1",
        None,
        predictor=_recording_predictor(),
    )
    manifest_path = Path(first.cache_source).with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "legacy"
    manifest["legacy_source"] = str((tmp_path / "legacy.pkl").resolve())
    manifest.pop("original_data_fingerprint", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictor = _recording_predictor()

    rebuilt = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-2",
        None,
        predictor=predictor,
    )

    assert predictor.ranges == [(20200102, 20200103)]
    assert rebuilt.cache_hit is False
    assert rebuilt.cache_mode == "miss"


def test_prediction_cache_hit_rewrites_run_aliases_without_mutating_cache(
    store: FeatureStore,
    tmp_path: Path,
):
    first_request = _prediction_request(
        tmp_path,
        20200102,
        20200103,
        job_name="source_job",
        x_group="source_group",
    )
    first = get_or_predict_cached(
        store,
        first_request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-1",
        None,
        predictor=_recording_predictor(),
    )
    cache_before = pd.read_pickle(first.cache_source)
    alias_request = _prediction_request(
        tmp_path,
        20200102,
        20200103,
        job_name="alias_job",
        x_group="alias_group",
    )

    alias = get_or_predict_cached(
        store,
        alias_request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "run-2",
        None,
        predictor=lambda *_args, **_kwargs: pytest.fail("alias cache hit predicted"),
    )

    alias_frame = pd.read_pickle(alias.prediction_path)
    cache_after = pd.read_pickle(alias.cache_source)
    assert alias.cache_hit is True
    assert set(alias_frame["train_job_id"]) == {"alias_job"}
    assert set(alias_frame["x_group"]) == {"alias_group"}
    assert set(alias_frame["cache_source_train_job_id"]) == {"source_job"}
    assert set(alias_frame["cache_source_x_group"]) == {"source_group"}
    pd.testing.assert_series_equal(alias_frame["score_raw"], cache_before["score_raw"])
    pd.testing.assert_frame_equal(cache_after, cache_before)


def test_legacy_prediction_with_unrecorded_model_hash_is_not_adopted(
    store: FeatureStore,
    tmp_path: Path,
):
    request = _prediction_request(
        tmp_path,
        20200102,
        20200103,
        legacy_artifact_hash=LEGACY_ARTIFACT_HASH,
    )
    legacy_dir = tmp_path / "legacy-run" / "predictions" / "current-model"
    legacy_dir.mkdir(parents=True)
    _prediction_frame(20200102, 20200103).assign(
        model_artifact_id="unrecorded-old-hash"
    ).to_pickle(legacy_dir / "legacy.pkl")
    predictor = _recording_predictor()

    result = get_or_predict_cached(
        store,
        request,
        MODEL_CACHE_KEY,
        tmp_path / "prediction-cache",
        tmp_path / "prediction-run",
        legacy_dir,
        predictor=predictor,
    )

    assert result.cache_mode == "miss"
    assert predictor.ranges == [(20200102, 20200103)]
    assert set(pd.read_pickle(result.prediction_path)["model_artifact_id"]) == {MODEL_CACHE_KEY}


def test_prediction_with_duplicate_keys_is_rejected_without_publishing_cache(
    store: FeatureStore,
    tmp_path: Path,
):
    predictor = _recording_predictor(frame_mutator=lambda frame: pd.concat([frame, frame.iloc[[0]]]))

    with pytest.raises(ValueError, match="duplicate"):
        get_or_predict_cached(
            store,
            _prediction_request(tmp_path, 20200102, 20200103),
            MODEL_CACHE_KEY,
            tmp_path / "prediction-cache",
            tmp_path / "invalid-run",
            None,
            predictor=predictor,
        )

    assert not list((tmp_path / "prediction-cache").glob("*.json"))


@pytest.mark.parametrize(
    "invalid_key",
    [".", "..", "../escaped-cache", "nested/cache", r"nested\cache", "a" * 63, "z" * 64],
    ids=["dot", "dot-dot", "parent", "slash", "backslash", "short", "non-hex"],
)
def test_prediction_cache_rejects_unsafe_or_noncanonical_model_key_before_writing(
    store: FeatureStore,
    tmp_path: Path,
    invalid_key: str,
):
    predictor = _recording_predictor()

    with pytest.raises(ValueError, match="model_cache_key"):
        get_or_predict_cached(
            store,
            _prediction_request(tmp_path, 20200102, 20200103),
            invalid_key,
            tmp_path / "prediction-cache",
            tmp_path / "invalid-run",
            None,
            predictor=predictor,
        )

    assert predictor.ranges == []
    assert not (tmp_path / "prediction-cache").exists()
    assert not (tmp_path / "escaped-cache").exists()


@pytest.mark.parametrize(
    "unsafe_job_name",
    [".", "..", "../escaped", "nested/name", r"nested\name"],
    ids=["dot", "dot-dot", "parent", "slash", "backslash"],
)
def test_prediction_cache_rejects_unsafe_metadata_job_name_before_writing(
    store: FeatureStore,
    tmp_path: Path,
    unsafe_job_name: str,
):
    predictor = _recording_predictor()

    with pytest.raises(ValueError, match="job name"):
        get_or_predict_cached(
            store,
            _prediction_request(
                tmp_path,
                20200102,
                20200103,
                job_name=unsafe_job_name,
            ),
            MODEL_CACHE_KEY,
            tmp_path / "prediction-cache",
            tmp_path / "prediction-run",
            None,
            predictor=predictor,
        )

    assert predictor.ranges == []
    assert not (tmp_path / "prediction-cache").exists()
    assert not (tmp_path / "prediction-run").exists()
    assert not (tmp_path / "escaped").exists()


def test_concurrent_prediction_cache_misses_predict_once_and_publish_one_valid_entry(
    store: FeatureStore,
    tmp_path: Path,
):
    database_path = store.db_path
    store.conn.close()
    stores = [
        FeatureStore(database_path, start_date=20200101, read_only=True),
        FeatureStore(database_path, start_date=20200101, read_only=True),
    ]
    request = _prediction_request(tmp_path, 20200102, 20200103)
    start = threading.Barrier(2)
    call_lock = threading.Lock()
    calls: list[int] = []

    def predictor(_store, prediction_request, output_dir):
        with call_lock:
            calls.append(len(calls) + 1)
        time.sleep(0.2)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "prediction.pkl"
        _prediction_frame(prediction_request.start_date, prediction_request.end_date).to_pickle(
            path
        )
        return PredictionArtifact(f"legacy_job__{MODEL_CACHE_KEY}", path, 4)

    def cached(index: int):
        start.wait()
        return get_or_predict_cached(
            stores[index],
            request,
            MODEL_CACHE_KEY,
            tmp_path / "prediction-cache",
            tmp_path / f"concurrent-run-{index}",
            None,
            predictor=predictor,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(cached, range(2)))
    finally:
        for read_store in stores:
            read_store.conn.close()

    assert calls == [1]
    assert sorted(result.cache_mode for result in results) == ["exact", "miss"]
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert {result.cache_key for result in results} == {results[0].cache_key}
    pd.testing.assert_frame_equal(
        pd.read_pickle(results[0].prediction_path),
        pd.read_pickle(results[1].prediction_path),
    )


@pytest.mark.parametrize(
    ("frame_mutator", "message"),
    [
        (lambda frame: frame.drop(columns="score_rank"), "required columns"),
        (
            lambda frame: frame.assign(
                model_artifact_id=[MODEL_CACHE_KEY] * (len(frame) - 1) + ["other-model"]
            ),
            "one model_artifact_id",
        ),
        (lambda frame: frame.assign(score_raw=[math.inf, *frame.score_raw.iloc[1:]]), "finite"),
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), "sorted"),
    ],
    ids=["missing-column", "multiple-models", "non-finite-score", "unsorted"],
)
def test_invalid_generated_prediction_is_rejected(
    store: FeatureStore,
    tmp_path: Path,
    frame_mutator,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        get_or_predict_cached(
            store,
            _prediction_request(tmp_path, 20200102, 20200103),
            MODEL_CACHE_KEY,
            tmp_path / "prediction-cache",
            tmp_path / "invalid-run",
            None,
            predictor=_recording_predictor(frame_mutator=frame_mutator),
        )


def _prediction_request(
    tmp_path: Path,
    start_date: int,
    end_date: int,
    *,
    job_name: str = "legacy_job",
    x_group: str = "base_v1",
    legacy_artifact_hash: str | None = None,
) -> PredictionRequest:
    meta_path = tmp_path / "prediction-model.meta.json"
    model_path = tmp_path / "prediction-model.pkl"
    model_path.write_bytes(b"synthetic-model")
    metadata = {
        "artifact_hash": MODEL_CACHE_KEY,
        "job": {
            "name": job_name,
            "x_group": x_group,
            "features": ["market.close.v1", "market.amount.v1"],
            "target": "target.ret_5d_rank_pct.v1",
        },
    }
    if legacy_artifact_hash is not None:
        metadata["legacy_artifact_hash"] = legacy_artifact_hash
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return PredictionRequest(model_path, meta_path, start_date, end_date)


def _recording_predictor(*, frame_mutator=None):
    def predictor(
        _store: FeatureStore,
        request: PredictionRequest,
        output_dir: str | Path,
    ) -> PredictionArtifact:
        predictor.ranges.append((request.start_date, request.end_date))
        metadata = json.loads(request.meta_path.read_text(encoding="utf-8"))
        source_job = metadata["job"]
        frame = _prediction_frame(
            request.start_date,
            request.end_date,
            job_name=str(source_job["name"]),
            x_group=str(source_job["x_group"]),
        )
        if frame_mutator is not None:
            frame = frame_mutator(frame)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "prediction.pkl"
        frame.to_pickle(path)
        return PredictionArtifact(f"{source_job['name']}__{MODEL_CACHE_KEY}", path, len(frame))

    predictor.ranges = []
    return predictor


def _prediction_frame(
    start_date: int,
    end_date: int,
    *,
    job_name: str = "legacy_job",
    x_group: str = "base_v1",
) -> pd.DataFrame:
    available_dates = [date for date in (20200102, 20200103, 20200106) if start_date <= date <= end_date]
    rows = []
    for date in available_dates:
        for index, code in enumerate(("000001.SZ", "000002.SZ"), start=1):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "score_raw": float(index + date % 100),
                    "score_rank": index / 2,
                    "target.ret_5d_rank_pct.v1": index / 10,
                    "model_artifact_id": MODEL_CACHE_KEY,
                    "train_job_id": job_name,
                    "x_group": x_group,
                    "y_name": "target.ret_5d_rank_pct.v1",
                }
            )
    return pd.DataFrame(rows)


def _seed_values(store: FeatureStore) -> None:
    feature_rows = []
    targets_5d = []
    targets_10d = []
    samples = [
        ("000001.SZ", 20200102, 10.0, 1000.0, 0.1, 0.11),
        ("000002.SZ", 20200102, 20.0, 2000.0, 0.3, 0.31),
        ("000001.SZ", 20200103, 11.0, 1100.0, 0.2, 0.21),
        ("000002.SZ", 20200103, 21.0, 2100.0, 0.4, 0.41),
        ("000001.SZ", 20200106, 12.0, 1200.0, 0.5, 0.51),
        ("000002.SZ", 20200106, 22.0, 2200.0, 0.6, 0.61),
    ]
    for ts_code, trade_date, close, amount, target_5d, target_10d in samples:
        feature_rows.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "market.close.v1": close,
                "market.amount.v1": amount,
            }
        )
        targets_5d.append({"ts_code": ts_code, "trade_date": trade_date, "value": target_5d})
        targets_10d.append({"ts_code": ts_code, "trade_date": trade_date, "value": target_10d})
    store.append_plugin_values(
        "market.raw_fields.v1",
        ["market.close.v1", "market.amount.v1"],
        pd.DataFrame(feature_rows),
    )
    store.append_target_values("target.ret_5d_rank_pct.v1", pd.DataFrame(targets_5d))
    store.append_target_values("target.ret_10d_rank_pct.v1", pd.DataFrame(targets_10d))


def _write_synthetic_model(
    store: FeatureStore,
    job: TrainingJob,
    output_dir: Path,
    model_bytes: bytes,
) -> ModelArtifact:
    contract = build_training_contract(store, job)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{job.name}__{contract.cache_key}.pkl"
    meta_path = output_dir / f"{job.name}__{contract.cache_key}.meta.json"
    model_path.write_bytes(model_bytes)
    metadata = {
        "artifact_hash": contract.cache_key,
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
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ModelArtifact(contract.cache_key, model_path, meta_path, 4)


def _write_legacy_model(
    store: FeatureStore,
    job: TrainingJob,
    legacy_root: Path,
    model_bytes: bytes,
) -> tuple[Path, Path]:
    artifact = _write_synthetic_model(store, job, legacy_root / "completed-run" / "models", model_bytes)
    return artifact.model_path, artifact.meta_path


def _write_model_stage_manifest(
    run_dir: Path,
    artifact: ModelArtifact,
    *,
    cache_key: str | None,
) -> None:
    stage_metadata = (
        {"cache_key": cache_key}
        if cache_key is not None
        else {"artifact_hash": "deadbeef"}
    )
    run_dir.joinpath("run_manifest.json").write_text(
        json.dumps(
            {
                "stages": {
                    "train:model": {
                        "metadata": stage_metadata,
                        "artifacts": [
                            {
                                "path": artifact.model_path.relative_to(
                                    run_dir
                                ).as_posix()
                            },
                            {
                                "path": artifact.meta_path.relative_to(
                                    run_dir
                                ).as_posix()
                            },
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _seed_meta(store: FeatureStore) -> None:
    for feature_name, code_hash in [
        ("market.close.v1", "close-hash"),
        ("market.amount.v1", "amount-hash"),
    ]:
        store.register_feature_meta(
            FeatureMeta(
                feature_name=feature_name,
                domain="market",
                version="v1",
                kind="raw_field",
                owner_plugin="market.raw_fields.v1",
                input_tables=["raw.daily"],
                lookback_days=0,
                code_hash=code_hash,
            )
        )
