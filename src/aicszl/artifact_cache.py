from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from aicszl.features.store import FeatureStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from aicszl.models.training import ModelArtifact, TrainingJob


MODEL_CACHE_SCHEMA_VERSION = 2
PREDICTION_CACHE_SCHEMA_VERSION = 2
_MODEL_CACHE_LOCK_TIMEOUT_SECONDS = 300.0
_MODEL_CACHE_LOCK_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class TrainingContract:
    cache_key: str
    payload: dict[str, Any]
    feature_code_hashes: dict[str, str]
    data_fingerprint: dict[str, Any]


@dataclass(frozen=True)
class CachedModelArtifact:
    artifact_hash: str
    cache_key: str
    cache_hit: bool
    cache_source: str
    model_path: Path
    meta_path: Path
    train_rows: int


@dataclass(frozen=True)
class CachedPredictionArtifact:
    prediction_id: str
    prediction_path: Path
    rows: int
    cache_hit: bool
    cache_mode: str
    cache_key: str
    cache_source: str
    reused_rows: int
    generated_rows: int
    generated_range: tuple[int, int] | None


def semantic_training_payload(job: TrainingJob) -> dict[str, Any]:
    return {
        "features": list(job.features),
        "target": str(job.target),
        "train_range": _inclusive_range(job.train_range[0], job.train_range[1]),
        "filters": list(job.filters),
        "model": str(job.model),
        "model_params": {key: job.model_params[key] for key in sorted(job.model_params)},
    }


def build_training_contract(store: FeatureStore, job: TrainingJob) -> TrainingContract:
    training_payload = semantic_training_payload(job)
    feature_code_hashes = _feature_code_hashes(store, job.features)
    data_fingerprint = prediction_data_fingerprint(
        store,
        job.features,
        job.target,
        job.train_range[0],
        job.train_range[1],
    )
    payload = {
        "training": training_payload,
        "feature_code_hashes": feature_code_hashes,
        "data_fingerprint": data_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return TrainingContract(
        cache_key=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        payload=payload,
        feature_code_hashes=feature_code_hashes,
        data_fingerprint=data_fingerprint,
    )


def get_or_train_cached_model(
    store: FeatureStore,
    job: TrainingJob,
    cache_root: str | Path,
    run_output_dir: str | Path,
    legacy_root: str | Path | None,
    *,
    trainer: Callable[[FeatureStore, TrainingJob, str | Path], ModelArtifact] | None = None,
) -> CachedModelArtifact:
    # Fingerprint-less pre-cache artifacts may only be resumed in their original
    # run. They cannot establish provenance for a global cache entry.
    _ = legacy_root
    contract = build_training_contract(store, job)
    cache_entry = Path(cache_root) / contract.cache_key
    manifest = _validated_model_manifest(cache_entry, contract)
    cache_hit = manifest is not None

    if manifest is None:
        with _model_cache_lock(cache_entry):
            manifest = _validated_model_manifest(cache_entry, contract)
            if manifest is not None:
                cache_hit = True
            else:
                if trainer is None:
                    from aicszl.models.training import train_lightgbm_regressor

                    trainer = train_lightgbm_regressor
                _train_and_publish_model(store, job, cache_entry, contract, trainer)
                manifest = _validated_model_manifest(cache_entry, contract)
                if manifest is None:
                    raise RuntimeError(f"Published model cache failed validation: {cache_entry}")

    model_source = cache_entry / manifest["model"]["path"]
    meta_source = cache_entry / manifest["metadata"]["path"]
    run_dir = Path(run_output_dir)
    run_model = run_dir / f"{job.name}__{contract.cache_key}.pkl"
    run_meta = run_dir / f"{job.name}__{contract.cache_key}.meta.json"
    _materialize_file(model_source, run_model)
    _materialize_model_metadata(
        meta_source,
        run_meta,
        run_model,
        job,
        cache_entry,
        contract.cache_key,
    )
    return CachedModelArtifact(
        artifact_hash=contract.cache_key,
        cache_key=contract.cache_key,
        cache_hit=cache_hit,
        cache_source=str(cache_entry.resolve()),
        model_path=run_model,
        meta_path=run_meta,
        train_rows=int(manifest["train_rows"]),
    )


def _train_and_publish_model(
    store: FeatureStore,
    job: TrainingJob,
    cache_entry: Path,
    contract: TrainingContract,
    trainer: Callable[[FeatureStore, TrainingJob, str | Path], ModelArtifact],
) -> dict[str, Any]:
    staging_dir = cache_entry.parent / f".{contract.cache_key}.{uuid.uuid4().hex}.tmp"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact = trainer(store, job, staging_dir)
        _require_staged_file(Path(artifact.model_path), staging_dir)
        _require_staged_file(Path(artifact.meta_path), staging_dir)
        return _publish_model_files(
            cache_entry,
            contract,
            Path(artifact.model_path),
            Path(artifact.meta_path),
            int(artifact.train_rows),
            source="trained",
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _publish_model_files(
    cache_entry: Path,
    contract: TrainingContract,
    staged_model: Path,
    staged_meta: Path,
    train_rows: int,
    *,
    source: str,
) -> dict[str, Any]:
    cache_entry.mkdir(parents=True, exist_ok=True)
    model_path = cache_entry / "model.pkl"
    meta_path = cache_entry / "model.meta.json"
    manifest_path = cache_entry / "cache.json"
    manifest_tmp = cache_entry / f".cache.{uuid.uuid4().hex}.json.tmp"
    manifest = {
        "schema_version": MODEL_CACHE_SCHEMA_VERSION,
        "cache_key": contract.cache_key,
        "contract": contract.payload,
        "model": {"path": model_path.name, "sha256": _sha256_file(staged_model)},
        "metadata": {"path": meta_path.name, "sha256": _sha256_file(staged_meta)},
        "train_rows": int(train_rows),
        "source": source,
    }
    try:
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staged_model, model_path)
        os.replace(staged_meta, meta_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def _validated_model_manifest(
    cache_entry: Path,
    contract: TrainingContract,
) -> dict[str, Any] | None:
    manifest_path = cache_entry / "cache.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema_version") != MODEL_CACHE_SCHEMA_VERSION:
            return None
        if manifest.get("cache_key") != contract.cache_key:
            return None
        if manifest.get("contract") != contract.payload:
            return None
        if manifest.get("source") != "trained" or "legacy_source" in manifest:
            return None
        model_info = manifest.get("model")
        meta_info = manifest.get("metadata")
        if not isinstance(model_info, dict) or not isinstance(meta_info, dict):
            return None
        if model_info.get("path") != "model.pkl" or meta_info.get("path") != "model.meta.json":
            return None
        model_path = cache_entry / model_info["path"]
        meta_path = cache_entry / meta_info["path"]
        if not model_path.is_file() or not meta_path.is_file():
            return None
        if _sha256_file(model_path) != model_info.get("sha256"):
            return None
        if _sha256_file(meta_path) != meta_info.get("sha256"):
            return None
        train_rows = manifest.get("train_rows")
        if not isinstance(train_rows, int) or isinstance(train_rows, bool) or train_rows < 0:
            return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return manifest


@contextmanager
def _model_cache_lock(cache_entry: Path):
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_entry.parent / f".{cache_entry.name}.lock"
    lock_file = lock_path.open("a+b")
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        deadline = time.monotonic() + _MODEL_CACHE_LOCK_TIMEOUT_SECONDS
        while not acquired:
            acquired = _try_lock_file(lock_file)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for model cache lock: {lock_path}")
            time.sleep(_MODEL_CACHE_LOCK_POLL_SECONDS)
        yield
    finally:
        if acquired:
            _unlock_file(lock_file)
        lock_file.close()


def _try_lock_file(lock_file) -> bool:
    lock_file.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_file(lock_file) -> None:
    lock_file.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _require_staged_file(path: Path, staging_dir: Path) -> None:
    if not path.is_file() or path.resolve().parent != staging_dir.resolve():
        raise ValueError(f"Trainer artifact is outside its staging directory: {path}")


def _materialize_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_model_metadata(
    source: Path,
    destination: Path,
    model_path: Path,
    job: TrainingJob,
    cache_entry: Path,
    cache_key: str,
) -> None:
    metadata = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("job"), dict):
        raise ValueError("Cached model metadata must contain a job mapping")
    source_job = dict(metadata["job"])
    metadata["job"] = {
        **source_job,
        "name": str(job.name),
        "x_group": str(job.x_group),
    }
    metadata["model_path"] = str(model_path.resolve())
    metadata["cache_provenance"] = {
        "cache_key": cache_key,
        "cache_source": str(cache_entry.resolve()),
        "source_job": source_job,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_or_predict_cached(
    store: FeatureStore,
    request,
    model_cache_key: str,
    cache_root: str | Path,
    run_output_dir: str | Path,
    legacy_prediction_dir: str | Path | None,
    *,
    predictor=None,
) -> CachedPredictionArtifact:
    # Kept for call compatibility only. Legacy predictions do not carry their
    # original data fingerprint and therefore cannot seed the global cache.
    _ = legacy_prediction_dir
    if not isinstance(model_cache_key, str) or re.fullmatch(r"[0-9a-f]{64}", model_cache_key) is None:
        raise ValueError("model_cache_key must be a lowercase 64-character hexadecimal digest")
    metadata = _prediction_metadata(request)
    if metadata["artifact_hash"] != model_cache_key:
        raise ValueError("Prediction model metadata artifact_hash must match model_cache_key")
    features = metadata["features"]
    target = metadata["target"]
    prediction_id = f"{metadata['job_name']}__{model_cache_key}"
    contract = _prediction_contract(
        store,
        model_cache_key,
        features,
        target,
        request.start_date,
        request.end_date,
    )
    cache_key = _prediction_cache_key(contract)
    model_cache_dir = Path(cache_root) / model_cache_key
    manifest_path = model_cache_dir / f"{cache_key}.json"

    exact = _validated_prediction_manifest(manifest_path, contract, store)
    if exact is not None:
        return _materialize_cached_prediction(
            exact,
            model_cache_dir,
            prediction_id,
            run_output_dir,
            metadata["job_name"],
            metadata["x_group"],
            cache_mode="exact",
            cache_hit=True,
            reused_rows=int(exact["rows"]),
            generated_rows=0,
            generated_range=None,
        )

    with _model_cache_lock(model_cache_dir):
        exact = _validated_prediction_manifest(manifest_path, contract, store)
        if exact is not None:
            return _materialize_cached_prediction(
                exact,
                model_cache_dir,
                prediction_id,
                run_output_dir,
                metadata["job_name"],
                metadata["x_group"],
                cache_mode="exact",
                cache_hit=True,
                reused_rows=int(exact["rows"]),
                generated_rows=0,
                generated_range=None,
            )

        candidates = _compatible_prediction_manifests(
            model_cache_dir,
            model_cache_key,
            features,
            target,
            store,
        )
        result = _reuse_prediction_candidate(
            store,
            request,
            contract,
            cache_key,
            model_cache_dir,
            prediction_id,
            run_output_dir,
            candidates,
            predictor,
            metadata["job_name"],
            metadata["x_group"],
        )
        if result is not None:
            return result

        if predictor is None:
            from aicszl.predictions.runner import predict_from_artifact

            predictor = predict_from_artifact
        frame = _run_prediction_into_staging(
            store,
            request,
            predictor,
            model_cache_dir,
            model_cache_key,
            features,
            target,
        )
        manifest = _publish_prediction_frame(
            model_cache_dir,
            cache_key,
            contract,
            frame,
            source="miss",
        )
        validated = _validated_prediction_manifest(manifest_path, contract, store)
        if validated is None:
            raise RuntimeError(f"Published prediction cache failed validation: {manifest_path}")
        generated_range = _frame_date_range(frame)
        return _materialize_cached_prediction(
            validated,
            model_cache_dir,
            prediction_id,
            run_output_dir,
            metadata["job_name"],
            metadata["x_group"],
            cache_mode="miss",
            cache_hit=False,
            reused_rows=0,
            generated_rows=int(manifest["rows"]),
            generated_range=generated_range,
        )


def _prediction_metadata(request) -> dict[str, Any]:
    metadata = json.loads(Path(request.meta_path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("job"), dict):
        raise ValueError("Prediction model metadata must contain a job mapping")
    job = metadata["job"]
    try:
        features = [str(feature) for feature in job["features"]]
        target = str(job["target"])
        job_name = str(job["name"])
        x_group = str(job["x_group"])
        artifact_hash = str(metadata["artifact_hash"])
    except (KeyError, TypeError) as error:
        raise ValueError("Prediction model metadata is missing required job fields") from error
    if not features or len(features) != len(set(features)):
        raise ValueError("Prediction model features must be non-empty and unique")
    if int(request.start_date) > int(request.end_date):
        raise ValueError("Prediction start_date must not exceed end_date")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_name) is None:
        raise ValueError("Prediction model job name must be a safe filename component")
    return {
        "features": features,
        "target": target,
        "job_name": job_name,
        "x_group": x_group,
        "artifact_hash": artifact_hash,
    }


def _prediction_contract(
    store: FeatureStore,
    model_cache_key: str,
    features: list[str],
    target: str,
    start_date: int,
    end_date: int,
) -> dict[str, Any]:
    return {
        "model_cache_key": str(model_cache_key),
        "range": _inclusive_range(start_date, end_date),
        "features": list(features),
        "target": str(target),
        "data_fingerprint": prediction_data_fingerprint(
            store,
            features,
            target,
            start_date,
            end_date,
        ),
    }


def _prediction_cache_key(contract: dict[str, Any]) -> str:
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_prediction_manifest(
    manifest_path: Path,
    expected_contract: dict[str, Any] | None,
    store: FeatureStore,
) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema_version") != PREDICTION_CACHE_SCHEMA_VERSION:
            return None
        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            return None
        cache_key = _prediction_cache_key(contract)
        if manifest.get("cache_key") != cache_key or manifest_path.stem != cache_key:
            return None
        if expected_contract is not None and contract != expected_contract:
            return None
        if manifest.get("source") not in {"miss", "slice", "extend"}:
            return None
        if "legacy_source" in manifest:
            return None
        prediction = manifest.get("prediction")
        if not isinstance(prediction, dict):
            return None
        expected_name = f"{cache_key}.pkl"
        if prediction.get("path") != expected_name:
            return None
        prediction_path = manifest_path.parent / expected_name
        if not prediction_path.is_file():
            return None
        if _sha256_file(prediction_path) != prediction.get("sha256"):
            return None
        rows = manifest.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            return None
        frame = pd.read_pickle(prediction_path)
        _validate_prediction_frame(
            frame,
            str(contract["model_cache_key"]),
            list(contract["features"]),
            str(contract["target"]),
            int(contract["range"]["start_date"]),
            int(contract["range"]["end_date"]),
            store,
        )
        if len(frame) != rows:
            return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, EOFError):
        return None
    return manifest


def _compatible_prediction_manifests(
    model_cache_dir: Path,
    model_cache_key: str,
    features: list[str],
    target: str,
    store: FeatureStore,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not model_cache_dir.is_dir():
        return candidates
    for manifest_path in model_cache_dir.glob("*.json"):
        manifest = _validated_prediction_manifest(manifest_path, None, store)
        if manifest is None:
            continue
        contract = manifest["contract"]
        if (
            contract.get("model_cache_key") == model_cache_key
            and contract.get("features") == features
            and contract.get("target") == target
        ):
            current_contract = _prediction_contract(
                store,
                model_cache_key,
                features,
                target,
                int(contract["range"]["start_date"]),
                int(contract["range"]["end_date"]),
            )
            if current_contract == contract:
                candidates.append(manifest)
    return candidates


def _reuse_prediction_candidate(
    store: FeatureStore,
    request,
    requested_contract: dict[str, Any],
    requested_cache_key: str,
    model_cache_dir: Path,
    prediction_id: str,
    run_output_dir: str | Path,
    candidates: list[dict[str, Any]],
    predictor,
    job_name: str,
    x_group: str,
) -> CachedPredictionArtifact | None:
    requested_start = int(request.start_date)
    requested_end = int(request.end_date)
    supersets = [
        candidate
        for candidate in candidates
        if int(candidate["contract"]["range"]["start_date"]) <= requested_start
        and int(candidate["contract"]["range"]["end_date"]) >= requested_end
    ]
    if supersets:
        candidate = min(
            supersets,
            key=lambda value: (
                int(value["contract"]["range"]["end_date"])
                - int(value["contract"]["range"]["start_date"]),
                value["cache_key"],
            ),
        )
        source = model_cache_dir / candidate["prediction"]["path"]
        frame = pd.read_pickle(source)
        sliced = frame[
            frame["trade_date"].between(requested_start, requested_end, inclusive="both")
        ].reset_index(drop=True)
        _validate_prediction_frame(
            sliced,
            str(requested_contract["model_cache_key"]),
            list(requested_contract["features"]),
            str(requested_contract["target"]),
            requested_start,
            requested_end,
            store,
        )
        manifest = _publish_prediction_frame(
            model_cache_dir,
            requested_cache_key,
            requested_contract,
            sliced,
            source="slice",
        )
        validated = _validated_prediction_manifest(
            model_cache_dir / f"{requested_cache_key}.json", requested_contract, store
        )
        if validated is None:
            raise RuntimeError("Published sliced prediction cache failed validation")
        return _materialize_cached_prediction(
            validated,
            model_cache_dir,
            prediction_id,
            run_output_dir,
            job_name,
            x_group,
            cache_mode="slice",
            cache_hit=True,
            reused_rows=int(manifest["rows"]),
            generated_rows=0,
            generated_range=None,
        )

    prefixes = [
        candidate
        for candidate in candidates
        if int(candidate["contract"]["range"]["start_date"]) == requested_start
        and int(candidate["contract"]["range"]["end_date"]) < requested_end
    ]
    if not prefixes:
        return None
    prefix = max(
        prefixes,
        key=lambda value: (int(value["contract"]["range"]["end_date"]), value["cache_key"]),
    )
    prefix_end = int(prefix["contract"]["range"]["end_date"])
    available_tail = _available_prediction_dates(
        store,
        list(requested_contract["features"]),
        prefix_end + 1,
        requested_end,
    )
    prefix_frame = pd.read_pickle(model_cache_dir / prefix["prediction"]["path"])
    if available_tail:
        if predictor is None:
            from aicszl.predictions.runner import predict_from_artifact

            predictor = predict_from_artifact
        tail_request = replace(request, start_date=available_tail[0], end_date=requested_end)
        generated = _run_prediction_into_staging(
            store,
            tail_request,
            predictor,
            model_cache_dir,
            str(requested_contract["model_cache_key"]),
            list(requested_contract["features"]),
            str(requested_contract["target"]),
        )
        combined = pd.concat([prefix_frame, generated], ignore_index=True)
        combined = combined.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        generated_range = _frame_date_range(generated)
    else:
        generated = prefix_frame.iloc[0:0].copy()
        combined = prefix_frame.copy(deep=True)
        generated_range = None
    _validate_prediction_frame(
        combined,
        str(requested_contract["model_cache_key"]),
        list(requested_contract["features"]),
        str(requested_contract["target"]),
        requested_start,
        requested_end,
        store,
    )
    manifest = _publish_prediction_frame(
        model_cache_dir,
        requested_cache_key,
        requested_contract,
        combined,
        source="extend",
    )
    validated = _validated_prediction_manifest(
        model_cache_dir / f"{requested_cache_key}.json", requested_contract, store
    )
    if validated is None:
        raise RuntimeError("Published extended prediction cache failed validation")
    return _materialize_cached_prediction(
        validated,
        model_cache_dir,
        prediction_id,
        run_output_dir,
        job_name,
        x_group,
        cache_mode="extend",
        cache_hit=True,
        reused_rows=len(prefix_frame),
        generated_rows=len(generated),
        generated_range=generated_range,
    )


def _run_prediction_into_staging(
    store: FeatureStore,
    request,
    predictor,
    model_cache_dir: Path,
    model_cache_key: str,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    staging_dir = model_cache_dir / f".predict.{uuid.uuid4().hex}.tmp"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact = predictor(store, request, staging_dir)
        artifact_path = Path(artifact.prediction_path)
        _require_staged_file(artifact_path, staging_dir)
        frame = pd.read_pickle(artifact_path)
        _validate_prediction_frame(
            frame,
            model_cache_key,
            features,
            target,
            int(request.start_date),
            int(request.end_date),
            store,
        )
        return frame.copy(deep=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _publish_prediction_frame(
    model_cache_dir: Path,
    cache_key: str,
    contract: dict[str, Any],
    frame: pd.DataFrame,
    *,
    source: str,
) -> dict[str, Any]:
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    staging_path = model_cache_dir / f".{cache_key}.{uuid.uuid4().hex}.pkl.tmp"
    prediction_path = model_cache_dir / f"{cache_key}.pkl"
    manifest_path = model_cache_dir / f"{cache_key}.json"
    manifest_tmp = model_cache_dir / f".{cache_key}.{uuid.uuid4().hex}.json.tmp"
    try:
        frame.to_pickle(staging_path)
        manifest = {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "contract": contract,
            "prediction": {
                "path": prediction_path.name,
                "sha256": _sha256_file(staging_path),
            },
            "rows": int(len(frame)),
            "source": source,
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging_path, prediction_path)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    finally:
        staging_path.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _materialize_cached_prediction(
    manifest: dict[str, Any],
    model_cache_dir: Path,
    prediction_id: str,
    run_output_dir: str | Path,
    job_name: str,
    x_group: str,
    *,
    cache_mode: str,
    cache_hit: bool,
    reused_rows: int,
    generated_rows: int,
    generated_range: tuple[int, int] | None,
) -> CachedPredictionArtifact:
    cache_path = model_cache_dir / manifest["prediction"]["path"]
    run_path = Path(run_output_dir) / f"{prediction_id}.pkl"
    _materialize_file(cache_path, run_path)
    _rewrite_prediction_identity(run_path, job_name, x_group)
    return CachedPredictionArtifact(
        prediction_id=prediction_id,
        prediction_path=run_path,
        rows=int(manifest["rows"]),
        cache_hit=cache_hit,
        cache_mode=cache_mode,
        cache_key=str(manifest["cache_key"]),
        cache_source=str(cache_path.resolve()),
        reused_rows=int(reused_rows),
        generated_rows=int(generated_rows),
        generated_range=generated_range,
    )


def _rewrite_prediction_identity(path: Path, job_name: str, x_group: str) -> None:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Cached prediction artifact must be a DataFrame")
    rewritten = frame.copy(deep=True)
    if "cache_source_train_job_id" not in rewritten:
        rewritten["cache_source_train_job_id"] = rewritten["train_job_id"]
    if "cache_source_x_group" not in rewritten:
        rewritten["cache_source_x_group"] = rewritten["x_group"]
    rewritten["train_job_id"] = job_name
    rewritten["x_group"] = x_group
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        rewritten.to_pickle(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_prediction_frame(
    frame: pd.DataFrame,
    model_cache_key: str,
    features: list[str],
    target: str,
    start_date: int,
    end_date: int,
    store: FeatureStore,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Prediction artifact must be a non-empty DataFrame")
    required = {
        "ts_code",
        "trade_date",
        "score_raw",
        "score_rank",
        target,
        "model_artifact_id",
        "train_job_id",
        "x_group",
        "y_name",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction artifact missing required columns: {missing}")
    model_ids = frame["model_artifact_id"].drop_duplicates().tolist()
    if len(model_ids) != 1 or model_ids[0] != model_cache_key:
        raise ValueError("Prediction artifact must contain one model_artifact_id matching the cache key")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Prediction artifact contains duplicate (trade_date, ts_code) keys")
    for column in ("score_raw", "score_rank"):
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Prediction artifact {column} values must be finite") from error
        if not numeric.map(lambda value: math.isfinite(float(value))).all():
            raise ValueError(f"Prediction artifact {column} values must be finite")
    sorted_keys = frame[["trade_date", "ts_code"]].sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)
    if not frame[["trade_date", "ts_code"]].reset_index(drop=True).equals(sorted_keys):
        raise ValueError("Prediction artifact rows must be sorted by trade_date, ts_code")
    dates = [int(value) for value in frame["trade_date"].drop_duplicates().tolist()]
    if dates[0] < int(start_date) or dates[-1] > int(end_date):
        raise ValueError("Prediction artifact dates fall outside the requested range")
    available_dates = _available_prediction_dates(store, features, start_date, end_date)
    if dates != available_dates:
        raise ValueError("Prediction artifact dates do not cover the continuous available date range")


def _available_prediction_dates(
    store: FeatureStore,
    features: list[str],
    start_date: int,
    end_date: int,
) -> list[int]:
    from aicszl.predictions.runner import prediction_available_dates

    return prediction_available_dates(store, features, start_date, end_date)


def _frame_date_range(frame: pd.DataFrame) -> tuple[int, int] | None:
    if frame.empty:
        return None
    return int(frame["trade_date"].min()), int(frame["trade_date"].max())


def prediction_data_fingerprint(
    store: FeatureStore,
    features: list[str],
    target: str,
    start_date: int,
    end_date: int,
) -> dict[str, Any]:
    normalized_features = list(features)
    feature_rows = _feature_value_aggregates(store, normalized_features, start_date, end_date)
    target_row = _target_value_aggregate(store, target, start_date, end_date)
    return {
        "range": _inclusive_range(start_date, end_date),
        "features": [
            {
                "feature_name": feature,
                **feature_rows.get(
                    feature,
                    {"row_count": 0, "hash_xor": "0", "hash_sum": "0"},
                ),
            }
            for feature in normalized_features
        ],
        "target": {
            "target_name": str(target),
            **target_row,
        },
    }


def _feature_code_hashes(store: FeatureStore, features: list[str]) -> dict[str, str]:
    if not features:
        return {}
    placeholders = ", ".join("?" for _ in features)
    rows = store.fetch_df(
        f"""
        SELECT feature_name, code_hash
        FROM feature_meta
        WHERE feature_name IN ({placeholders})
        """,
        list(features),
    )
    hashes = dict(zip(rows["feature_name"], rows["code_hash"], strict=False))
    missing = [feature for feature in features if feature not in hashes]
    if missing:
        raise ValueError(f"Missing feature metadata: {missing}")
    return {feature: str(hashes[feature]) for feature in sorted(hashes)}


def _feature_value_aggregates(
    store: FeatureStore,
    features: list[str],
    start_date: int,
    end_date: int,
) -> dict[str, dict[str, int | str]]:
    return store.feature_value_aggregates(features, start_date, end_date)


def _target_value_aggregate(
    store: FeatureStore,
    target: str,
    start_date: int,
    end_date: int,
) -> dict[str, int | str]:
    return store.target_value_aggregate(target, start_date, end_date)


def _inclusive_range(start_date: int, end_date: int) -> dict[str, int | bool]:
    return {
        "start_date": int(start_date),
        "end_date": int(end_date),
        "inclusive": True,
    }
