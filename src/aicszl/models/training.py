from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lightgbm import LGBMRegressor

from aicszl.datasets import DatasetRequest, assemble_dataset
from aicszl.features.store import FeatureStore


@dataclass(frozen=True)
class TrainingJob:
    name: str
    x_group: str
    features: list[str]
    target: str
    train_range: tuple[int, int]
    filters: list[str] = field(default_factory=list)
    model: str = "lgbm_regressor_v1"
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifact:
    artifact_hash: str
    model_path: Path
    meta_path: Path
    train_rows: int


def train_lightgbm_regressor(
    store: FeatureStore,
    job: TrainingJob,
    output_dir: str | Path,
) -> ModelArtifact:
    dataset = assemble_dataset(
        store,
        DatasetRequest(
            features=job.features,
            target=job.target,
            start_date=job.train_range[0],
            end_date=job.train_range[1],
            filters=job.filters,
        ),
    )
    if dataset.empty:
        raise ValueError("Training dataset is empty")

    feature_hashes = _feature_code_hashes(store, job.features)
    artifact_hash = compute_artifact_identity(job, feature_hashes)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{job.name}__{artifact_hash}.pkl"
    meta_path = output_path / f"{job.name}__{artifact_hash}.meta.json"

    model = LGBMRegressor(**job.model_params)
    model.fit(dataset[job.features], dataset[job.target])

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    metadata = {
        "artifact_hash": artifact_hash,
        "job": _normalized_job(job),
        "feature_code_hashes": feature_hashes,
        "train_rows": int(len(dataset)),
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ModelArtifact(
        artifact_hash=artifact_hash,
        model_path=model_path,
        meta_path=meta_path,
        train_rows=int(len(dataset)),
    )


def compute_artifact_identity(job: TrainingJob, feature_code_hashes: dict[str, str]) -> str:
    payload = {
        "job": _normalized_job(job),
        "feature_code_hashes": {name: feature_code_hashes[name] for name in sorted(feature_code_hashes)},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


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
    return {feature: str(hashes[feature]) for feature in features}


def _normalized_job(job: TrainingJob) -> dict[str, Any]:
    raw = asdict(job)
    raw["features"] = list(job.features)
    raw["filters"] = list(job.filters)
    raw["train_range"] = list(job.train_range)
    raw["model_params"] = {key: job.model_params[key] for key in sorted(job.model_params)}
    return raw
