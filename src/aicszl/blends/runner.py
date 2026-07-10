from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BlendInput:
    prediction_id: str
    path: Path
    weight: float


@dataclass(frozen=True)
class BlendJob:
    name: str
    inputs: list[BlendInput]
    method: str = "weighted_mean"
    normalize: str = "daily_rank"


@dataclass(frozen=True)
class BlendArtifact:
    blend_id: str
    blend_path: Path
    rows: int


def blend_predictions(job: BlendJob, output_dir: str | Path) -> BlendArtifact:
    if job.method != "weighted_mean":
        raise ValueError(f"Unsupported blend method: {job.method}")
    if job.normalize != "daily_rank":
        raise ValueError(f"Unsupported blend normalization: {job.normalize}")
    if not job.inputs:
        raise ValueError("Blend job must contain at least one input")

    merged: pd.DataFrame | None = None
    weighted_columns: list[str] = []
    total_weight = 0.0
    for index, item in enumerate(job.inputs):
        frame = pd.read_pickle(item.path)[["ts_code", "trade_date", "score_raw"]].copy()
        column = f"weighted_score_{index}"
        frame[column] = frame["score_raw"] * float(item.weight)
        frame = frame[["ts_code", "trade_date", column]]
        merged = frame if merged is None else merged.merge(frame, on=["ts_code", "trade_date"], how="inner")
        weighted_columns.append(column)
        total_weight += float(item.weight)
    if merged is None or merged.empty:
        raise ValueError("Blend input intersection is empty")
    if total_weight == 0:
        raise ValueError("Blend input weights must not sum to zero")

    input_ids = ",".join(item.prediction_id for item in job.inputs)
    result = merged[["ts_code", "trade_date"]].copy()
    result["score_raw_blend"] = merged[weighted_columns].sum(axis=1) / total_weight
    result["score_rank_blend"] = result.groupby("trade_date")["score_raw_blend"].rank(
        method="average",
        pct=True,
    )
    result["input_prediction_ids"] = input_ids
    result["blend_job_id"] = job.name
    result = result[
        [
            "ts_code",
            "trade_date",
            "score_raw_blend",
            "score_rank_blend",
            "input_prediction_ids",
            "blend_job_id",
        ]
    ].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    blend_id = f"{job.name}__{_blend_hash(job)}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    blend_path = output_path / f"{blend_id}.pkl"
    result.to_pickle(blend_path)
    return BlendArtifact(blend_id=blend_id, blend_path=blend_path, rows=int(len(result)))


def _blend_hash(job: BlendJob) -> str:
    payload = asdict(job)
    payload["inputs"] = [
        {
            "prediction_id": item.prediction_id,
            "path": str(item.path),
            "weight": item.weight,
        }
        for item in job.inputs
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
