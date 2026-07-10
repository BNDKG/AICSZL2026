from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aicszl.features.store import FeatureStore


@dataclass(frozen=True)
class PredictionRequest:
    model_path: Path
    meta_path: Path
    start_date: int
    end_date: int


@dataclass(frozen=True)
class PredictionArtifact:
    prediction_id: str
    prediction_path: Path
    rows: int


def predict_from_artifact(
    store: FeatureStore,
    request: PredictionRequest,
    output_dir: str | Path,
) -> PredictionArtifact:
    metadata = json.loads(Path(request.meta_path).read_text(encoding="utf-8"))
    job = metadata["job"]
    model_artifact_id = metadata["artifact_hash"]
    features = list(job["features"])
    target = str(job["target"])

    with Path(request.model_path).open("rb") as file:
        model = pickle.load(file)

    dataset = _assemble_prediction_frame(store, features, target, request.start_date, request.end_date)
    if dataset.empty:
        raise ValueError("Prediction dataset is empty")

    result = dataset[["ts_code", "trade_date"]].copy()
    result["score_raw"] = model.predict(dataset[features])
    result["score_rank"] = result.groupby("trade_date")["score_raw"].rank(method="average", pct=True)
    result[target] = dataset[target].astype(object).where(pd.notna(dataset[target]), None)
    result["model_artifact_id"] = model_artifact_id
    result["train_job_id"] = job["name"]
    result["x_group"] = job["x_group"]
    result["y_name"] = target
    result = result[
        [
            "ts_code",
            "trade_date",
            "score_raw",
            "score_rank",
            target,
            "model_artifact_id",
            "train_job_id",
            "x_group",
            "y_name",
        ]
    ].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_id = f"{job['name']}__{model_artifact_id}"
    prediction_path = output_path / f"{prediction_id}.pkl"
    result.to_pickle(prediction_path)
    return PredictionArtifact(prediction_id=prediction_id, prediction_path=prediction_path, rows=int(len(result)))


def _assemble_prediction_frame(
    store: FeatureStore,
    features: list[str],
    target: str,
    start_date: int,
    end_date: int,
) -> pd.DataFrame:
    feature_values = _load_feature_values(store, features, start_date, end_date)
    if feature_values.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", *features, target])
    x = feature_values.pivot(
        index=["ts_code", "trade_date"],
        columns="feature_name",
        values="value",
    ).reset_index()
    x.columns.name = None
    x = x.dropna(subset=features)
    y = store.fetch_df(
        """
        SELECT ts_code, trade_date, value
        FROM target_values
        WHERE trade_date BETWEEN ? AND ?
          AND target_name = ?
        """,
        [int(start_date), int(end_date), target],
    ).rename(columns={"value": target})
    if y.empty:
        x[target] = None
        return x[["ts_code", "trade_date", *features, target]]
    return x.merge(y[["ts_code", "trade_date", target]], on=["ts_code", "trade_date"], how="left")[
        ["ts_code", "trade_date", *features, target]
    ]


def _load_feature_values(
    store: FeatureStore,
    features: list[str],
    start_date: int,
    end_date: int,
) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in features)
    return store.fetch_df(
        f"""
        SELECT ts_code, trade_date, feature_name, value
        FROM feature_values
        WHERE trade_date BETWEEN ? AND ?
          AND feature_name IN ({placeholders})
        """,
        [int(start_date), int(end_date), *features],
    )
