from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aicszl.features.store import FeatureStore


@dataclass(frozen=True)
class DatasetRequest:
    features: list[str]
    target: str
    start_date: int
    end_date: int
    filters: list[str] = field(default_factory=list)


def validate_feature_group(store: FeatureStore, features: list[str]) -> None:
    if not features:
        raise ValueError("Feature group must contain at least one feature")
    placeholders = ", ".join("?" for _ in features)
    known = store.fetch_df(
        f"SELECT feature_name FROM feature_meta WHERE status = 'active' "
        f"AND feature_name IN ({placeholders})",
        list(features),
    )
    known_features = set(known["feature_name"].tolist())
    missing = [feature for feature in features if feature not in known_features]
    if missing:
        raise ValueError(f"Unknown features: {missing}")


def assemble_dataset(store: FeatureStore, request: DatasetRequest) -> pd.DataFrame:
    validate_feature_group(store, request.features)
    x = store.load_feature_frame(request.features, request.start_date, request.end_date)
    y = store.load_target_frame(request.target, request.start_date, request.end_date)
    if x.empty or y.empty:
        return _empty_dataset(request)
    dataset = x.merge(y, on=["ts_code", "trade_date"], how="inner")
    required_columns = ["ts_code", "trade_date", *request.features, request.target]
    dataset = dataset.dropna(subset=[*request.features, request.target])
    dataset = dataset[required_columns]

    for expression in request.filters:
        dataset = dataset.query(_quote_filter_columns(expression, dataset.columns), engine="python")

    return dataset.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _empty_dataset(request: DatasetRequest) -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", *request.features, request.target])


def _quote_filter_columns(expression: str, columns: pd.Index) -> str:
    quoted = expression
    for column in sorted((str(name) for name in columns), key=len, reverse=True):
        if "." in column:
            quoted = quoted.replace(column, f"`{column}`")
    return quoted
