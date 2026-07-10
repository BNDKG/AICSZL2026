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
        f"""
        SELECT DISTINCT feature_name
        FROM feature_values
        WHERE feature_name IN ({placeholders})
        """,
        list(features),
    )
    known_features = set(known["feature_name"].tolist())
    missing = [feature for feature in features if feature not in known_features]
    if missing:
        raise ValueError(f"Unknown features: {missing}")


def assemble_dataset(store: FeatureStore, request: DatasetRequest) -> pd.DataFrame:
    validate_feature_group(store, request.features)
    feature_values = _load_feature_values(store, request)
    target_values = _load_target_values(store, request)
    if feature_values.empty or target_values.empty:
        return _empty_dataset(request)

    x = feature_values.pivot(
        index=["ts_code", "trade_date"],
        columns="feature_name",
        values="value",
    ).reset_index()
    x.columns.name = None
    y = target_values.rename(columns={"value": request.target})[
        ["ts_code", "trade_date", request.target]
    ]
    dataset = x.merge(y, on=["ts_code", "trade_date"], how="inner")
    required_columns = ["ts_code", "trade_date", *request.features, request.target]
    dataset = dataset.dropna(subset=[*request.features, request.target])
    dataset = dataset[required_columns]

    for expression in request.filters:
        dataset = dataset.query(_quote_filter_columns(expression, dataset.columns), engine="python")

    return dataset.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _load_feature_values(store: FeatureStore, request: DatasetRequest) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in request.features)
    return store.fetch_df(
        f"""
        SELECT ts_code, trade_date, feature_name, value
        FROM feature_values
        WHERE trade_date BETWEEN ? AND ?
          AND feature_name IN ({placeholders})
        """,
        [int(request.start_date), int(request.end_date), *request.features],
    )


def _load_target_values(store: FeatureStore, request: DatasetRequest) -> pd.DataFrame:
    return store.fetch_df(
        """
        SELECT ts_code, trade_date, target_name, value
        FROM target_values
        WHERE trade_date BETWEEN ? AND ?
          AND target_name = ?
        """,
        [int(request.start_date), int(request.end_date), request.target],
    )


def _empty_dataset(request: DatasetRequest) -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_code", "trade_date", *request.features, request.target])


def _quote_filter_columns(expression: str, columns: pd.Index) -> str:
    quoted = expression
    for column in sorted((str(name) for name in columns), key=len, reverse=True):
        if "." in column:
            quoted = quoted.replace(column, f"`{column}`")
    return quoted
