from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from aicszl.targets import TargetDefinition


@dataclass(frozen=True)
class ResolvedExperimentTiming:
    train_dates: tuple[int, ...]
    predict_dates: tuple[int, ...]
    execution_dates: tuple[int, ...]
    last_train_target_exit: int
    signal_to_execution: dict[int, int]


def resolve_experiment_timing(
    *,
    calendar: Sequence[int],
    train_dates: Sequence[int],
    predict_dates: Sequence[int],
    definition: TargetDefinition,
) -> ResolvedExperimentTiming:
    open_dates = tuple(sorted({int(date) for date in calendar}))
    requested_train = tuple(int(date) for date in train_dates)
    requested_predict = tuple(int(date) for date in predict_dates)
    if not requested_train:
        raise ValueError("Training range contains no open trading dates")
    if not requested_predict:
        raise ValueError("Prediction range contains no open trading dates")
    index_by_date = {date: index for index, date in enumerate(open_dates)}

    target_exits = {
        date: _future_date(
            open_dates,
            index_by_date,
            date,
            definition.exit_offset,
        )
        for date in requested_train
    }
    if definition.purge_before_predict:
        first_prediction = requested_predict[0]
        resolved_train = tuple(
            date for date in requested_train if target_exits[date] < first_prediction
        )
    else:
        resolved_train = requested_train
    if not resolved_train:
        raise ValueError("Training range is empty after target-overlap purge")

    signal_to_execution: dict[int, int] = {}
    for date in requested_predict:
        try:
            execution_date = _future_date(
                open_dates,
                index_by_date,
                date,
                definition.execution_delay,
            )
        except ValueError:
            if date not in index_by_date:
                raise
            target_index = index_by_date[date] + definition.execution_delay
            if target_index < len(open_dates):
                raise
            continue
        signal_to_execution[date] = execution_date
    if not signal_to_execution:
        _future_date(
            open_dates,
            index_by_date,
            requested_predict[0],
            definition.execution_delay,
        )
    return ResolvedExperimentTiming(
        train_dates=resolved_train,
        predict_dates=tuple(signal_to_execution),
        execution_dates=tuple(signal_to_execution.values()),
        last_train_target_exit=target_exits[resolved_train[-1]],
        signal_to_execution=signal_to_execution,
    )


def shift_score_frames_to_execution(
    scores: Mapping[str, pd.DataFrame],
    signal_to_execution: Mapping[int, int],
) -> dict[str, pd.DataFrame]:
    if not scores:
        raise ValueError("Execution score shifting requires at least one score frame")
    expected_keys: pd.DataFrame | None = None
    shifted_frames: dict[str, pd.DataFrame] = {}
    for label, frame in scores.items():
        required = {"trade_date", "ts_code", "score"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Score frame '{label}' is missing required columns: {sorted(missing)}"
            )
        keys = frame[["trade_date", "ts_code"]].reset_index(drop=True)
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise ValueError("Execution score frames must use identical signal keys")

        shifted = frame.copy(deep=True)
        shifted_dates = shifted["trade_date"].map(signal_to_execution)
        if shifted_dates.isna().any():
            missing_dates = sorted(
                int(value)
                for value in shifted.loc[shifted_dates.isna(), "trade_date"].unique()
            )
            raise ValueError(
                f"Execution mapping is missing signal dates: {missing_dates}"
            )
        shifted["trade_date"] = shifted_dates.astype("int64")
        shifted_frames[label] = shifted
    return shifted_frames


def _future_date(
    calendar: tuple[int, ...],
    index_by_date: Mapping[int, int],
    date: int,
    offset: int,
) -> int:
    try:
        index = index_by_date[int(date)]
    except KeyError as exc:
        raise ValueError(f"Trading calendar does not contain date: {date}") from exc
    target_index = index + int(offset)
    if target_index >= len(calendar):
        raise ValueError(
            f"Trading calendar has no required future open date for {date} offset={offset}"
        )
    return calendar[target_index]
