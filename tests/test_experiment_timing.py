from __future__ import annotations

import pandas as pd
import pytest

from aicszl.experiments.timing import (
    resolve_experiment_timing,
    shift_score_frames_to_execution,
)
from aicszl.targets import EXECUTABLE_OPEN_5D_TARGET, get_target_definition


CALENDAR = [
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
]


def test_resolve_experiment_timing_purges_overlapping_labels_and_maps_next_open():
    timing = resolve_experiment_timing(
        calendar=CALENDAR,
        train_dates=CALENDAR[:7],
        predict_dates=[20240102, 20240103, 20240104],
        definition=get_target_definition(EXECUTABLE_OPEN_5D_TARGET),
    )

    assert timing.train_dates == (20231221,)
    assert timing.predict_dates == (20240102, 20240103, 20240104)
    assert timing.last_train_target_exit == 20231229
    assert timing.execution_dates == (20240103, 20240104, 20240105)
    assert timing.signal_to_execution == {
        20240102: 20240103,
        20240103: 20240104,
        20240104: 20240105,
    }


def test_shift_score_frames_uses_identical_execution_keys_without_mutating_inputs():
    keys = pd.DataFrame(
        {
            "trade_date": [20240102, 20240102, 20240103, 20240103],
            "ts_code": ["A", "B", "A", "B"],
            "score": [0.1, 0.2, 0.3, 0.4],
        }
    )
    frames = {
        "random_baseline": keys,
        "5_features": keys.assign(score=[0.4, 0.3, 0.2, 0.1]),
    }
    originals = {name: frame.copy(deep=True) for name, frame in frames.items()}

    shifted = shift_score_frames_to_execution(
        frames,
        {20240102: 20240103, 20240103: 20240104},
    )

    for frame in shifted.values():
        assert frame["trade_date"].tolist() == [20240103, 20240103, 20240104, 20240104]
        assert frame[["trade_date", "ts_code"]].equals(
            shifted["random_baseline"][["trade_date", "ts_code"]]
        )
    for name, original in originals.items():
        pd.testing.assert_frame_equal(frames[name], original)


def test_shift_score_frames_rejects_mismatched_keys_and_missing_date_mapping():
    base = pd.DataFrame(
        {
            "trade_date": [20240102, 20240102],
            "ts_code": ["A", "B"],
            "score": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="identical signal keys"):
        shift_score_frames_to_execution(
            {"one": base, "two": base.iloc[:1].copy()},
            {20240102: 20240103},
        )
    with pytest.raises(ValueError, match="missing signal dates"):
        shift_score_frames_to_execution({"one": base}, {})


def test_resolve_experiment_timing_rejects_calendar_without_required_future_date():
    with pytest.raises(ValueError, match="required future open date"):
        resolve_experiment_timing(
            calendar=CALENDAR[:9],
            train_dates=[20231221],
            predict_dates=[20240103],
            definition=get_target_definition(EXECUTABLE_OPEN_5D_TARGET),
        )


def test_resolve_experiment_timing_trims_trailing_signal_without_next_open():
    timing = resolve_experiment_timing(
        calendar=CALENDAR[:10],
        train_dates=[20231221],
        predict_dates=[20240103, 20240104],
        definition=get_target_definition(EXECUTABLE_OPEN_5D_TARGET),
    )

    assert timing.predict_dates == (20240103,)
    assert timing.execution_dates == (20240104,)
    assert timing.signal_to_execution == {20240103: 20240104}
