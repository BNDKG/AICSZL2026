from pathlib import Path

import pandas as pd
import pytest

from aicszl.features.registry import FeatureRegistry
from aicszl.features.store import FeatureStore
from aicszl.features.updater import FeatureUpdater
from aicszl.raw.store import RawStore


PLUGIN = "test.wide.v1"
OUTPUTS = ["test.alpha.v1", "test.beta.v1"]
DATES = [20200102, 20200103, 20200106]


def test_feature_updater_batches_continuous_dates_and_only_appends_new_dates(
    tmp_path: Path,
):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    calls: list[list[int]] = []
    registry = _registry(calls=calls)
    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=object(),
        plugin_ids=[PLUGIN],
        batch_days=2,
    )

    first = updater.update_to(20200106)
    second = updater.update_to(20200107)

    assert calls == [[20200102, 20200103], [20200106]]
    assert first[PLUGIN].last_success_trade_date == 20200106
    assert second[PLUGIN].status == "up-to-date"
    assert features.feature_available_dates(OUTPUTS, 20200101, 20200131) == DATES


def test_feature_updater_plan_is_read_only_and_uses_last_open_date(tmp_path: Path):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    calls: list[list[int]] = []
    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=_registry(calls=calls),
        calc_context=object(),
    )

    plan = updater.plan_to(20200105)

    assert calls == []
    assert [(item.start_trade_date, item.target_trade_date, item.trade_days) for item in plan] == [
        (20200102, 20200103, 2)
    ]
    assert set(features.fetch_df("SHOW TABLES")["name"]) == {
        "feature_meta",
        "feature_store_meta",
        "feature_update_state",
    }


def test_feature_updater_rejects_unknown_and_duplicate_plugin_selection(
    tmp_path: Path,
):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    registry = _registry()

    with pytest.raises(ValueError, match="Unknown feature plugin"):
        FeatureUpdater(
            raw_store=raw,
            feature_store=features,
            registry=registry,
            calc_context=object(),
            plugin_ids=["test.unknown.v1"],
        )
    with pytest.raises(ValueError, match="Duplicate feature plugin"):
        FeatureUpdater(
            raw_store=raw,
            feature_store=features,
            registry=registry,
            calc_context=object(),
            plugin_ids=[PLUGIN, PLUGIN],
        )


def test_feature_updater_rejects_incomplete_wide_output_without_partial_table(
    tmp_path: Path,
):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id=PLUGIN,
        outputs=OUTPUTS,
        inputs=[],
        lookback_days=0,
    )
    def incomplete(_ctx, dates):
        return pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": dates[0], OUTPUTS[0]: 1.0}]
        )

    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=object(),
    )

    with pytest.raises(RuntimeError, match="columns must be exactly"):
        updater.update_to(20200102)

    assert "fv_test_wide_v1" not in set(features.fetch_df("SHOW TABLES")["name"])
    assert all(features.get_state(output).status == "failed" for output in OUTPUTS)


def test_feature_updater_requires_raw_dependency_watermark(tmp_path: Path):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    registry = _registry(inputs=["raw.daily"])
    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=object(),
    )

    with pytest.raises(RuntimeError, match="dependency not ready"):
        updater.update_to(20200106)

    raw.mark_success("daily", 20200106, row_count=1)
    assert updater.update_to(20200106)[PLUGIN].status == "success"


def test_feature_updater_keeps_committed_batch_and_resumes_after_failure(
    tmp_path: Path,
):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    failures: set[int] = set()
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id=PLUGIN,
        outputs=OUTPUTS,
        inputs=[],
        lookback_days=0,
    )
    def flaky(_ctx, dates):
        date = dates[0]
        if date == 20200103 and date not in failures:
            failures.add(date)
            raise RuntimeError("temporary")
        return _values(dates)

    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=object(),
        batch_days=1,
    )

    with pytest.raises(RuntimeError, match="temporary"):
        updater.update_to(20200106)
    assert features.feature_available_dates(OUTPUTS, 20200101, 20200131) == [20200102]
    assert features.get_state(OUTPUTS[0]).last_success_trade_date == 20200102

    updater.update_to(20200106)
    assert features.feature_available_dates(OUTPUTS, 20200101, 20200131) == DATES


def test_feature_updater_resets_only_changed_plugin_and_rebuilds_from_start(
    tmp_path: Path,
):
    raw = _raw_store(tmp_path, DATES)
    features = FeatureStore(tmp_path / "features.duckdb", start_date=20200101)
    first_registry = _registry(multiplier=1.0)
    FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=first_registry,
        calc_context=object(),
    ).update_to(20200106)

    second_registry = _registry(multiplier=2.0, changed_source=True)
    FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=second_registry,
        calc_context=object(),
    ).update_to(20200106)

    rebuilt = features.load_feature_frame(OUTPUTS, 20200101, 20200131)
    assert rebuilt[OUTPUTS[0]].tolist() == [2.0, 2.0, 2.0]
    assert features.feature_available_dates(OUTPUTS, 20200101, 20200131) == DATES


def _raw_store(tmp_path: Path, dates: list[int]) -> RawStore:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "trade_cal",
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": date,
                    "is_open": 1,
                    "pretrade_date": 20191231 if index == 0 else dates[index - 1],
                }
                for index, date in enumerate(dates)
            ]
        ),
    )
    return raw


def _registry(
    *,
    calls: list[list[int]] | None = None,
    inputs: list[str] | None = None,
    multiplier: float = 1.0,
    changed_source: bool = False,
) -> FeatureRegistry:
    registry = FeatureRegistry()
    if changed_source:

        @registry.feature_plugin(
            plugin_id=PLUGIN,
            outputs=OUTPUTS,
            inputs=inputs or [],
            lookback_days=0,
        )
        def calculate_changed(_ctx, dates):
            if calls is not None:
                calls.append(list(dates))
            return _values(dates, multiplier)

    else:

        @registry.feature_plugin(
            plugin_id=PLUGIN,
            outputs=OUTPUTS,
            inputs=inputs or [],
            lookback_days=0,
        )
        def calculate(_ctx, dates):
            if calls is not None:
                calls.append(list(dates))
            return _values(dates, multiplier)

    return registry


def _values(dates: list[int], multiplier: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": date,
                OUTPUTS[0]: multiplier,
                OUTPUTS[1]: multiplier + 1.0,
            }
            for date in dates
        ],
        columns=["ts_code", "trade_date", *OUTPUTS],
    )
