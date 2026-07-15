from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from aicszl.features.registry import FeatureRegistry
from aicszl.features.store import FeatureStore
from aicszl.features.updater import FeatureUpdater
from aicszl.raw.store import RawStore


OPEN_DATES = [20200102, 20200103, 20200106, 20200107, 20200108]


@dataclass
class CalcContext:
    calls: dict[str, list[list[int]]]
    fail_from: int | None = None


def test_feature_updater_selects_one_plugin_and_batches_continuous_dates(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1", "test.b.v1"])
    _register_pack(registry, "test.pack_b.v1", ["test.unselected.v1"])

    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
        plugin_ids=["test.pack_a.v1"],
        batch_days=2,
    )
    summary = updater.update_to(20200108)

    assert list(summary) == ["test.pack_a.v1"]
    assert context.calls["test.pack_a.v1"] == [
        [20200102, 20200103],
        [20200106, 20200107],
        [20200108],
    ]
    assert features.get_state("test.a.v1").last_success_trade_date == 20200108
    assert features.get_state("test.b.v1").last_success_trade_date == 20200108
    assert features.get_state("test.unselected.v1").status == "pending"
    assert set(features.fetch_df("SELECT DISTINCT feature_name FROM feature_values")["feature_name"]) == {
        "test.a.v1",
        "test.b.v1",
    }


def test_feature_updater_defaults_to_all_plugins_and_only_processes_new_dates(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])
    _register_pack(registry, "test.pack_b.v1", ["test.b.v1"])

    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
        batch_days=10,
    )
    updater.update_to(20200107)
    updater.update_to(20200108)

    assert context.calls == {
        "test.pack_a.v1": [[20200102, 20200103, 20200106, 20200107], [20200108]],
        "test.pack_b.v1": [[20200102, 20200103, 20200106, 20200107], [20200108]],
    }


def test_feature_updater_uses_last_open_date_for_weekend_target(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])

    summary = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
    ).update_to(20200105)

    assert context.calls["test.pack_a.v1"] == [[20200102, 20200103]]
    assert summary["test.pack_a.v1"].last_success_trade_date == 20200103


def test_feature_updater_plan_does_not_calculate_or_write(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])

    plans = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
        plugin_ids=["test.pack_a.v1"],
    ).plan_to(20200108)

    assert [(plan.plugin_id, plan.start_trade_date, plan.target_trade_date, plan.trade_days) for plan in plans] == [
        ("test.pack_a.v1", 20200102, 20200108, 5)
    ]
    assert context.calls == {}
    assert features.fetch_df("SELECT count(*) AS n FROM feature_values").iloc[0]["n"] == 0
    assert features.fetch_df("SELECT count(*) AS n FROM feature_meta").iloc[0]["n"] == 0
    assert features.fetch_df("SELECT count(*) AS n FROM feature_update_state").iloc[0]["n"] == 0


def test_feature_updater_rejects_unknown_and_duplicate_plugin_selection(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])

    with pytest.raises(ValueError, match="Unknown feature plugin"):
        FeatureUpdater(
            raw_store=raw,
            feature_store=features,
            registry=registry,
            calc_context=CalcContext(calls={}),
            plugin_ids=["test.missing.v1"],
        )

    with pytest.raises(ValueError, match="Duplicate feature plugin"):
        FeatureUpdater(
            raw_store=raw,
            feature_store=features,
            registry=registry,
            calc_context=CalcContext(calls={}),
            plugin_ids=["test.pack_a.v1", "test.pack_a.v1"],
        )


def test_feature_updater_rolls_back_plugin_when_an_output_is_missing(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id="test.partial.v1",
        outputs=["test.a.v1", "test.b.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
    )
    def partial(ctx: CalcContext, dates: list[int]) -> pd.DataFrame:
        return _values("test.partial.v1", ["test.a.v1"], dates, ctx)

    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=CalcContext(calls={}),
    )

    with pytest.raises(RuntimeError, match="missing declared outputs"):
        updater.update_to(20200103)

    assert features.fetch_df("SELECT count(*) AS n FROM feature_values").iloc[0]["n"] == 0
    assert features.get_state("test.a.v1").status == "failed"
    assert features.get_state("test.b.v1").status == "failed"
    assert features.get_state("test.a.v1").last_success_trade_date is None


def test_feature_updater_requires_raw_dependency_watermark(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES, daily_watermark=20200103)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])

    with pytest.raises(RuntimeError, match=r"dependency not ready.*actual=20200103.*required=20200108"):
        FeatureUpdater(
            raw_store=raw,
            feature_store=features,
            registry=registry,
            calc_context=context,
        ).update_to(20200108)

    assert context.calls == {}
    assert features.get_state("test.a.v1").status == "pending"


def test_feature_updater_keeps_committed_batch_and_resumes_after_failure(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={}, fail_from=20200106)
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1"])
    updater = FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
        batch_days=2,
    )

    with pytest.raises(RuntimeError, match="plugin=test.pack_a.v1"):
        updater.update_to(20200108)

    assert features.get_state("test.a.v1").last_success_trade_date == 20200103
    assert features.fetch_df("SELECT DISTINCT trade_date FROM feature_values ORDER BY trade_date")["trade_date"].tolist() == [
        20200102,
        20200103,
    ]

    context.fail_from = None
    context.calls.clear()
    updater.update_to(20200108)
    assert context.calls["test.pack_a.v1"] == [[20200106, 20200107], [20200108]]


def test_feature_updater_bootstraps_only_verified_common_prefix(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1", "test.b.v1"])
    features.upsert_feature_values(
        _static_values(["test.a.v1", "test.b.v1"], [20200102, 20200103, 20200106])
    )

    FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
    ).update_to(20200108)

    assert context.calls["test.pack_a.v1"] == [[20200107, 20200108]]


def test_feature_updater_repairs_from_first_legacy_gap(tmp_path: Path):
    raw, features = _stores(tmp_path, OPEN_DATES)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.pack_a.v1", ["test.a.v1", "test.b.v1"])
    existing = pd.concat(
        [
            _static_values(["test.a.v1"], [20200102, 20200103, 20200106]),
            _static_values(["test.b.v1"], [20200102, 20200106]),
        ],
        ignore_index=True,
    )
    features.upsert_feature_values(existing)

    FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
    ).update_to(20200108)

    assert context.calls["test.pack_a.v1"] == [[20200103, 20200106, 20200107, 20200108]]


def test_feature_updater_accepts_legacy_lookback_warmup(tmp_path: Path):
    dates = [20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110]
    raw, features = _stores(tmp_path, dates, daily_watermark=20200110)
    registry = FeatureRegistry()
    context = CalcContext(calls={})
    _register_pack(registry, "test.lookback.v1", ["test.rank.v1"], lookback_days=5)
    features.upsert_feature_values(_static_values(["test.rank.v1"], [20200109]))

    FeatureUpdater(
        raw_store=raw,
        feature_store=features,
        registry=registry,
        calc_context=context,
    ).update_to(20200110)

    assert context.calls["test.lookback.v1"] == [[20200110]]
    assert features.get_state("test.rank.v1").last_success_trade_date == 20200110


def _stores(
    tmp_path: Path,
    open_dates: list[int],
    daily_watermark: int | None = None,
) -> tuple[RawStore, FeatureStore]:
    raw = RawStore(tmp_path / "raw.duckdb", start_date=20200101)
    raw.upsert(
        "trade_cal",
        pd.DataFrame(
            [
                {
                    "cal_date": date,
                    "exchange": "SSE",
                    "is_open": 1,
                    "pretrade_date": open_dates[index - 1] if index else 20191231,
                }
                for index, date in enumerate(open_dates)
            ]
        ),
    )
    raw.mark_success("daily", daily_watermark or open_dates[-1], row_count=1)
    return raw, FeatureStore(tmp_path / "features.duckdb", start_date=20200101)


def _register_pack(
    registry: FeatureRegistry,
    plugin_id: str,
    outputs: list[str],
    lookback_days: int = 0,
) -> None:
    @registry.feature_plugin(
        plugin_id=plugin_id,
        outputs=outputs,
        inputs=["raw.daily"],
        lookback_days=lookback_days,
    )
    def calculate(ctx: CalcContext, dates: list[int]) -> pd.DataFrame:
        if ctx.fail_from is not None and dates[0] >= ctx.fail_from:
            raise ValueError("planned failure")
        return _values(plugin_id, outputs, dates, ctx)


def _values(
    plugin_id: str,
    outputs: list[str],
    dates: list[int],
    context: CalcContext,
) -> pd.DataFrame:
    context.calls.setdefault(plugin_id, []).append(list(dates))
    return _static_values(outputs, dates)


def _static_values(outputs: list[str], dates: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": date,
                "feature_name": output,
                "value": float(date),
            }
            for date in dates
            for output in outputs
        ],
        columns=["ts_code", "trade_date", "feature_name", "value"],
    )
