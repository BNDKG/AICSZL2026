import hashlib

import pandas as pd
import pytest

import aicszl.features.registry as registry_module
from aicszl.features.builtins import register_builtin_features
from aicszl.features.registry import FeatureRegistry


def test_feature_plugin_decorator_registers_outputs_and_metadata():
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id="market.raw_fields.v1",
        outputs=["market.close.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
        kind="raw_field",
        description="Daily close price",
    )
    def calc_close(ctx, dates):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": dates[0],
                    "feature_name": "market.close.v1",
                    "value": 10.5,
                }
            ]
        )

    plugin = registry.get("market.close.v1")
    assert registry.get_plugin("market.raw_fields.v1") is plugin

    assert plugin.func is calc_close
    assert plugin.plugin_id == "market.raw_fields.v1"
    assert plugin.outputs == ["market.close.v1"]
    assert plugin.inputs == ["raw.daily"]
    assert plugin.lookback_days == 0
    assert plugin.code_hash
    assert plugin.to_meta()[0].feature_name == "market.close.v1"
    assert plugin.to_meta()[0].domain == "market"
    assert plugin.to_meta()[0].version == "v1"
    assert plugin.to_meta()[0].owner_plugin == "market.raw_fields.v1"
    assert plugin.to_meta()[0].input_tables == ["raw.daily"]
    assert plugin.to_meta()[0].description == "Daily close price"


def test_feature_plugin_rejects_duplicate_feature_name():
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id="market.raw_fields.v1",
        outputs=["market.close.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
    )
    def first(ctx, dates):
        return pd.DataFrame()

    with pytest.raises(ValueError, match="already registered"):

        @registry.feature_plugin(
            plugin_id="market.other.v1",
            outputs=["market.close.v1"],
            inputs=["raw.daily"],
            lookback_days=0,
        )
        def second(ctx, dates):
            return pd.DataFrame()


def test_feature_plugin_rejects_duplicate_plugin_id():
    registry = FeatureRegistry()

    @registry.feature_plugin(
        plugin_id="market.raw_fields.v1",
        outputs=["market.close.v1"],
        inputs=["raw.daily"],
        lookback_days=0,
    )
    def first(ctx, dates):
        return pd.DataFrame()

    with pytest.raises(ValueError, match="already registered"):

        @registry.feature_plugin(
            plugin_id="market.raw_fields.v1",
            outputs=["market.amount.v1"],
            inputs=["raw.daily"],
            lookback_days=0,
        )
        def second(ctx, dates):
            return pd.DataFrame()


def test_feature_plugin_requires_domain_name_version_shape():
    registry = FeatureRegistry()

    with pytest.raises(ValueError, match="domain.name.version"):

        @registry.feature_plugin(
            plugin_id="market.bad.v1",
            outputs=["close"],
            inputs=["raw.daily"],
            lookback_days=0,
        )
        def bad_name(ctx, dates):
            return pd.DataFrame()


def test_feature_plugin_requires_domain_name_version_plugin_id():
    registry = FeatureRegistry()

    with pytest.raises(ValueError, match="plugin ID"):

        @registry.feature_plugin(
            plugin_id="raw_fields",
            outputs=["market.close.v1"],
            inputs=["raw.daily"],
            lookback_days=0,
        )
        def bad_plugin_id(ctx, dates):
            return pd.DataFrame()


def test_code_hash_matches_golden_digest(monkeypatch):
    source = "def calculation(ctx, dates):\n    return dates\n"

    def calculation(_ctx, _dates):
        return pd.DataFrame()

    monkeypatch.setattr(registry_module.inspect, "getsource", lambda _value: source)

    assert registry_module._code_hash(calculation) == (
        "bbaaf25b28c7d7ecce3ed03739137e7bbcfcf919d073e3f0c66f378284e9fcce"
    )


def test_builtin_hash_uses_function_source_formula():
    registry = FeatureRegistry()
    register_builtin_features(registry)
    plugin = registry.get_plugin("market.raw_fields.v1")
    source = registry_module.inspect.getsource(plugin.func)

    assert plugin.code_hash == hashlib.sha256(source.encode("utf-8")).hexdigest()
