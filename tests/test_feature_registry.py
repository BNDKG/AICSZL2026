import hashlib

import pandas as pd
import pytest

import aicszl.features.registry as registry_module
import aicszl.features.history as history_module
from aicszl.features.builtins import register_builtin_features
from aicszl.features.registry import FeatureRegistry


_HASH_SCALE = 1


def _hash_helper(value):
    return value * _HASH_SCALE


def _hash_plugin(_ctx, dates):
    return _hash_helper(len(dates))


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

    assert plugin.code_hash == registry_module._code_hash(plugin.func)
    assert plugin.code_hash != hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_code_hash_changes_when_referenced_helper_source_changes(monkeypatch):
    original_getsource = registry_module.inspect.getsource
    helper_source = {"value": "def _hash_helper(value):\n    return value\n"}

    def fake_getsource(value):
        if value is _hash_helper:
            return helper_source["value"]
        return original_getsource(value)

    monkeypatch.setattr(registry_module.inspect, "getsource", fake_getsource)
    before = registry_module._code_hash(_hash_plugin)
    helper_source["value"] = "def _hash_helper(value):\n    return value + 1\n"

    assert registry_module._code_hash(_hash_plugin) != before


def test_code_hash_changes_when_referenced_module_constant_changes(monkeypatch):
    before = registry_module._code_hash(_hash_plugin)

    monkeypatch.setattr(__import__(__name__, fromlist=["_HASH_SCALE"]), "_HASH_SCALE", 2)

    assert registry_module._code_hash(_hash_plugin) != before


def test_builtin_code_hash_covers_shared_project_helper_source(monkeypatch):
    registry = FeatureRegistry()
    register_builtin_features(registry)
    plugin = registry.get_plugin("market.ret_5d_rank.v1")
    original_getsource = registry_module.inspect.getsource
    helper_source = {"value": original_getsource(history_module.fetch_bounded_history)}

    def fake_getsource(value):
        if value is history_module.fetch_bounded_history:
            return helper_source["value"]
        return original_getsource(value)

    monkeypatch.setattr(registry_module.inspect, "getsource", fake_getsource)
    before = registry_module._code_hash(plugin.func)
    helper_source["value"] += "\n# changed bounded history behavior\n"

    assert registry_module._code_hash(plugin.func) != before
