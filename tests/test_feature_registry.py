import pandas as pd
import pytest

from aicszl.features.registry import FeatureRegistry


def test_feature_plugin_decorator_registers_outputs_and_metadata():
    registry = FeatureRegistry()

    @registry.feature_plugin(
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

    assert plugin.func is calc_close
    assert plugin.outputs == ["market.close.v1"]
    assert plugin.inputs == ["raw.daily"]
    assert plugin.lookback_days == 0
    assert plugin.code_hash
    assert plugin.to_meta()[0].feature_name == "market.close.v1"
    assert plugin.to_meta()[0].domain == "market"
    assert plugin.to_meta()[0].version == "v1"
    assert plugin.to_meta()[0].owner_plugin == "calc_close"
    assert plugin.to_meta()[0].input_tables == ["raw.daily"]
    assert plugin.to_meta()[0].description == "Daily close price"


def test_feature_plugin_rejects_duplicate_feature_name():
    registry = FeatureRegistry()

    @registry.feature_plugin(outputs=["market.close.v1"], inputs=["raw.daily"], lookback_days=0)
    def first(ctx, dates):
        return pd.DataFrame()

    with pytest.raises(ValueError, match="already registered"):

        @registry.feature_plugin(outputs=["market.close.v1"], inputs=["raw.daily"], lookback_days=0)
        def second(ctx, dates):
            return pd.DataFrame()


def test_feature_plugin_requires_domain_name_version_shape():
    registry = FeatureRegistry()

    with pytest.raises(ValueError, match="domain.name.version"):

        @registry.feature_plugin(outputs=["close"], inputs=["raw.daily"], lookback_days=0)
        def bad_name(ctx, dates):
            return pd.DataFrame()
