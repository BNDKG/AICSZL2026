from pathlib import Path

import pytest

from aicszl.config import load_feature_groups


BASE = [
    "market.close.v1",
    "market.amount.v1",
    "market.ret_5d_rank.v1",
    "limit.high_stop.v1",
    "moneyflow.net_mf_amount_rank.v1",
]
EXPERIMENT = [
    "market.ret_20d_rank.v1",
    "market.reversal_1d_rank.v1",
    "risk.volatility_20d_rank.v1",
    "liquidity.amount_ratio_5d_rank.v1",
    "market.close_position_20d_rank.v1",
]


def test_project_feature_groups_are_stable_and_ordered():
    groups = load_feature_groups("configs/features.yaml")

    assert list(groups) == ["base_v1", "price_volume_exp_v1", "base_plus_price_volume_v1"]
    assert groups["base_v1"].features == BASE
    assert groups["price_volume_exp_v1"].features == EXPERIMENT
    assert groups["base_plus_price_volume_v1"].features == BASE + EXPERIMENT


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("feature_groups:\n  empty:\n    features: []\n", "must contain at least one feature"),
        (
            "feature_groups:\n  duplicate:\n    features:\n"
            "      - market.close.v1\n      - market.close.v1\n",
            "duplicate feature",
        ),
        (
            "feature_groups:\n  invalid:\n    features:\n      - close\n",
            "domain.name.version",
        ),
    ],
)
def test_feature_group_loader_rejects_invalid_groups(
    tmp_path: Path,
    yaml_text: str,
    message: str,
):
    path = tmp_path / "features.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_feature_groups(path)
