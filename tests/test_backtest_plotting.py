from pathlib import Path

import pandas as pd
import pytest

from aicszl.backtests.plotting import save_equity_curve


def test_save_equity_curve_writes_non_empty_png(tmp_path: Path):
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    reports = {
        "LightGBM TopK=50": pd.DataFrame(
            {"account": [100_000.0, 105_000.0, 103_000.0]}, index=index
        ),
        "Random baseline (seed 42)": pd.DataFrame(
            {"account": [100_000.0, 97_000.0, 98_000.0]}, index=index
        ),
    }
    output_path = tmp_path / "nested" / "equity_curve.png"

    result = save_equity_curve(reports, output_path, title="Validation equity")

    assert result == output_path.resolve()
    assert result.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.stat().st_size > 1_000


def test_save_equity_curve_rejects_empty_report_mapping(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one report"):
        save_equity_curve({}, tmp_path / "equity_curve.png")


def test_save_equity_curve_rejects_missing_account_column(tmp_path: Path):
    report = pd.DataFrame(
        {"return": [0.1]}, index=pd.to_datetime(["2024-01-02"])
    )

    with pytest.raises(ValueError, match="account"):
        save_equity_curve({"model": report}, tmp_path / "equity_curve.png")


def test_save_equity_curve_rejects_empty_report(tmp_path: Path):
    report = pd.DataFrame(columns=["account"])

    with pytest.raises(ValueError, match="must not be empty"):
        save_equity_curve({"model": report}, tmp_path / "equity_curve.png")


def test_save_equity_curve_rejects_mismatched_report_dates(tmp_path: Path):
    model = pd.DataFrame(
        {"account": [100_000.0]}, index=pd.to_datetime(["2024-01-02"])
    )
    random = pd.DataFrame(
        {"account": [100_000.0]}, index=pd.to_datetime(["2024-01-03"])
    )

    with pytest.raises(ValueError, match="identical indexes"):
        save_equity_curve(
            {"model": model, "random": random}, tmp_path / "equity_curve.png"
        )


@pytest.mark.parametrize(
    "account",
    [
        [float("nan"), 100_000.0],
        [100_000.0, float("inf")],
        [0.0, 100_000.0],
        [-100_000.0, -90_000.0],
        ["not-a-number", 100_000.0],
    ],
)
def test_save_equity_curve_rejects_invalid_account_values(
    tmp_path: Path, account: list[object]
):
    report = pd.DataFrame(
        {"account": account},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    with pytest.raises(ValueError, match="finite positive"):
        save_equity_curve({"model": report}, tmp_path / "equity_curve.png")
