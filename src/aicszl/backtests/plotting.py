from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_equity_curve(
    reports: Mapping[str, pd.DataFrame],
    output_path: str | Path,
    *,
    title: str = "Backtest net equity",
) -> Path:
    if not reports:
        raise ValueError("Equity curve requires at least one report")

    normalized: dict[str, pd.Series] = {}
    expected_index: pd.Index | None = None
    for label, report in reports.items():
        if report.empty:
            raise ValueError(f"Report '{label}' must not be empty")
        if "account" not in report.columns:
            raise ValueError(f"Report '{label}' is missing required column: account")
        if expected_index is None:
            expected_index = report.index
        elif not report.index.equals(expected_index):
            raise ValueError("Equity curve reports must have identical indexes")
        try:
            account = pd.to_numeric(report["account"], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Report '{label}' account values must be finite positive numbers"
            ) from exc
        initial_account = float(account.iloc[0])
        if not np.isfinite(account.to_numpy()).all() or initial_account <= 0:
            raise ValueError(
                f"Report '{label}' account values must be finite positive numbers"
            )
        normalized[label] = account / initial_account

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 6), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    try:
        for label, equity in normalized.items():
            line = axes.plot(equity.index, equity, linewidth=2.0, label=label)[0]
            axes.annotate(
                f"{equity.iloc[-1]:.3f}",
                (equity.index[-1], equity.iloc[-1]),
                xytext=(-42, 10),
                textcoords="offset points",
                color=line.get_color(),
            )
        axes.axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
        axes.set_title(title)
        axes.set_xlabel("Date")
        axes.set_ylabel("Net equity (initial = 1.0, after costs)")
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best")
        axes.margins(x=0.01)
        figure.savefig(destination, format="png")
    finally:
        figure.clear()
    return destination
