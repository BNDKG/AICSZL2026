from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicszl.backtests import BacktestRunSettings, build_score_dataset, save_equity_curve
from aicszl.backtests.qlib_adapter import export_qlib_provider, run_qlib_topk_backtest
from aicszl.config import load_settings
from aicszl.raw import RawStore


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run model and random Qlib backtests, then automatically save "
            "equity_curve.png under --output-dir."
        )
    )
    parser.add_argument("--blend-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def _metrics(report: pd.DataFrame) -> dict[str, object]:
    equity = report["account"] / float(report["account"].iloc[0])
    return {
        "report_rows": int(len(report)),
        "start": str(report.index.min().date()),
        "end": str(report.index.max().date()),
        "net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "mean_turnover": float(report["turnover"].mean()),
        "total_cost": float(report["total_cost"].iloc[-1]),
        "nan_cells": int(report.isna().sum().sum()),
    }


def _save_result(output_dir: Path, report: pd.DataFrame, positions: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_pickle(output_dir / "report.pkl")
    pd.to_pickle(positions, output_dir / "positions.pkl")


def _publish_artifacts(
    staging_dir: Path, output_dir: Path, relative_paths: list[Path]
) -> None:
    for relative_path in relative_paths:
        source = staging_dir / relative_path
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = BacktestRunSettings(
        topk=args.topk,
        n_drop=args.n_drop,
        initial_cash=args.initial_cash,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion.json"
    completion_path.unlink(missing_ok=True)
    app_settings = load_settings(args.config)
    raw_store = RawStore(
        app_settings.paths.raw_db,
        app_settings.project.start_date,
    )
    try:
        dataset = build_score_dataset(raw_store, args.blend_path, output_dir)
        scores = pd.read_pickle(dataset.dataset_path).sort_values(
            ["trade_date", "ts_code"]
        ).reset_index(drop=True)
        factors = raw_store.fetch_df(
            """
            SELECT ts_code, trade_date, adj_factor
            FROM adj_factor
            WHERE trade_date BETWEEN ? AND ?
            """,
            [int(scores["trade_date"].min()), int(scores["trade_date"].max())],
        )
        provider = export_qlib_provider(
            scores, factors, output_dir / "shared_provider"
        )
    finally:
        raw_store.close()

    model_metrics, _ = run_qlib_topk_backtest(
        provider,
        scores,
        topk=settings.topk,
        n_drop=settings.n_drop,
        initial_cash=settings.initial_cash,
    )
    model_report, model_positions = model_metrics["1day"]

    random_scores = scores[["trade_date", "ts_code", "score"]].copy()
    random_scores["score"] = np.random.default_rng(args.random_seed).random(
        len(random_scores)
    )
    random_metrics, _ = run_qlib_topk_backtest(
        provider,
        random_scores,
        topk=settings.topk,
        n_drop=settings.n_drop,
        initial_cash=settings.initial_cash,
    )
    random_report, random_positions = random_metrics["1day"]
    random_name = f"random_seed_{args.random_seed}"

    if not model_report.index.equals(random_report.index):
        raise RuntimeError("Model and random reports do not have identical dates")
    if model_report.isna().any().any() or random_report.isna().any().any():
        raise RuntimeError("Backtest reports contain NaN values")
    if len(model_positions) != len(model_report) or len(random_positions) != len(
        random_report
    ):
        raise RuntimeError("Position dates do not match report dates")

    summary = {
        "contract": {
            "score_rows": int(len(scores)),
            "score_dates": int(scores["trade_date"].nunique()),
            "score_range": [
                int(scores["trade_date"].min()),
                int(scores["trade_date"].max()),
            ],
            "topk": settings.topk,
            "n_drop": settings.n_drop,
            "initial_cash": settings.initial_cash,
            "random_seed": args.random_seed,
            "dataset_path": str(dataset.dataset_path.resolve()),
            "provider_path": str(provider.resolve()),
        },
        "model": _metrics(model_report),
        random_name: _metrics(random_report),
    }
    artifact_paths = [
        Path("model/report.pkl"),
        Path("model/positions.pkl"),
        Path(f"{random_name}/report.pkl"),
        Path(f"{random_name}/positions.pkl"),
        Path("metrics.json"),
        Path("equity_curve.png"),
    ]
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=output_dir) as temp:
        staging_dir = Path(temp)
        _save_result(staging_dir / "model", model_report, model_positions)
        _save_result(
            staging_dir / random_name, random_report, random_positions
        )
        staging_dir.joinpath("metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_equity_curve(
            {
                f"LightGBM TopK={settings.topk}": model_report,
                f"Random baseline (seed {args.random_seed})": random_report,
            },
            staging_dir / "equity_curve.png",
            title="Model vs random baseline net equity",
        )
        staging_dir.joinpath("completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "artifacts": [str(path.as_posix()) for path in artifact_paths],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _publish_artifacts(staging_dir, output_dir, artifact_paths)
        _publish_artifacts(
            staging_dir, output_dir, [Path("completion.json")]
        )

    labels = [
        "model report",
        "model positions",
        "random report",
        "random positions",
        "metrics",
        "equity curve",
    ]
    for label, relative_path in zip(labels, artifact_paths, strict=True):
        print(f"{label} path={(output_dir / relative_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
