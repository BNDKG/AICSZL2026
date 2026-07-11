from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BacktestRunArtifact, BacktestRunSettings
from .dataset import BacktestDatasetArtifact


LIMIT_PRICE_ATOL = 1e-6


def export_qlib_provider(
    scores: pd.DataFrame,
    adj_factors: pd.DataFrame,
    target_dir: str | Path,
) -> Path:
    required = {"trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount", "is_tradable", "limit_up", "limit_down"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Score dataset is missing Qlib fields: {', '.join(missing)}")
    merged = scores.merge(adj_factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if merged["adj_factor"].isna().any():
        raise ValueError("Qlib export requires adj_factor for every score row")
    root = Path(target_dir)
    calendar = sorted(pd.to_datetime(merged["trade_date"].astype(str), format="%Y%m%d").unique())
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text("\n".join(pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar) + "\n", encoding="utf-8")
    ranges = []
    for code, frame in merged.groupby("ts_code", sort=True):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
        frame = frame.set_index("date").reindex(calendar)
        is_tradable = frame["is_tradable"].astype("boolean").fillna(False)
        frame.loc[~is_tradable, "close"] = np.nan
        directory = root / "features" / str(code).lower()
        directory.mkdir(parents=True, exist_ok=True)
        start = float(calendar.index(frame.index.min()))
        frame["limit_buy"] = np.isclose(
            frame["open"], frame["limit_up"], rtol=0.0, atol=LIMIT_PRICE_ATOL
        ).astype(float)
        frame["limit_sell"] = np.isclose(
            frame["open"], frame["limit_down"], rtol=0.0, atol=LIMIT_PRICE_ATOL
        ).astype(float)
        fields = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "vol", "amount": "amount", "factor": "adj_factor", "limit_buy": "limit_buy", "limit_sell": "limit_sell"}
        for name, source in fields.items():
            np.hstack(([start], frame[source].to_numpy(dtype="float32"))).astype("<f4").tofile(directory / f"{name}.day.bin")
        ranges.append(f"{code}\t{frame.index.min():%Y-%m-%d}\t{frame.index.max():%Y-%m-%d}")
    (root / "instruments" / "all.txt").write_text("\n".join(ranges) + "\n", encoding="utf-8")
    return root


def run_qlib_topk_backtest(provider_uri: str | Path, scores: pd.DataFrame, *, topk: int, n_drop: int, initial_cash: float):
    import qlib
    from qlib.backtest import backtest, executor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    qlib.init(provider_uri=str(provider_uri), region="cn", clear_mem_cache=True)
    signal = scores.copy()
    signal["datetime"] = pd.to_datetime(signal["trade_date"].astype(str), format="%Y%m%d")
    dates = sorted(signal["datetime"].unique())
    if len(dates) < 3:
        raise ValueError("Qlib daily backtest requires at least three trade dates")
    prediction = signal.rename(columns={"ts_code": "instrument"}).set_index(["datetime", "instrument"])["score"]
    strategy = TopkDropoutStrategy(signal=prediction, topk=topk, n_drop=n_drop, only_tradable=False, forbid_all_trade_at_limit=False)
    runner = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    # Qlib 0.9.7 turns benchmark=None into its CSI300 default; an empty Series preserves
    # the project's no-benchmark POC contract without requiring external index data.
    return backtest(str(dates[0].date()), str(dates[-2].date()), strategy, runner, benchmark=pd.Series(dtype=float), account=initial_cash, exchange_kwargs={"freq": "day", "deal_price": "open", "limit_threshold": ("$limit_buy", "$limit_sell")})


class QlibBacktestAdapter:
    def __init__(self, raw_store) -> None:
        self.raw_store = raw_store

    def run(self, dataset: BacktestDatasetArtifact, settings: BacktestRunSettings) -> BacktestRunArtifact:
        scores = pd.read_pickle(dataset.dataset_path)
        output_dir = dataset.dataset_path.parent / dataset.dataset_id
        try:
            factors = self.raw_store.fetch_df(
                "SELECT ts_code, trade_date, adj_factor FROM adj_factor"
            )
            provider = export_qlib_provider(scores, factors, output_dir / "provider")
        except Exception as exc:
            raise RuntimeError(
                f"Qlib provider export failed for dataset {dataset.dataset_id}: {exc}"
            ) from exc
        try:
            metrics, _ = run_qlib_topk_backtest(
                provider,
                scores,
                topk=settings.topk,
                n_drop=settings.n_drop,
                initial_cash=settings.initial_cash,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Qlib backtest requires pyqlib==0.9.7; install project dependencies "
                "with `python -m pip install -e .`"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Qlib backtest failed for dataset {dataset.dataset_id}: {exc}"
            ) from exc
        report, positions = metrics["1day"]
        report_path = output_dir / "report.pkl"
        positions_path = output_dir / "positions.pkl"
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_pickle(report_path)
        pd.to_pickle(positions, positions_path)
        return BacktestRunArtifact(engine="qlib", report_path=report_path, positions_path=positions_path)
