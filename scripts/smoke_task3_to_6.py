from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicszl.blends import BlendInput, BlendJob, blend_predictions
from aicszl.config import load_settings
from aicszl.features import FeatureRegistry, FeatureStore
from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.models.training import TrainingJob, train_lightgbm_regressor
from aicszl.predictions import PredictionRequest, predict_from_artifact
from aicszl.raw import RawStore
from aicszl.targets.builtins import TargetCalcContext, calc_ret_5d_rank_pct


DEFAULT_DATES = "20200109,20200110,20200113,20200114"


def main() -> int:
    args = _parse_args()
    settings = load_settings(args.config)
    artifact_dir = Path(args.artifact_dir)

    raw_store = RawStore(settings.paths.raw_db, settings.project.start_date)
    feature_store = FeatureStore(settings.paths.feature_db, settings.project.start_date)
    try:
        train_dates, predict_dates = _resolve_date_sets(raw_store, args)
        dates = sorted(set(train_dates + predict_dates))
        train_start, train_end = min(train_dates), max(train_dates)
        predict_start, predict_end = min(predict_dates), max(predict_dates)
        print(
            f"dates train={train_start}-{train_end} train_days={len(train_dates)} "
            f"predict={predict_start}-{predict_end} predict_days={len(predict_dates)}"
        )

        registry = FeatureRegistry()
        register_builtin_features(registry)
        for plugin in registry.plugins():
            for meta in plugin.to_meta():
                feature_store.register_feature_meta(meta)

        feature_ctx = FeatureCalcContext(raw_store)
        for plugin in registry.plugins():
            values = plugin.func(feature_ctx, dates)
            rows = feature_store.upsert_feature_values(values)
            print(f"feature outputs={','.join(plugin.outputs)} rows={rows}")

        target_values = calc_ret_5d_rank_pct(TargetCalcContext(raw_store), dates)
        target_rows = feature_store.upsert_target_values(target_values)
        print(f"target name=target.ret_5d_rank_pct.v1 rows={target_rows}")

        job = TrainingJob(
            name=args.job_name,
            x_group="base_v1",
            features=[
                "market.close.v1",
                "market.amount.v1",
                "market.ret_5d_rank.v1",
                "limit.high_stop.v1",
                "moneyflow.net_mf_amount_rank.v1",
            ],
            target="target.ret_5d_rank_pct.v1",
            train_range=(train_start, train_end),
            filters=["market.amount.v1 > 0"],
            model_params={
                "n_estimators": args.n_estimators,
                "learning_rate": 0.1,
                "min_data_in_leaf": 1,
                "verbose": -1,
            },
        )
        model_artifact = train_lightgbm_regressor(feature_store, job, artifact_dir / "models")
        print(f"model path={model_artifact.model_path}")

        prediction = predict_from_artifact(
            feature_store,
            PredictionRequest(
                model_path=model_artifact.model_path,
                meta_path=model_artifact.meta_path,
                start_date=predict_start,
                end_date=predict_end,
            ),
            artifact_dir / "predictions",
        )
        print(f"prediction path={prediction.prediction_path} rows={prediction.rows}")

        blend = blend_predictions(
            BlendJob(
                name=args.blend_name,
                inputs=[
                    BlendInput(
                        prediction_id=prediction.prediction_id,
                        path=prediction.prediction_path,
                        weight=1.0,
                    )
                ],
            ),
            artifact_dir / "blends",
        )
        print(f"blend path={blend.blend_path} rows={blend.rows}")
    finally:
        raw_store.close()
        feature_store.close()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Task 3-6 smoke workflow.")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--dates", default=DEFAULT_DATES, help="Comma-separated trade dates for short smoke mode.")
    parser.add_argument("--train-start", type=int, help="Inclusive train range start date, e.g. 20200101.")
    parser.add_argument("--train-end", type=int, help="Inclusive train range end date, e.g. 20210101.")
    parser.add_argument("--predict-start", type=int, help="Inclusive prediction range start date, e.g. 20210101.")
    parser.add_argument("--predict-end", type=int, help="Inclusive prediction range end date, e.g. 20220101.")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--job-name", default="lgb_rank5_smoke_v1")
    parser.add_argument("--blend-name", default="blend_smoke_v1")
    parser.add_argument("--n-estimators", type=int, default=5)
    return parser.parse_args()


def _resolve_date_sets(raw_store: RawStore, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    range_values = [args.train_start, args.train_end, args.predict_start, args.predict_end]
    if any(value is not None for value in range_values):
        if any(value is None for value in range_values):
            raise ValueError(
                "Range mode requires --train-start, --train-end, --predict-start, and --predict-end"
            )
        train_dates = _trade_dates(raw_store, args.train_start, args.train_end)
        predict_dates = _trade_dates(raw_store, args.predict_start, args.predict_end)
        if not train_dates:
            raise ValueError("No open trade dates found in train range")
        if not predict_dates:
            raise ValueError("No open trade dates found in prediction range")
        return train_dates, predict_dates

    dates = _parse_dates(args.dates)
    return dates, dates


def _trade_dates(raw_store: RawStore, start_date: int, end_date: int) -> list[int]:
    rows = raw_store.fetch_df(
        """
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date BETWEEN ? AND ?
          AND is_open = 1
        ORDER BY cal_date
        """,
        [int(start_date), int(end_date)],
    )
    return [int(value) for value in rows["cal_date"].tolist()]


def _parse_dates(raw: str) -> list[int]:
    dates = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not dates:
        raise ValueError("--dates must contain at least one date")
    return dates


if __name__ == "__main__":
    raise SystemExit(main())
