from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import yaml

from aicszl.artifact_cache import (
    MODEL_CACHE_SCHEMA_VERSION,
    PREDICTION_CACHE_SCHEMA_VERSION,
    get_or_predict_cached,
    get_or_train_cached_model,
)
from aicszl.backtests import build_score_dataset, save_equity_curve
from aicszl.backtests.qlib_adapter import export_qlib_provider, run_qlib_topk_backtest
from aicszl.config import load_feature_groups, load_settings
from aicszl.features import FeatureRegistry, FeatureStore, FeatureUpdater
from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.models.training import TrainingJob
from aicszl.predictions import PredictionRequest, predict_from_artifact
from aicszl.raw import RawStore
from aicszl.targets import TargetCalcContext, calculate_target, get_target_definition

from .comparison import build_common_scores, summarize_reports
from .config import (
    ExperimentConfig,
    ResolvedModel,
    load_experiment_config,
    normalized_config_hash,
    resolve_feature_groups,
)
from .manifest import (
    RunManifest,
    complete_run,
    create_run_layout,
    fail_run,
    record_stage,
    validate_resumable_run,
)


_PRE_CACHE_UNAVAILABLE_REASON = "pre_cache_model_missing_original_data_fingerprint"
_PRE_CACHE_PREDICTION_UNAVAILABLE_REASON = (
    "pre_cache_prediction_missing_original_data_fingerprint"
)
from .timing import resolve_experiment_timing, shift_score_frames_to_execution


@dataclass(frozen=True)
class ExperimentRunRequest:
    experiment_path: Path
    settings_path: Path = Path("configs/settings.yaml")
    feature_groups_path: Path = Path("configs/features.yaml")
    resume_dir: Path | None = None
    dry_run: bool = False
    on_progress: Callable[[str], None] | None = None


@dataclass(frozen=True)
class ExperimentRunResult:
    dry_run: bool
    run_dir: Path | None
    manifest_path: Path | None
    effective_train_range: tuple[int, int]
    effective_predict_range: tuple[int, int]
    required_plugins: tuple[str, ...]
    equity_curve_path: Path | None = None
    metrics_json_path: Path | None = None
    metrics_csv_path: Path | None = None
    common_rows: int = 0
    common_dates: int = 0


def run_experiment(request: ExperimentRunRequest) -> ExperimentRunResult:
    config = load_experiment_config(request.experiment_path)
    settings = load_settings(request.settings_path)
    groups = load_feature_groups(request.feature_groups_path)
    models = resolve_feature_groups(config, groups)
    cache_root = settings.paths.artifacts_dir / "cache"
    model_cache_root = cache_root / "models"
    prediction_cache_root = cache_root / "predictions"
    registry = FeatureRegistry()
    register_builtin_features(registry)
    required_plugins, feature_hashes = _required_feature_contract(registry, models)
    config_hash = normalized_config_hash(config)

    raw_store = RawStore(
        settings.paths.raw_db,
        settings.project.start_date,
        read_only=True,
    )
    feature_store: FeatureStore | None = None
    manifest: RunManifest | None = None
    current_stage = "preflight"
    try:
        requested_train_dates = _open_dates(
            raw_store, config.train.start, config.train.end
        )
        predict_dates = _open_dates(raw_store, config.predict.start, config.predict.end)
        cutoff_dates = _open_dates(
            raw_store, settings.project.start_date, config.data.feature_cutoff
        )
        if not requested_train_dates:
            raise ValueError("Training range contains no open trading dates")
        if not predict_dates:
            raise ValueError("Prediction range contains no open trading dates")
        if not cutoff_dates:
            raise ValueError("Feature cutoff contains no open trading dates")
        cutoff_trade_date = cutoff_dates[-1]
        target_definition = get_target_definition(config.train.target)
        timing = resolve_experiment_timing(
            calendar=cutoff_dates,
            train_dates=requested_train_dates,
            predict_dates=predict_dates,
            definition=target_definition,
        )
        train_dates = list(timing.train_dates)
        predict_dates = list(timing.predict_dates)
        timing_contract = {
            "signal_available": "close",
            "execution_price": (
                "next_open" if target_definition.execution_delay == 1 else "same_day_open"
            ),
            "execution_delay": target_definition.execution_delay,
            "target_entry_offset": target_definition.entry_offset,
            "target_exit_offset": target_definition.exit_offset,
            "holding_days": target_definition.holding_days,
            "purge_before_predict": target_definition.purge_before_predict,
            "last_train_target_exit": timing.last_train_target_exit,
            "execution_range": [
                timing.execution_dates[0],
                timing.execution_dates[-1],
            ],
        }
        _validate_raw_dependencies(
            raw_store, registry, required_plugins, cutoff_trade_date
        )
        installed_qlib = importlib.metadata.version("pyqlib")
        if installed_qlib != "0.9.7":
            raise RuntimeError(
                f"Experiment requires pyqlib==0.9.7, found {installed_qlib}"
            )
        requested_contract = _requested_contract(config)
        resolved_contract = {
            "train": [train_dates[0], train_dates[-1]],
            "predict": [predict_dates[0], predict_dates[-1]],
            "feature_cutoff": cutoff_trade_date,
            "label_horizon_end": timing.last_train_target_exit,
            "timing": timing_contract,
            "models": [
                {
                    "label": model.label,
                    "feature_group": model.feature_group,
                    "features": list(model.features),
                }
                for model in models
            ],
            "required_plugins": list(required_plugins),
        }
        _emit(
            request,
            f"preflight train={train_dates[0]}-{train_dates[-1]} "
            f"predict={predict_dates[0]}-{predict_dates[-1]} "
            f"execution={timing.execution_dates[0]}-{timing.execution_dates[-1]} "
            f"target={config.train.target} n_drop={config.backtest.n_drop}",
        )
        if request.dry_run:
            return ExperimentRunResult(
                dry_run=True,
                run_dir=None,
                manifest_path=None,
                effective_train_range=(train_dates[0], train_dates[-1]),
                effective_predict_range=(predict_dates[0], predict_dates[-1]),
                required_plugins=required_plugins,
            )

        if request.resume_dir is None:
            layout = create_run_layout(
                settings.paths.artifacts_dir, config.name, config_hash
            )
            manifest = RunManifest.create(
                layout,
                config_hash=config_hash,
                source_config=request.experiment_path,
                requested_contract=requested_contract,
                resolved_contract=resolved_contract,
                feature_hashes=feature_hashes,
            )
            _write_config_snapshot(config, manifest.run_dir / "experiment.yaml")
        else:
            manifest = validate_resumable_run(
                request.resume_dir,
                config_hash=config_hash,
                feature_hashes=feature_hashes,
                stage_order=_stage_order(models),
            )
        if "preflight" not in manifest.data["stages"]:
            record_stage(
                manifest,
                "preflight",
                artifacts=[manifest.run_dir / "experiment.yaml"],
                metadata=resolved_contract,
            )

        feature_store = FeatureStore(
            settings.paths.feature_db,
            settings.project.start_date,
        )

        current_stage = "features"
        if current_stage not in manifest.data["stages"]:
            updater = FeatureUpdater(
                raw_store=raw_store,
                feature_store=feature_store,
                registry=registry,
                calc_context=FeatureCalcContext(raw_store),
                plugin_ids=list(required_plugins),
                on_progress=lambda event: _emit(
                    request, f"feature {json.dumps(event, default=str, sort_keys=True)}"
                ),
            )
            updater.update_to(config.data.feature_cutoff)
            coverage = _validate_feature_coverage(
                feature_store,
                models,
                settings.project.start_date,
                cutoff_trade_date,
            )
            record_stage(
                manifest,
                current_stage,
                artifacts=[],
                metadata={"coverage": coverage},
            )
            _emit(request, f"stage complete {current_stage}")

        current_stage = "targets"
        if current_stage not in manifest.data["stages"]:
            targets = calculate_target(
                TargetCalcContext(raw_store),
                config.train.target,
                train_dates,
            )
            rows = feature_store.append_target_values(config.train.target, targets)
            target_contract = _validate_target_coverage(
                feature_store,
                config.train.target,
                train_dates,
            )
            record_stage(
                manifest,
                current_stage,
                artifacts=[],
                metadata={"upsert_rows": int(rows), **target_contract},
            )
            _emit(request, f"stage complete {current_stage} rows={rows}")

        model_artifacts: dict[str, SimpleNamespace] = {}
        for model in models:
            current_stage = f"train:{model.label}"
            job = TrainingJob(
                name=f"{config.name}_{model.label}",
                x_group=model.feature_group,
                features=list(model.features),
                target=config.train.target,
                train_range=(train_dates[0], train_dates[-1]),
                filters=["market.amount.v1 > 0"],
                model_params=asdict(config.model_params),
            )
            if current_stage in manifest.data["stages"]:
                paths = _stage_artifact_paths(manifest, current_stage)
                metadata = manifest.data["stages"][current_stage]["metadata"]
                cache_key = _trusted_model_cache_key(
                    metadata,
                    model,
                    model_cache_root,
                )
                if cache_key is not None:
                    model_artifacts[model.label] = SimpleNamespace(
                        model_path=_only_suffix(paths, ".pkl"),
                        meta_path=_only_suffix(paths, ".json"),
                        cache_key=cache_key,
                        cache_trusted=True,
                    )
                    continue
                model_artifacts[model.label] = SimpleNamespace(
                    model_path=_only_suffix(paths, ".pkl"),
                    meta_path=_only_suffix(paths, ".json"),
                    cache_key=None,
                    cache_trusted=False,
                )
                record_stage(
                    manifest,
                    current_stage,
                    artifacts=paths,
                    metadata={
                        **metadata,
                        "cache_hit": False,
                        "cache_schema_version": None,
                        "cache_mode": "unavailable",
                        "cache_key": None,
                        "cache_source": None,
                        "cache_unavailable_reason": _PRE_CACHE_UNAVAILABLE_REASON,
                    },
                )
                _emit(request, f"stage resumed {current_stage} cache=unavailable")
                continue
            artifact = get_or_train_cached_model(
                feature_store,
                job,
                model_cache_root,
                manifest.run_dir / "models" / model.label,
                None,
            )
            model_artifacts[model.label] = SimpleNamespace(
                model_path=artifact.model_path,
                meta_path=artifact.meta_path,
                cache_key=artifact.cache_key,
                cache_trusted=True,
            )
            record_stage(
                manifest,
                current_stage,
                artifacts=[artifact.model_path, artifact.meta_path],
                metadata=_model_stage_metadata(artifact, model),
            )
            _emit(request, f"stage complete {current_stage} rows={artifact.train_rows}")

        prediction_paths: dict[str, Path] = {}
        for model in models:
            current_stage = f"predict:{model.label}"
            if current_stage in manifest.data["stages"]:
                paths = _stage_artifact_paths(manifest, current_stage)
                prediction_paths[model.label] = _only_suffix(paths, ".pkl")
                metadata = manifest.data["stages"][current_stage]["metadata"]
                model_artifact = model_artifacts[model.label]
                if not _trusted_prediction_cache_stage(
                    metadata,
                    model_artifact,
                    prediction_cache_root,
                ):
                    reason = (
                        _PRE_CACHE_PREDICTION_UNAVAILABLE_REASON
                        if model_artifact.cache_trusted
                        else _PRE_CACHE_UNAVAILABLE_REASON
                    )
                    record_stage(
                        manifest,
                        current_stage,
                        artifacts=paths,
                        metadata={
                            **metadata,
                            "cache_hit": False,
                            "cache_schema_version": None,
                            "cache_mode": "unavailable",
                            "cache_key": None,
                            "cache_source": None,
                            "cache_unavailable_reason": reason,
                        },
                    )
                    _emit(request, f"stage resumed {current_stage} cache=unavailable")
                continue
            prediction_request = PredictionRequest(
                model_path=model_artifacts[model.label].model_path,
                meta_path=model_artifacts[model.label].meta_path,
                start_date=predict_dates[0],
                end_date=predict_dates[-1],
            )
            model_cache_key = model_artifacts[model.label].cache_key
            if _is_model_cache_key(model_cache_key):
                artifact = get_or_predict_cached(
                    feature_store,
                    prediction_request,
                    model_cache_key,
                    prediction_cache_root,
                    manifest.run_dir / "predictions" / model.label,
                    None,
                )
                prediction_metadata = {
                    "cache_hit": bool(artifact.cache_hit),
                    "cache_schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
                    "cache_mode": artifact.cache_mode,
                    "cache_key": artifact.cache_key,
                    "cache_source": artifact.cache_source,
                    "reused_rows": int(artifact.reused_rows),
                    "generated_rows": int(artifact.generated_rows),
                    "generated_range": artifact.generated_range,
                }
            else:
                artifact = predict_from_artifact(
                    feature_store,
                    prediction_request,
                    manifest.run_dir / "predictions" / model.label,
                )
                prediction_metadata = {
                    "cache_hit": False,
                    "cache_schema_version": None,
                    "cache_mode": "unavailable",
                    "cache_key": None,
                    "cache_source": None,
                    "reused_rows": 0,
                    "generated_rows": int(artifact.rows),
                    "generated_range": [predict_dates[0], predict_dates[-1]],
                    "cache_unavailable_reason": _PRE_CACHE_UNAVAILABLE_REASON,
                }
            prediction_paths[model.label] = artifact.prediction_path
            record_stage(
                manifest,
                current_stage,
                artifacts=[artifact.prediction_path],
                metadata={
                    "prediction_id": artifact.prediction_id,
                    "rows": int(artifact.rows),
                    **prediction_metadata,
                },
            )
            _emit(request, f"stage complete {current_stage} rows={artifact.rows}")

        current_stage = "scores"
        if current_stage in manifest.data["stages"]:
            score_frames = {
                path.stem: pd.read_pickle(path)
                for path in _stage_artifact_paths(manifest, current_stage)
            }
        else:
            predictions = {
                label: pd.read_pickle(path) for label, path in prediction_paths.items()
            }
            score_frames = build_common_scores(
                predictions,
                random_seed=config.backtest.random_seed,
                topk=config.backtest.topk,
            )
            score_paths: list[Path] = []
            for label, frame in score_frames.items():
                path = manifest.run_dir / "scores" / f"{label}.pkl"
                _atomic_pickle(frame, path)
                score_paths.append(path)
            record_stage(
                manifest,
                current_stage,
                artifacts=score_paths,
                metadata={
                    "rows": int(len(next(iter(score_frames.values())))),
                    "dates": int(next(iter(score_frames.values()))["trade_date"].nunique()),
                },
            )
            _emit(request, f"stage complete {current_stage}")

        current_stage = "execution_scores"
        if current_stage in manifest.data["stages"]:
            execution_score_frames = {
                path.stem: pd.read_pickle(path)
                for path in _stage_artifact_paths(manifest, current_stage)
            }
        else:
            execution_score_frames = shift_score_frames_to_execution(
                score_frames,
                timing.signal_to_execution,
            )
            execution_score_paths: list[Path] = []
            for label, frame in execution_score_frames.items():
                path = manifest.run_dir / "execution_scores" / f"{label}.pkl"
                _atomic_pickle(frame, path)
                execution_score_paths.append(path)
            record_stage(
                manifest,
                current_stage,
                artifacts=execution_score_paths,
                metadata={
                    "rows": int(len(next(iter(execution_score_frames.values())))),
                    "dates": int(
                        next(iter(execution_score_frames.values()))[
                            "trade_date"
                        ].nunique()
                    ),
                    "execution_range": [
                        timing.execution_dates[0],
                        timing.execution_dates[-1],
                    ],
                },
            )
            _emit(request, f"stage complete {current_stage}")

        current_stage = "provider"
        if current_stage in manifest.data["stages"]:
            provider_path = _only_directory(
                _stage_artifact_paths(manifest, current_stage)
            )
        else:
            seed_scores = execution_score_frames["random_baseline"].rename(
                columns={"score": "score_raw_blend"}
            )
            blend_path = manifest.run_dir / "provider_source" / "common_universe.pkl"
            _atomic_pickle(seed_scores, blend_path)
            dataset = build_score_dataset(
                raw_store,
                blend_path,
                manifest.run_dir / "provider_source",
            )
            market_scores = pd.read_pickle(dataset.dataset_path)
            factors = raw_store.fetch_df(
                """
                SELECT ts_code, trade_date, adj_factor
                FROM adj_factor
                WHERE trade_date BETWEEN ? AND ?
                """,
                [timing.execution_dates[0], timing.execution_dates[-1]],
            )
            temporary_provider = manifest.run_dir / "provider.pending"
            if temporary_provider.exists():
                _remove_inside(manifest.run_dir, temporary_provider)
            export_qlib_provider(market_scores, factors, temporary_provider)
            provider_path = manifest.run_dir / "provider"
            if provider_path.exists():
                _remove_inside(manifest.run_dir, provider_path)
            os.replace(temporary_provider, provider_path)
            record_stage(
                manifest,
                current_stage,
                artifacts=[provider_path, dataset.dataset_path],
                metadata={"dataset_rows": int(dataset.rows)},
            )
            _emit(request, f"stage complete {current_stage}")

        reports: dict[str, pd.DataFrame] = {}
        positions_by_label: dict[str, dict] = {}
        for label, scores in execution_score_frames.items():
            current_stage = f"backtest:{label}"
            if current_stage in manifest.data["stages"]:
                paths = _stage_artifact_paths(manifest, current_stage)
                report_path = next(path for path in paths if path.name == "report.pkl")
                positions_path = next(path for path in paths if path.name == "positions.pkl")
                reports[label] = pd.read_pickle(report_path)
                positions_by_label[label] = pd.read_pickle(positions_path)
                continue
            metrics, _ = run_qlib_topk_backtest(
                provider_path,
                scores,
                topk=config.backtest.topk,
                n_drop=config.backtest.n_drop,
                initial_cash=config.backtest.initial_cash,
            )
            report, positions = metrics["1day"]
            reports[label] = report
            positions_by_label[label] = positions
            output = manifest.run_dir / "backtests" / label
            report_path = output / "report.pkl"
            positions_path = output / "positions.pkl"
            _atomic_pickle(report, report_path)
            _atomic_pickle(positions, positions_path)
            record_stage(
                manifest,
                current_stage,
                artifacts=[report_path, positions_path],
                metadata={"report_rows": int(len(report)), "position_dates": int(len(positions))},
            )
            _emit(request, f"stage complete {current_stage}")
        _validate_backtest_outputs(reports, positions_by_label)

        current_stage = "publish"
        metrics_json_path = manifest.run_dir / "metrics.json"
        metrics_csv_path = manifest.run_dir / "metrics.csv"
        equity_curve_path = manifest.run_dir / "equity_curve.png"
        if current_stage not in manifest.data["stages"]:
            metrics, metrics_table = summarize_reports(reports)
            summary = {
                "contract": {
                    **resolved_contract,
                    "common_rows": int(len(next(iter(score_frames.values())))),
                    "common_dates": int(
                        next(iter(score_frames.values()))["trade_date"].nunique()
                    ),
                    "execution_rows": int(
                        len(next(iter(execution_score_frames.values())))
                    ),
                    "execution_dates": int(
                        next(iter(execution_score_frames.values()))[
                            "trade_date"
                        ].nunique()
                    ),
                    "topk": config.backtest.topk,
                    "n_drop": config.backtest.n_drop,
                    "initial_cash": config.backtest.initial_cash,
                    "random_seed": config.backtest.random_seed,
                },
                "series": metrics,
            }
            _atomic_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                metrics_json_path,
            )
            _atomic_csv(metrics_table, metrics_csv_path)
            display_reports = {
                _display_label(label, config, models): report
                for label, report in reports.items()
            }
            temporary_curve = equity_curve_path.with_suffix(".png.tmp")
            save_equity_curve(
                display_reports,
                temporary_curve,
                title=_equity_curve_title(models),
            )
            os.replace(temporary_curve, equity_curve_path)
            record_stage(
                manifest,
                current_stage,
                artifacts=[metrics_json_path, metrics_csv_path, equity_curve_path],
                metadata={"series": list(reports)},
            )
            _emit(request, f"stage complete {current_stage}")
        complete_run(manifest)
        first_scores = next(iter(score_frames.values()))
        return ExperimentRunResult(
            dry_run=False,
            run_dir=manifest.run_dir,
            manifest_path=manifest.path,
            effective_train_range=(train_dates[0], train_dates[-1]),
            effective_predict_range=(predict_dates[0], predict_dates[-1]),
            required_plugins=required_plugins,
            equity_curve_path=equity_curve_path,
            metrics_json_path=metrics_json_path,
            metrics_csv_path=metrics_csv_path,
            common_rows=int(len(first_scores)),
            common_dates=int(first_scores["trade_date"].nunique()),
        )
    except BaseException as exc:
        if manifest is not None and manifest.data.get("status") != "complete":
            fail_run(manifest, current_stage, exc)
        raise
    finally:
        if feature_store is not None:
            feature_store.close()
        raw_store.close()


def _required_feature_contract(
    registry: FeatureRegistry, models: tuple[ResolvedModel, ...]
) -> tuple[tuple[str, ...], dict[str, str]]:
    selected_features = {
        feature for model in models for feature in model.features
    }
    unknown: list[str] = []
    feature_hashes: dict[str, str] = {}
    selected_plugins: set[str] = set()
    for feature in selected_features:
        try:
            plugin = registry.get(feature)
        except KeyError:
            unknown.append(feature)
            continue
        selected_plugins.add(plugin.plugin_id)
        feature_hashes[feature] = plugin.code_hash
    if unknown:
        raise ValueError(f"Configured features are not registered: {sorted(unknown)}")
    ordered_plugins = tuple(
        plugin.plugin_id
        for plugin in registry.plugins()
        if plugin.plugin_id in selected_plugins
    )
    return ordered_plugins, dict(sorted(feature_hashes.items()))


def _open_dates(raw_store: RawStore, start: int, end: int) -> list[int]:
    frame = raw_store.fetch_df(
        """
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date BETWEEN ? AND ?
          AND is_open = 1
        ORDER BY cal_date
        """,
        [int(start), int(end)],
    )
    return [int(value) for value in frame["cal_date"].tolist()]


def _future_open_dates(raw_store: RawStore, start: int, *, limit: int) -> list[int]:
    frame = raw_store.fetch_df(
        f"""
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date > ?
          AND is_open = 1
        ORDER BY cal_date
        LIMIT {int(limit)}
        """,
        [int(start)],
    )
    return [int(value) for value in frame["cal_date"].tolist()]


def _validate_raw_dependencies(
    raw_store: RawStore,
    registry: FeatureRegistry,
    plugin_ids: tuple[str, ...],
    cutoff: int,
) -> None:
    for plugin_id in plugin_ids:
        plugin = registry.get_plugin(plugin_id)
        for raw_input in plugin.inputs:
            if not raw_input.startswith("raw."):
                raise RuntimeError(
                    f"Feature plugin input is not a raw table: {plugin_id} {raw_input}"
                )
            table_name = raw_input.removeprefix("raw.")
            state = raw_store.get_state(table_name)
            if state.last_success_trade_date is None or state.last_success_trade_date < cutoff:
                raise RuntimeError(
                    f"Raw dependency not ready table={table_name} "
                    f"actual={state.last_success_trade_date} required={cutoff}"
                )


def _validate_feature_coverage(
    store: FeatureStore,
    models: tuple[ResolvedModel, ...],
    start_date: int,
    cutoff: int,
) -> list[dict[str, object]]:
    features = sorted({feature for model in models for feature in model.features})
    for feature in features:
        state = store.get_state(feature)
        if state.last_success_trade_date is None or state.last_success_trade_date < cutoff:
            raise RuntimeError(
                f"Feature watermark not ready feature={feature} "
                f"actual={state.last_success_trade_date} required={cutoff}"
            )
    coverage = store.feature_coverage(features, start_date, cutoff)
    actual = set(str(value) for value in coverage.get("feature_name", []))
    missing = set(features).difference(actual)
    if missing:
        raise RuntimeError(f"Features contain no values: {sorted(missing)}")
    if (coverage["finite_count"] <= 0).any():
        invalid = coverage.loc[coverage["finite_count"] <= 0, "feature_name"].tolist()
        raise RuntimeError(f"Features contain no finite values: {invalid}")
    return json.loads(coverage.to_json(orient="records"))


def _validate_target_coverage(
    store: FeatureStore,
    target: str,
    train_dates: list[int],
) -> dict[str, int]:
    coverage = store.target_coverage(target, train_dates[0], train_dates[-1])
    row_count = int(coverage["row_count"])
    date_count = int(coverage["date_count"])
    if row_count <= 0 or date_count != len(train_dates):
        raise RuntimeError(
            f"Target coverage incomplete target={target} dates={date_count}/{len(train_dates)}"
        )
    return {"row_count": row_count, "date_count": date_count}


def _validate_backtest_outputs(
    reports: dict[str, pd.DataFrame], positions: dict[str, dict]
) -> None:
    expected_index: pd.Index | None = None
    for label, report in reports.items():
        if expected_index is None:
            expected_index = report.index
        elif not report.index.equals(expected_index):
            raise RuntimeError("Backtest reports do not have identical indexes")
        if report.isna().any().any():
            raise RuntimeError(f"Backtest report contains NaN values: {label}")
        if len(positions[label]) != len(report):
            raise RuntimeError(f"Backtest positions do not match report dates: {label}")


def _requested_contract(config: ExperimentConfig) -> dict[str, object]:
    return {
        "train": [config.train.start, config.train.end],
        "predict": [config.predict.start, config.predict.end],
        "feature_cutoff": config.data.feature_cutoff,
    }


def _write_config_snapshot(config: ExperimentConfig, path: Path) -> None:
    normalized = json.loads(json.dumps(asdict(config)))
    _atomic_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        path,
    )


def _stage_order(models: tuple[ResolvedModel, ...]) -> list[str]:
    return [
        "preflight",
        "features",
        "targets",
        *(f"train:{model.label}" for model in models),
        *(f"predict:{model.label}" for model in models),
        "scores",
        "execution_scores",
        "provider",
        "backtest:random_baseline",
        *(f"backtest:{model.label}" for model in models),
        "publish",
    ]


def _stage_artifact_paths(manifest: RunManifest, stage: str) -> list[Path]:
    return [
        manifest.run_dir / artifact["path"]
        for artifact in manifest.data["stages"][stage]["artifacts"]
    ]


def _only_suffix(paths: list[Path], suffix: str) -> Path:
    matches = [path for path in paths if path.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one '{suffix}' artifact, found {matches}")
    return matches[0]


def _only_directory(paths: list[Path]) -> Path:
    matches = [path for path in paths if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one provider directory, found {matches}")
    return matches[0]


def _atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.to_pickle(value, temporary)
    os.replace(temporary, path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _remove_inside(run_dir: Path, target: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_run)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to remove path outside run directory: {target}") from exc
    if resolved_target == resolved_run:
        raise RuntimeError("Refusing to remove the run directory")
    if resolved_target.is_dir():
        shutil.rmtree(resolved_target)
    elif resolved_target.exists():
        resolved_target.unlink()


def _display_label(
    label: str,
    config: ExperimentConfig,
    models: tuple[ResolvedModel, ...],
) -> str:
    if label == "random_baseline":
        return f"Random baseline (seed {config.backtest.random_seed})"
    model = next(model for model in models if model.label == label)
    return f"{len(model.features)} features"


def _model_stage_metadata(artifact, model: ResolvedModel) -> dict[str, object]:
    return {
        "artifact_hash": artifact.artifact_hash,
        "cache_hit": bool(artifact.cache_hit),
        "cache_schema_version": MODEL_CACHE_SCHEMA_VERSION,
        "cache_key": artifact.cache_key,
        "cache_source": artifact.cache_source,
        "train_rows": int(artifact.train_rows),
        "feature_group": model.feature_group,
        "features": list(model.features),
    }


def _is_model_cache_key(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trusted_model_cache_key(
    metadata: object,
    model: ResolvedModel,
    model_cache_root: Path,
) -> str | None:
    if not isinstance(metadata, dict):
        return None
    cache_key = metadata.get("cache_key")
    if (
        not _schema_version_matches(
            metadata.get("cache_schema_version"), MODEL_CACHE_SCHEMA_VERSION
        )
        or not _is_model_cache_key(cache_key)
        or metadata.get("artifact_hash") != cache_key
        or not isinstance(metadata.get("cache_hit"), bool)
        or metadata.get("feature_group") != model.feature_group
        or metadata.get("features") != list(model.features)
        or not _valid_nonnegative_int(metadata.get("train_rows"))
    ):
        return None
    expected_source = (model_cache_root / cache_key).resolve()
    if not _resolved_path_matches(metadata.get("cache_source"), expected_source):
        return None
    return cache_key


def _trusted_prediction_cache_stage(
    metadata: object,
    model_artifact: SimpleNamespace,
    prediction_cache_root: Path,
) -> bool:
    if not model_artifact.cache_trusted or not isinstance(metadata, dict):
        return False
    cache_key = metadata.get("cache_key")
    model_cache_key = model_artifact.cache_key
    if (
        not _schema_version_matches(
            metadata.get("cache_schema_version"), PREDICTION_CACHE_SCHEMA_VERSION
        )
        or not _is_model_cache_key(cache_key)
        or not _is_model_cache_key(model_cache_key)
        or not isinstance(metadata.get("cache_hit"), bool)
        or metadata.get("cache_mode") not in {"exact", "slice", "extend", "miss"}
        or not _valid_nonnegative_int(metadata.get("rows"))
        or not _valid_nonnegative_int(metadata.get("reused_rows"))
        or not _valid_nonnegative_int(metadata.get("generated_rows"))
    ):
        return False
    expected_source = (
        prediction_cache_root / model_cache_key / f"{cache_key}.pkl"
    ).resolve()
    return _resolved_path_matches(metadata.get("cache_source"), expected_source)


def _resolved_path_matches(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        path = Path(value)
        return path.is_absolute() and path.resolve() == expected
    except (OSError, RuntimeError, ValueError):
        return False


def _schema_version_matches(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _valid_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _equity_curve_title(models: tuple[ResolvedModel, ...]) -> str:
    comparisons = [
        "Random baseline",
        *(f"{len(model.features)} features" for model in models),
    ]
    return " vs ".join(comparisons) + " net equity"


def _emit(request: ExperimentRunRequest, message: str) -> None:
    if request.on_progress is not None:
        request.on_progress(message)
