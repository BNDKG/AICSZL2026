from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicszl.experiments import (  # noqa: E402
    ExperimentRunRequest,
    ExperimentRunResult,
    run_experiment,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible multi-feature-group training and backtest experiment."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--feature-groups", default="configs/features.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _progress(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    request = ExperimentRunRequest(
        experiment_path=Path(args.experiment),
        settings_path=Path(args.config),
        feature_groups_path=Path(args.feature_groups),
        resume_dir=None if args.resume is None else Path(args.resume),
        dry_run=bool(args.dry_run),
        on_progress=_progress,
    )
    try:
        result = run_experiment(request)
    except Exception as exc:
        print(f"experiment failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1

    _print_result(result)
    return 0


def _print_result(result: ExperimentRunResult) -> None:
    train = result.effective_train_range
    predict = result.effective_predict_range
    if result.dry_run:
        print(
            f"dry-run complete train={train[0]}-{train[1]} "
            f"predict={predict[0]}-{predict[1]}",
            flush=True,
        )
        print(f"required_plugins={','.join(result.required_plugins)}", flush=True)
        return

    print(f"run complete path={result.run_dir.resolve()}", flush=True)
    print(f"manifest path={result.manifest_path.resolve()}", flush=True)
    print(f"equity curve path={result.equity_curve_path.resolve()}", flush=True)
    print(f"metrics json path={result.metrics_json_path.resolve()}", flush=True)
    print(f"metrics csv path={result.metrics_csv_path.resolve()}", flush=True)
    print(
        f"effective train={train[0]}-{train[1]} "
        f"predict={predict[0]}-{predict[1]}",
        flush=True,
    )
    print(
        f"common_rows={result.common_rows} common_dates={result.common_dates}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
