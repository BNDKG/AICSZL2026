from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aicszl.config import load_settings
from aicszl.features import FeatureRegistry, FeatureStore, FeatureUpdater
from aicszl.features.builtins import FeatureCalcContext, register_builtin_features
from aicszl.raw import RawStore, RawUpdater, TushareRawClient


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aicszl",
        description="AICSZL2026 research workflow",
    )
    subparsers = parser.add_subparsers(dest="command")

    raw_parser = subparsers.add_parser("raw", help="Raw data workflows")
    raw_subparsers = raw_parser.add_subparsers(dest="raw_command")
    raw_update = raw_subparsers.add_parser("update", help="Update raw Tushare data")
    raw_update.add_argument("--to", required=True, dest="target_date")
    raw_update.add_argument("--config", default="configs/settings.yaml")
    raw_update.add_argument("--tables", default="daily")
    raw_update.add_argument("--batch-days", type=int, default=20)
    raw_update.add_argument("--retries", type=int, default=3)
    raw_update.add_argument("--retry-sleep-ms", type=int, default=1000)
    raw_update.add_argument("--dry-run", action="store_true")
    raw_update.set_defaults(handler=_handle_raw_update)

    feature_parser = subparsers.add_parser("feature", help="Feature workflows")
    feature_subparsers = feature_parser.add_subparsers(dest="feature_command")
    feature_list = feature_subparsers.add_parser("list", help="List registered feature plugins")
    feature_list.add_argument("--config", default="configs/settings.yaml")
    feature_list.set_defaults(handler=_handle_feature_list)
    feature_update = feature_subparsers.add_parser(
        "update", help="Update registered feature plugins"
    )
    feature_update.add_argument("--to", required=True, dest="target_date")
    feature_update.add_argument("--config", default="configs/settings.yaml")
    feature_update.add_argument("--plugins")
    feature_update.add_argument("--batch-days", type=int, default=20)
    feature_update.add_argument("--dry-run", action="store_true")
    feature_update.set_defaults(handler=_handle_feature_update)

    for name in ["target", "train", "predict", "blend", "backtest"]:
        child = subparsers.add_parser(name, help=f"{name.title()} workflows")
        child.set_defaults(handler=_handle_placeholder)

    return parser


def _handle_raw_update(args: argparse.Namespace) -> int:
    tables = [name.strip() for name in args.tables.split(",") if name.strip()]
    if args.dry_run:
        print(f"raw update dry-run to {args.target_date} tables={','.join(tables)}")
        return 0

    settings = load_settings(args.config)
    store = RawStore(settings.paths.raw_db, start_date=settings.project.start_date)
    try:
        client = TushareRawClient.from_token_file(settings.tushare.token_file)
        updater = RawUpdater(
            store=store,
            client=client,
            tables=tables,
            batch_days=args.batch_days,
            max_retries=args.retries,
            retry_sleep_seconds=args.retry_sleep_ms / 1000,
            on_progress=_print_raw_progress,
        )
        try:
            summary = updater.update_to(int(args.target_date))
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    for table_name, state in summary.items():
        print(
            f"{table_name} {state.status} "
            f"{state.last_success_trade_date} rows={state.row_count}"
        )
    return 0


def _handle_feature_list(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    registry = _builtin_feature_registry()
    store = FeatureStore(
        settings.paths.feature_db,
        start_date=settings.project.start_date,
        read_only=True,
    )
    try:
        for plugin in registry.plugins():
            statuses = store.get_feature_statuses(plugin.outputs)
            persisted_statuses = list(statuses.values())
            if not persisted_statuses or all(status == "active" for status in persisted_statuses):
                status = "active"
            elif all(status != "active" for status in persisted_statuses):
                status = "inactive"
            else:
                status = "invalid-mixed"
            watermarks = [
                store.get_state(output).last_success_trade_date for output in plugin.outputs
            ]
            present = [int(value) for value in watermarks if value is not None]
            watermark = min(present) if present else None
            print(
                f"{plugin.plugin_id} status={status} watermark={watermark} "
                f"outputs={','.join(plugin.outputs)} inputs={','.join(plugin.inputs)} "
                f"lookback_days={plugin.lookback_days}"
            )
    finally:
        store.close()
    return 0


def _handle_feature_update(args: argparse.Namespace) -> int:
    plugin_ids = _parse_plugin_ids(args.plugins)
    settings = load_settings(args.config)
    registry = _builtin_feature_registry()
    read_only = bool(args.dry_run)
    raw_store = RawStore(
        settings.paths.raw_db,
        start_date=settings.project.start_date,
        read_only=read_only,
    )
    feature_store = FeatureStore(
        settings.paths.feature_db,
        start_date=settings.project.start_date,
        read_only=read_only,
    )
    try:
        try:
            updater = FeatureUpdater(
                raw_store=raw_store,
                feature_store=feature_store,
                registry=registry,
                calc_context=FeatureCalcContext(raw_store),
                plugin_ids=plugin_ids,
                batch_days=args.batch_days,
                on_progress=_print_feature_progress,
            )
            if args.dry_run:
                for plan in updater.plan_to(int(args.target_date)):
                    start = plan.start_trade_date
                    end = plan.target_trade_date
                    print(
                        f"{plan.plugin_id} {plan.status} {start}-{end} "
                        f"trade_days={plan.trade_days}"
                    )
                return 0
            summary = updater.update_to(int(args.target_date))
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    finally:
        feature_store.close()
        raw_store.close()

    for plugin_id, state in summary.items():
        print(
            f"{plugin_id} {state.status} "
            f"{state.last_success_trade_date} rows={state.row_count}"
        )
    return 0


def _builtin_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    register_builtin_features(registry)
    return registry


def _parse_plugin_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def _handle_placeholder(args: argparse.Namespace) -> int:
    print(f"{args.command} workflow is not implemented yet")
    return 2


def _print_raw_progress(event: dict[str, object]) -> None:
    name = event.get("event")
    table = event.get("table")
    trade_date = event.get("trade_date")
    if name == "fetch":
        print(
            f"[raw] fetch {table} {trade_date} rows={event.get('rows')} "
            f"fetch_ms={event.get('fetch_ms')}",
            flush=True,
        )
    elif name == "commit":
        print(
            f"[raw] commit {table} {event.get('start_trade_date')}-{event.get('end_trade_date')} "
            f"dates={event.get('dates')} rows={event.get('rows')} "
            f"commit_ms={event.get('commit_ms')}",
            flush=True,
        )
    elif name == "failed":
        print(f"[raw] failed {table} {trade_date} error={event.get('error')}", flush=True)
    elif name == "retry":
        print(
            f"[raw] retry {table} {trade_date} attempt={event.get('attempt')}/"
            f"{event.get('max_retries')} sleep_seconds={event.get('sleep_seconds')} "
            f"error={event.get('error')}",
            flush=True,
        )


def _print_feature_progress(event: dict[str, object]) -> None:
    name = event.get("event")
    plugin = event.get("plugin")
    if name == "commit":
        print(
            f"[feature] commit {plugin} "
            f"{event.get('start_trade_date')}-{event.get('end_trade_date')} "
            f"dates={event.get('dates')} rows={event.get('rows')} "
            f"commit_ms={event.get('commit_ms')}",
            flush=True,
        )
    elif name == "failed":
        print(
            f"[feature] failed {plugin} {event.get('trade_date')} "
            f"error={event.get('error')}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
