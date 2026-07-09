from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aicszl.config import load_settings
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
    raw_update.add_argument("--dry-run", action="store_true")
    raw_update.set_defaults(handler=_handle_raw_update)

    for name in ["feature", "target", "train", "predict", "blend", "backtest"]:
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


if __name__ == "__main__":
    raise SystemExit(main())
