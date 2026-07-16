from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from aicszl.raw.store import RawStore

from .registry import FeaturePlugin, FeatureRegistry
from .store import FeatureStore


@dataclass(frozen=True)
class FeaturePluginUpdatePlan:
    plugin_id: str
    outputs: list[str]
    start_trade_date: int | None
    target_trade_date: int | None
    trade_days: int
    status: str


@dataclass(frozen=True)
class FeaturePluginUpdateSummary:
    plugin_id: str
    status: str
    last_success_trade_date: int | None
    row_count: int


class FeatureUpdater:
    def __init__(
        self,
        *,
        raw_store: RawStore,
        feature_store: FeatureStore,
        registry: FeatureRegistry,
        calc_context: object,
        plugin_ids: list[str] | None = None,
        batch_days: int = 20,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ):
        self.raw_store = raw_store
        self.feature_store = feature_store
        self.registry = registry
        self.calc_context = calc_context
        self.plugin_ids = None if plugin_ids is None else list(plugin_ids)
        self.batch_days = max(1, int(batch_days))
        self.on_progress = on_progress
        self._plugins = self._select_plugins()

    def plan_to(self, target_date: int) -> list[FeaturePluginUpdatePlan]:
        all_dates = self._trade_dates(int(target_date))
        target_trade_date = all_dates[-1] if all_dates else None
        plans: list[FeaturePluginUpdatePlan] = []
        for plugin in self._plugins:
            code_changed = self._plugin_code_changed(plugin)
            if target_trade_date is not None:
                self._check_dependencies(plugin, target_trade_date)
            watermark, _ = self._effective_watermark(
                plugin,
                all_dates,
                force_from_start=code_changed,
            )
            pending_dates = self._pending_dates(all_dates, watermark)
            plan = FeaturePluginUpdatePlan(
                plugin_id=plugin.plugin_id,
                outputs=list(plugin.outputs),
                start_trade_date=pending_dates[0] if pending_dates else None,
                target_trade_date=target_trade_date,
                trade_days=len(pending_dates),
                status="pending" if pending_dates else "up-to-date",
            )
            plans.append(plan)
            self._emit(
                {
                    "event": "plan",
                    "plugin": plugin.plugin_id,
                    "start_trade_date": plan.start_trade_date,
                    "target_trade_date": plan.target_trade_date,
                    "trade_days": plan.trade_days,
                    "status": plan.status,
                }
            )
        return plans

    def update_to(self, target_date: int) -> dict[str, FeaturePluginUpdateSummary]:
        all_dates = self._trade_dates(int(target_date))
        target_trade_date = all_dates[-1] if all_dates else None
        summary: dict[str, FeaturePluginUpdateSummary] = {}
        for plugin in self._plugins:
            code_changed = self._plugin_code_changed(plugin)
            if target_trade_date is not None:
                self._check_dependencies(plugin, target_trade_date)
            if code_changed:
                with self.feature_store.transaction():
                    self.feature_store.reset_feature_plugin(plugin.plugin_id, plugin.outputs)
                    for meta in plugin.to_meta():
                        self.feature_store.register_feature_meta(meta)
            else:
                for meta in plugin.to_meta():
                    self.feature_store.register_feature_meta(meta)

            watermark, _ = self._effective_watermark(
                plugin,
                all_dates,
                force_from_start=code_changed,
            )
            pending_dates = self._pending_dates(all_dates, watermark)
            if not pending_dates:
                last_success = self._common_success_watermark(plugin)
                summary[plugin.plugin_id] = FeaturePluginUpdateSummary(
                    plugin_id=plugin.plugin_id,
                    status="up-to-date",
                    last_success_trade_date=last_success,
                    row_count=0,
                )
                self._emit(
                    {
                        "event": "up_to_date",
                        "plugin": plugin.plugin_id,
                        "trade_date": last_success,
                    }
                )
                continue

            total_rows = 0
            for batch in _batches(pending_dates, self.batch_days):
                total_rows += self._update_batch(plugin, batch, all_dates)
            summary[plugin.plugin_id] = FeaturePluginUpdateSummary(
                plugin_id=plugin.plugin_id,
                status="success",
                last_success_trade_date=pending_dates[-1],
                row_count=total_rows,
            )
        return summary

    def _select_plugins(self) -> list[FeaturePlugin]:
        if self.plugin_ids is not None:
            if not self.plugin_ids:
                raise ValueError("Feature plugin selection must not be empty")
            duplicates = _duplicates(self.plugin_ids)
            if duplicates:
                raise ValueError(f"Duplicate feature plugin IDs: {','.join(duplicates)}")
            plugins: list[FeaturePlugin] = []
            for plugin_id in self.plugin_ids:
                try:
                    plugin = self.registry.get_plugin(plugin_id)
                except KeyError as exc:
                    raise ValueError(f"Unknown feature plugin: {plugin_id}") from exc
                self._require_active(plugin, explicit=True)
                plugins.append(plugin)
            return plugins

        plugins = []
        for plugin in self.registry.plugins():
            if self._require_active(plugin, explicit=False):
                plugins.append(plugin)
        return plugins

    def _require_active(self, plugin: FeaturePlugin, *, explicit: bool) -> bool:
        statuses = self.feature_store.get_feature_statuses(plugin.outputs)
        persisted = list(statuses.values())
        if not persisted or all(status == "active" for status in persisted):
            return True
        if any(status == "active" for status in persisted):
            raise ValueError(f"Feature plugin has mixed output statuses: {plugin.plugin_id}")
        if explicit:
            raise ValueError(f"Feature plugin is not active: {plugin.plugin_id}")
        return False

    def _trade_dates(self, target_date: int) -> list[int]:
        rows = self.raw_store.fetch_df(
            """
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date BETWEEN ? AND ?
              AND is_open = 1
            ORDER BY cal_date
            """,
            [self.feature_store.start_date, int(target_date)],
        )
        return [int(value) for value in rows["cal_date"].tolist()]

    def _check_dependencies(self, plugin: FeaturePlugin, required_date: int) -> None:
        for raw_input in plugin.inputs:
            if not raw_input.startswith("raw."):
                raise RuntimeError(
                    f"feature update input must be raw table plugin={plugin.plugin_id} input={raw_input}"
                )
            table_name = raw_input.removeprefix("raw.")
            state = self.raw_store.get_state(table_name)
            actual = state.last_success_trade_date
            if actual is None or actual < required_date:
                raise RuntimeError(
                    f"feature update dependency not ready plugin={plugin.plugin_id} "
                    f"table={table_name} actual={actual} required={required_date}"
                )

    def _effective_watermark(
        self,
        plugin: FeaturePlugin,
        all_dates: list[int],
        *,
        force_from_start: bool = False,
    ) -> tuple[int | None, int | None]:
        if force_from_start:
            return None, None
        states = [self.feature_store.get_state(output) for output in plugin.outputs]
        watermarks = [state.last_success_trade_date for state in states]
        if all(watermark is None for watermark in watermarks):
            return None, None
        if any(watermark is None for watermark in watermarks):
            return None, None
        return min(int(watermark) for watermark in watermarks if watermark is not None), None

    def _plugin_code_changed(self, plugin: FeaturePlugin) -> bool:
        stored = self.feature_store.get_feature_code_hashes(plugin.outputs)
        return any(
            output in stored and stored[output] != plugin.code_hash
            for output in plugin.outputs
        )

    def _pending_dates(self, all_dates: list[int], watermark: int | None) -> list[int]:
        if watermark is None:
            return list(all_dates)
        return [trade_date for trade_date in all_dates if trade_date > watermark]

    def _update_batch(
        self,
        plugin: FeaturePlugin,
        batch: list[int],
        all_dates: list[int],
    ) -> int:
        for output in plugin.outputs:
            self.feature_store.mark_attempt(output, batch[-1], batch[-1])
        started = perf_counter()
        try:
            values = plugin.func(self.calc_context, list(batch))
            self._validate_values(plugin, values, batch, all_dates)
            with self.feature_store.transaction():
                rows = self.feature_store.append_plugin_values(
                    plugin.plugin_id,
                    plugin.outputs,
                    values,
                )
                for output in plugin.outputs:
                    output_rows = int(values[output].notna().sum()) if not values.empty else 0
                    self.feature_store.mark_success(
                        output,
                        batch[-1],
                        batch[-1],
                        row_count=output_rows,
                    )
        except Exception as exc:
            for output in plugin.outputs:
                self.feature_store.mark_failure(output, batch[-1], str(exc))
            self._emit(
                {
                    "event": "failed",
                    "plugin": plugin.plugin_id,
                    "trade_date": batch[-1],
                    "error": str(exc),
                }
            )
            raise RuntimeError(
                f"feature update failed plugin={plugin.plugin_id} "
                f"trade_dates={batch[0]}-{batch[-1]}: {exc}"
            ) from exc

        self._emit(
            {
                "event": "commit",
                "plugin": plugin.plugin_id,
                "start_trade_date": batch[0],
                "end_trade_date": batch[-1],
                "dates": len(batch),
                "rows": rows,
                "commit_ms": int((perf_counter() - started) * 1000),
            }
        )
        return rows

    def _validate_values(
        self,
        plugin: FeaturePlugin,
        values: pd.DataFrame,
        batch: list[int],
        all_dates: list[int],
    ) -> None:
        expected_columns = ["ts_code", "trade_date", *plugin.outputs]
        if list(values.columns) != expected_columns:
            raise ValueError(
                f"feature plugin result columns must be exactly {expected_columns}"
            )
        if values.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("feature plugin returned duplicate (ts_code, trade_date) keys")

        returned_dates = set(int(value) for value in values["trade_date"].unique())
        unexpected_dates = returned_dates.difference(batch)
        if unexpected_dates:
            raise ValueError(f"feature plugin returned dates outside batch: {sorted(unexpected_dates)}")

    def _common_success_watermark(self, plugin: FeaturePlugin) -> int | None:
        watermarks = [
            self.feature_store.get_state(output).last_success_trade_date for output in plugin.outputs
        ]
        present = [int(value) for value in watermarks if value is not None]
        return min(present) if present else None

    def _emit(self, event: dict[str, object]) -> None:
        if self.on_progress is not None:
            self.on_progress(event)


def _batches(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
