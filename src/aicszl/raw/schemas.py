from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTableSpec:
    name: str
    columns: tuple[tuple[str, str], ...]
    primary_keys: tuple[str, ...]

    @property
    def column_names(self) -> list[str]:
        return [name for name, _ in self.columns]


RAW_TABLES: dict[str, RawTableSpec] = {
    "trade_cal": RawTableSpec(
        name="trade_cal",
        columns=(
            ("cal_date", "INTEGER"),
            ("exchange", "VARCHAR"),
            ("is_open", "INTEGER"),
            ("pretrade_date", "INTEGER"),
        ),
        primary_keys=("cal_date",),
    ),
    "daily": RawTableSpec(
        name="daily",
        columns=(
            ("ts_code", "VARCHAR"),
            ("trade_date", "INTEGER"),
            ("open", "DOUBLE"),
            ("high", "DOUBLE"),
            ("low", "DOUBLE"),
            ("close", "DOUBLE"),
            ("pre_close", "DOUBLE"),
            ("change", "DOUBLE"),
            ("pct_chg", "DOUBLE"),
            ("vol", "DOUBLE"),
            ("amount", "DOUBLE"),
        ),
        primary_keys=("ts_code", "trade_date"),
    ),
    "adj_factor": RawTableSpec(
        name="adj_factor",
        columns=(("ts_code", "VARCHAR"), ("trade_date", "INTEGER"), ("adj_factor", "DOUBLE")),
        primary_keys=("ts_code", "trade_date"),
    ),
    "stk_limit": RawTableSpec(
        name="stk_limit",
        columns=(
            ("trade_date", "INTEGER"),
            ("ts_code", "VARCHAR"),
            ("up_limit", "DOUBLE"),
            ("down_limit", "DOUBLE"),
        ),
        primary_keys=("ts_code", "trade_date"),
    ),
    "moneyflow": RawTableSpec(
        name="moneyflow",
        columns=(
            ("ts_code", "VARCHAR"),
            ("trade_date", "INTEGER"),
            ("buy_sm_vol", "DOUBLE"),
            ("buy_sm_amount", "DOUBLE"),
            ("sell_sm_vol", "DOUBLE"),
            ("sell_sm_amount", "DOUBLE"),
            ("buy_md_vol", "DOUBLE"),
            ("buy_md_amount", "DOUBLE"),
            ("sell_md_vol", "DOUBLE"),
            ("sell_md_amount", "DOUBLE"),
            ("buy_lg_vol", "DOUBLE"),
            ("buy_lg_amount", "DOUBLE"),
            ("sell_lg_vol", "DOUBLE"),
            ("sell_lg_amount", "DOUBLE"),
            ("buy_elg_vol", "DOUBLE"),
            ("buy_elg_amount", "DOUBLE"),
            ("sell_elg_vol", "DOUBLE"),
            ("sell_elg_amount", "DOUBLE"),
            ("net_mf_vol", "DOUBLE"),
            ("net_mf_amount", "DOUBLE"),
        ),
        primary_keys=("ts_code", "trade_date"),
    ),
    "daily_basic": RawTableSpec(
        name="daily_basic",
        columns=(
            ("ts_code", "VARCHAR"),
            ("trade_date", "INTEGER"),
            ("turnover_rate", "DOUBLE"),
            ("volume_ratio", "DOUBLE"),
            ("pe", "DOUBLE"),
            ("pb", "DOUBLE"),
            ("ps_ttm", "DOUBLE"),
            ("dv_ttm", "DOUBLE"),
            ("circ_mv", "DOUBLE"),
            ("total_mv", "DOUBLE"),
        ),
        primary_keys=("ts_code", "trade_date"),
    ),
}
