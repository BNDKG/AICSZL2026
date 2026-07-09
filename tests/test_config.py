from pathlib import Path

import pytest

from aicszl.config import load_settings


def test_load_settings_returns_normalized_paths(tmp_path: Path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  start_date: 20200101",
                "paths:",
                "  raw_db: data/raw.duckdb",
                "  feature_db: data/features.duckdb",
                "  artifacts_dir: artifacts",
                "tushare:",
                "  token_file: token.txt",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.project.start_date == 20200101
    assert settings.paths.raw_db == Path("data/raw.duckdb")
    assert settings.paths.feature_db == Path("data/features.duckdb")
    assert settings.paths.artifacts_dir == Path("artifacts")
    assert settings.tushare.token_file == Path("token.txt")


def test_load_settings_rejects_start_date_before_20200101(tmp_path: Path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  start_date: 20190101",
                "paths:",
                "  raw_db: data/raw.duckdb",
                "  feature_db: data/features.duckdb",
                "  artifacts_dir: artifacts",
                "tushare:",
                "  token_file: token.txt",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start_date"):
        load_settings(config_path)
