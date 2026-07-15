from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from aicszl.experiments.manifest import (
    RunManifest,
    complete_run,
    create_run_layout,
    fail_run,
    record_stage,
    sha256_path,
    validate_resumable_run,
)


def _manifest(tmp_path: Path) -> RunManifest:
    layout = create_run_layout(
        tmp_path,
        "example",
        "abc12345",
        now=datetime(2026, 7, 14, 10, 30, 45),
    )
    return RunManifest.create(
        layout,
        config_hash="abc12345",
        source_config=tmp_path / "source.yaml",
        requested_contract={"train": [20200101, 20240101]},
        resolved_contract={"train": [20200102, 20231229]},
        feature_hashes={"market.close.v1": "code-one"},
    )


def test_create_layout_and_record_stage_publish_relative_hashed_artifacts(
    tmp_path: Path, monkeypatch
):
    manifest = _manifest(tmp_path)
    assert manifest.run_dir.name == "20260714-103045-abc12345"
    artifact = manifest.run_dir / "models" / "model.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")

    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def observed_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr("aicszl.experiments.manifest.os.replace", observed_replace)
    record_stage(manifest, "train:five", artifacts=[artifact], metadata={"rows": 12})

    saved = json.loads(manifest.path.read_text(encoding="utf-8"))
    stage = saved["stages"]["train:five"]
    assert stage["metadata"] == {"rows": 12}
    assert stage["artifacts"] == [
        {
            "path": "models/model.pkl",
            "kind": "file",
            "sha256": sha256_path(artifact),
        }
    ]
    assert calls[-1][0].name == "run_manifest.json.tmp"
    assert calls[-1][1] == manifest.path
    assert not manifest.path.with_name("run_manifest.json.tmp").exists()


def test_sha256_path_hashes_directory_contents_and_relative_names(tmp_path: Path):
    directory = tmp_path / "provider"
    directory.mkdir()
    (directory / "a.bin").write_bytes(b"a")
    nested = directory / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"b")
    before = sha256_path(directory)
    (nested / "b.bin").write_bytes(b"changed")
    after = sha256_path(directory)

    assert before != after


def test_resume_rejects_config_feature_mismatch_and_complete_run(tmp_path: Path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="configuration hash"):
        validate_resumable_run(
            manifest.run_dir,
            config_hash="different",
            feature_hashes={"market.close.v1": "code-one"},
        )
    with pytest.raises(ValueError, match="feature code hashes"):
        validate_resumable_run(
            manifest.run_dir,
            config_hash="abc12345",
            feature_hashes={"market.close.v1": "code-two"},
        )

    complete_run(manifest)
    with pytest.raises(ValueError, match="already complete"):
        validate_resumable_run(
            manifest.run_dir,
            config_hash="abc12345",
            feature_hashes={"market.close.v1": "code-one"},
        )
    with pytest.raises(RuntimeError, match="immutable"):
        record_stage(manifest, "late", artifacts=[])


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_resume_invalidates_damaged_stage_and_downstream(tmp_path: Path, damage: str):
    manifest = _manifest(tmp_path)
    artifacts: dict[str, Path] = {}
    for stage_name in ["features", "train:five", "publish"]:
        artifact = manifest.run_dir / f"{stage_name.replace(':', '_')}.bin"
        artifact.write_bytes(stage_name.encode())
        artifacts[stage_name] = artifact
        record_stage(manifest, stage_name, artifacts=[artifact])
    fail_run(manifest, "publish", RuntimeError("plot failed"))

    if damage == "missing":
        artifacts["train:five"].unlink()
    else:
        artifacts["train:five"].write_bytes(b"changed")
    resumed = validate_resumable_run(
        manifest.run_dir,
        config_hash="abc12345",
        feature_hashes={"market.close.v1": "code-one"},
        stage_order=["features", "train:five", "publish"],
    )

    assert list(resumed.data["stages"]) == ["features"]
    assert resumed.data["status"] == "running"
    assert resumed.data["failure"] is None


def test_fail_run_records_stage_and_exception(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fail_run(manifest, "features", RuntimeError("dependency missing"))

    saved = RunManifest.load(manifest.run_dir)
    assert saved.data["status"] == "failed"
    assert saved.data["failure"] == {
        "stage": "features",
        "type": "RuntimeError",
        "message": "dependency missing",
    }
