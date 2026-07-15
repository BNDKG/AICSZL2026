from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    manifest_path: Path


class RunManifest:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path.resolve()
        self.run_dir = self.path.parent
        self.data = data

    @classmethod
    def create(
        cls,
        layout: RunLayout,
        *,
        config_hash: str,
        source_config: str | Path,
        requested_contract: dict[str, Any],
        resolved_contract: dict[str, Any],
        feature_hashes: dict[str, str],
    ) -> RunManifest:
        now = _timestamp()
        manifest = cls(
            layout.manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "config_hash": str(config_hash),
                "source_config": str(Path(source_config).resolve()),
                "requested_contract": requested_contract,
                "resolved_contract": resolved_contract,
                "feature_hashes": dict(sorted(feature_hashes.items())),
                "stages": {},
                "failure": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        manifest.save()
        return manifest

    @classmethod
    def load(cls, run_dir_or_path: str | Path) -> RunManifest:
        candidate = Path(run_dir_or_path)
        path = candidate if candidate.name == "run_manifest.json" else candidate / "run_manifest.json"
        if not path.is_file():
            raise ValueError(f"Run manifest does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported run manifest schema: {data.get('schema_version')}")
        return cls(path, data)

    def save(self) -> None:
        self.data["updated_at"] = _timestamp()
        temporary = self.path.with_name("run_manifest.json.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.path)


def create_run_layout(
    artifacts_dir: str | Path,
    experiment_name: str,
    config_hash: str,
    *,
    now: datetime | None = None,
) -> RunLayout:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(artifacts_dir)
        / "experiments"
        / experiment_name
        / "runs"
        / f"{timestamp}-{config_hash}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunLayout(run_dir=run_dir, manifest_path=run_dir / "run_manifest.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise ValueError(f"Artifact path does not exist: {target}")
    digest = hashlib.sha256()
    for file in sorted(item for item in target.rglob("*") if item.is_file()):
        relative = file.relative_to(target).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def record_stage(
    manifest: RunManifest,
    stage: str,
    *,
    artifacts: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    _require_mutable(manifest)
    artifact_records = [
        _artifact_record(manifest.run_dir, Path(artifact)) for artifact in artifacts
    ]
    manifest.data["stages"][stage] = {
        "completed_at": _timestamp(),
        "metadata": metadata or {},
        "artifacts": artifact_records,
    }
    manifest.data["status"] = "running"
    manifest.data["failure"] = None
    manifest.save()


def fail_run(manifest: RunManifest, stage: str, error: BaseException) -> None:
    _require_mutable(manifest)
    manifest.data["status"] = "failed"
    manifest.data["failure"] = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
    }
    manifest.save()


def complete_run(manifest: RunManifest) -> None:
    _require_mutable(manifest)
    manifest.data["status"] = "complete"
    manifest.data["failure"] = None
    manifest.data["completed_at"] = _timestamp()
    manifest.save()


def validate_resumable_run(
    run_dir: str | Path,
    *,
    config_hash: str,
    feature_hashes: dict[str, str],
    stage_order: list[str] | None = None,
) -> RunManifest:
    manifest = RunManifest.load(run_dir)
    if manifest.data.get("status") == "complete":
        raise ValueError("Run is already complete and immutable")
    if manifest.data.get("config_hash") != config_hash:
        raise ValueError("Run configuration hash does not match")
    expected_feature_hashes = dict(sorted(feature_hashes.items()))
    if manifest.data.get("feature_hashes") != expected_feature_hashes:
        raise ValueError("Run feature code hashes do not match")

    recorded = list(manifest.data.get("stages", {}))
    invalid_stage: str | None = None
    for stage in recorded:
        if not _stage_artifacts_valid(manifest, stage):
            invalid_stage = stage
            break
    if invalid_stage is not None:
        order = stage_order or recorded
        if invalid_stage in order:
            invalid_index = order.index(invalid_stage)
            invalid_names = set(order[invalid_index:])
            manifest.data["stages"] = {
                name: value
                for name, value in manifest.data["stages"].items()
                if name not in invalid_names
            }
        else:
            invalid_index = recorded.index(invalid_stage)
            manifest.data["stages"] = {
                name: value
                for name, value in manifest.data["stages"].items()
                if recorded.index(name) < invalid_index
            }
    manifest.data["status"] = "running"
    manifest.data["failure"] = None
    manifest.save()
    return manifest


def _artifact_record(run_dir: Path, artifact: Path) -> dict[str, str]:
    resolved = artifact.resolve()
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact must be inside run directory: {resolved}") from exc
    if not resolved.exists():
        raise ValueError(f"Artifact path does not exist: {resolved}")
    return {
        "path": relative.as_posix(),
        "kind": "directory" if resolved.is_dir() else "file",
        "sha256": sha256_path(resolved),
    }


def _stage_artifacts_valid(manifest: RunManifest, stage: str) -> bool:
    stage_data = manifest.data["stages"].get(stage, {})
    for artifact in stage_data.get("artifacts", []):
        path = manifest.run_dir / artifact["path"]
        if not path.exists():
            return False
        expected_kind = artifact.get("kind")
        if expected_kind == "file" and not path.is_file():
            return False
        if expected_kind == "directory" and not path.is_dir():
            return False
        if sha256_path(path) != artifact.get("sha256"):
            return False
    return True


def _require_mutable(manifest: RunManifest) -> None:
    if manifest.data.get("status") == "complete":
        raise RuntimeError("Complete run manifest is immutable")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
