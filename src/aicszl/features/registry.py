from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

from .store import FeatureMeta


@dataclass(frozen=True)
class FeaturePlugin:
    outputs: list[str]
    inputs: list[str]
    lookback_days: int
    func: Callable
    kind: str
    description: str
    code_hash: str

    def to_meta(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                feature_name=feature_name,
                domain=_parse_feature_name(feature_name)[0],
                version=_parse_feature_name(feature_name)[2],
                kind=self.kind,
                owner_plugin=self.func.__name__,
                input_tables=list(self.inputs),
                lookback_days=self.lookback_days,
                code_hash=self.code_hash,
                status="active",
                description=self.description,
            )
            for feature_name in self.outputs
        ]


class FeatureRegistry:
    def __init__(self):
        self._plugins_by_output: dict[str, FeaturePlugin] = {}

    def feature_plugin(
        self,
        outputs: list[str],
        inputs: list[str],
        lookback_days: int,
        kind: str = "derived",
        description: str = "",
    ):
        for output in outputs:
            _parse_feature_name(output)
            if output in self._plugins_by_output:
                raise ValueError(f"Feature {output} is already registered")

        def decorator(func: Callable) -> Callable:
            code_hash = _code_hash(func)
            plugin = FeaturePlugin(
                outputs=list(outputs),
                inputs=list(inputs),
                lookback_days=int(lookback_days),
                func=func,
                kind=kind,
                description=description,
                code_hash=code_hash,
            )
            for output in plugin.outputs:
                if output in self._plugins_by_output:
                    raise ValueError(f"Feature {output} is already registered")
                self._plugins_by_output[output] = plugin
            return func

        return decorator

    def get(self, feature_name: str) -> FeaturePlugin:
        if feature_name not in self._plugins_by_output:
            raise KeyError(f"Unknown feature plugin output: {feature_name}")
        return self._plugins_by_output[feature_name]

    def plugins(self) -> list[FeaturePlugin]:
        seen: set[int] = set()
        plugins: list[FeaturePlugin] = []
        for plugin in self._plugins_by_output.values():
            marker = id(plugin)
            if marker not in seen:
                seen.add(marker)
                plugins.append(plugin)
        return plugins


def _parse_feature_name(feature_name: str) -> tuple[str, str, str]:
    parts = feature_name.split(".")
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("v"):
        raise ValueError(f"Feature name must use domain.name.version: {feature_name}")
    return parts[0], parts[1], parts[2]


def _code_hash(func: Callable) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = repr(func)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
