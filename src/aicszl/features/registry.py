from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass

from .store import FeatureMeta


@dataclass(frozen=True)
class FeaturePlugin:
    plugin_id: str
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
                owner_plugin=self.plugin_id,
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
        self._plugins_by_id: dict[str, FeaturePlugin] = {}

    def feature_plugin(
        self,
        *,
        plugin_id: str,
        outputs: list[str],
        inputs: list[str],
        lookback_days: int,
        kind: str = "derived",
        description: str = "",
    ):
        _parse_plugin_id(plugin_id)
        if plugin_id in self._plugins_by_id:
            raise ValueError(f"Feature plugin {plugin_id} is already registered")
        for output in outputs:
            _parse_feature_name(output)
            if output in self._plugins_by_output:
                raise ValueError(f"Feature {output} is already registered")

        def decorator(func: Callable) -> Callable:
            code_hash = _code_hash(func)
            plugin = FeaturePlugin(
                plugin_id=plugin_id,
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
            self._plugins_by_id[plugin.plugin_id] = plugin
            return func

        return decorator

    def get(self, feature_name: str) -> FeaturePlugin:
        if feature_name not in self._plugins_by_output:
            raise KeyError(f"Unknown feature plugin output: {feature_name}")
        return self._plugins_by_output[feature_name]

    def get_plugin(self, plugin_id: str) -> FeaturePlugin:
        if plugin_id not in self._plugins_by_id:
            raise KeyError(f"Unknown feature plugin: {plugin_id}")
        return self._plugins_by_id[plugin_id]

    def plugins(self) -> list[FeaturePlugin]:
        return list(self._plugins_by_id.values())


def _parse_feature_name(feature_name: str) -> tuple[str, str, str]:
    parts = feature_name.split(".")
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("v"):
        raise ValueError(f"Feature name must use domain.name.version: {feature_name}")
    return parts[0], parts[1], parts[2]


def _parse_plugin_id(plugin_id: str) -> tuple[str, str, str]:
    parts = plugin_id.split(".")
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("v"):
        raise ValueError(f"Feature plugin ID must use domain.name.version: {plugin_id}")
    return parts[0], parts[1], parts[2]


def _code_hash(func: Callable) -> str:
    source = _function_source(func)
    dependencies: list[dict[str, object]] = []
    visited = {id(func)}
    _collect_code_dependencies(func, dependencies, visited)
    if not dependencies:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    payload = source + "\n#aicszl-code-dependencies-v1\n" + json.dumps(
        dependencies,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_code_dependencies(
    func: Callable,
    dependencies: list[dict[str, object]],
    visited: set[int],
) -> None:
    defaults = _stable_dependency_value(getattr(func, "__defaults__", None))
    if defaults is not _UNSUPPORTED and defaults is not None:
        dependencies.append(
            {"kind": "defaults", "owner": func.__qualname__, "value": defaults}
        )
    keyword_defaults = _stable_dependency_value(getattr(func, "__kwdefaults__", None))
    if keyword_defaults is not _UNSUPPORTED and keyword_defaults is not None:
        dependencies.append(
            {"kind": "keyword_defaults", "owner": func.__qualname__, "value": keyword_defaults}
        )
    closure = getattr(func, "__closure__", None) or ()
    for name, cell in sorted(
        zip(getattr(func.__code__, "co_freevars", ()), closure, strict=False),
        key=lambda item: item[0],
    ):
        try:
            value = _stable_dependency_value(cell.cell_contents)
        except ValueError:
            continue
        if value is not _UNSUPPORTED:
            dependencies.append(
                {"kind": "closure", "owner": func.__qualname__, "name": name, "value": value}
            )

    for name in sorted(set(getattr(func.__code__, "co_names", ()))):
        if name not in func.__globals__:
            continue
        value = func.__globals__[name]
        if inspect.isfunction(value) and (
            value.__module__ == func.__module__
            or (value.__module__ or "").startswith("aicszl.")
        ):
            if id(value) in visited:
                continue
            visited.add(id(value))
            dependencies.append(
                {
                    "kind": "function",
                    "name": value.__qualname__,
                    "source": _function_source(value),
                }
            )
            _collect_code_dependencies(value, dependencies, visited)
            continue
        stable = _stable_dependency_value(value)
        if stable is not _UNSUPPORTED:
            dependencies.append(
                {"kind": "global", "owner": func.__qualname__, "name": name, "value": stable}
            )


def _function_source(func: Callable) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return repr(func)


_UNSUPPORTED = object()


def _stable_dependency_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": repr(value)}
    if isinstance(value, tuple):
        items = [_stable_dependency_value(item) for item in value]
        if any(item is _UNSUPPORTED for item in items):
            return _UNSUPPORTED
        return {"tuple": items}
    if isinstance(value, list):
        items = [_stable_dependency_value(item) for item in value]
        if any(item is _UNSUPPORTED for item in items):
            return _UNSUPPORTED
        return {"list": items}
    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for key in sorted(value, key=lambda item: repr(item)):
            stable_key = _stable_dependency_value(key)
            stable_value = _stable_dependency_value(value[key])
            if stable_key is _UNSUPPORTED or stable_value is _UNSUPPORTED:
                return _UNSUPPORTED
            items.append((json.dumps(stable_key, sort_keys=True), stable_value))
        return {"dict": items}
    if isinstance(value, (set, frozenset)):
        items = [_stable_dependency_value(item) for item in value]
        if any(item is _UNSUPPORTED for item in items):
            return _UNSUPPORTED
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    return _UNSUPPORTED
