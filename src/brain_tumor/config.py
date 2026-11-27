"""Configuration loading.

All settings live in ``params.yaml``. Entry points accept dotted overrides such as
``train.model=resnet18 train.epochs=20``; values are parsed as YAML so numbers,
booleans, lists and ``null`` work as expected.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_ENV_VAR = "BRAIN_TUMOR_PARAMS"


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` as a float (plain YAML 1.1 reads it as a string)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


def find_params_file(path: str | os.PathLike | None = None) -> Path:
    """Locate params.yaml: explicit path > env var > current directory > project root."""
    if path is not None:
        return Path(path)
    if os.environ.get(PARAMS_ENV_VAR):
        return Path(os.environ[PARAMS_ENV_VAR])
    cwd_params = Path.cwd() / "params.yaml"
    if cwd_params.exists():
        return cwd_params
    return PROJECT_ROOT / "params.yaml"


def load_params(
    path: str | os.PathLike | None = None,
    overrides: Iterable[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load params.yaml and apply overrides.

    The directory containing the file is stored under ``cfg["_root"]`` so that
    relative paths resolve the same way whatever the working directory is.
    """
    params_path = find_params_file(path).resolve()
    if not params_path.exists():
        raise ConfigError(f"Parameter file not found: {params_path}")
    with params_path.open() as f:
        cfg = yaml_load(f.read()) or {}
    cfg["_root"] = str(params_path.parent)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def parse_override(item: str) -> tuple[str, Any]:
    """Parse ``a.b.c=value`` into (``a.b.c``, parsed value)."""
    if "=" not in item:
        raise ConfigError(f"Override must look like key=value, got: {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Empty key in override: {item!r}")
    return key, yaml_load(raw) if raw.strip() else None


def apply_overrides(
    cfg: dict[str, Any], overrides: Iterable[str] | Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``cfg`` with dotted overrides applied.

    Unknown keys raise :class:`ConfigError` so that typos never fail silently.
    """
    cfg = copy.deepcopy(cfg)
    items = overrides.items() if isinstance(overrides, Mapping) else map(parse_override, overrides)
    for key, value in items:
        set_by_path(cfg, key, value, strict=True)
    return cfg


def get_by_path(cfg: Mapping[str, Any], dotted: str, default: Any = ...) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is ...:
                raise ConfigError(f"Unknown config key: {dotted}")
            return default
        node = node[part]
    return node


def set_by_path(cfg: dict[str, Any], dotted: str, value: Any, strict: bool = True) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            if strict:
                raise ConfigError(f"Unknown config key: {dotted}")
            node[part] = {}
        node = node[part]
    if strict and parts[-1] not in node:
        raise ConfigError(f"Unknown config key: {dotted}")
    node[parts[-1]] = value


def project_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Resolve ``paths.<key>`` relative to the directory of params.yaml."""
    value = Path(get_by_path(cfg, f"paths.{key}"))
    return value if value.is_absolute() else Path(cfg.get("_root", ".")) / value


def load_best_params(path: str | os.PathLike) -> dict[str, Any]:
    """Load a ``{dotted.key: value}`` file written by the Optuna tuning stage."""
    with Path(path).open() as f:
        data = yaml_load(f.read()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping of dotted keys to values")
    return data.get("overrides", data)


def public_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration without private keys (those starting with '_')."""
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}
