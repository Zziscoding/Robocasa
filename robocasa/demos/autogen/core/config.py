"""YAML config helpers used by the autogen entry point.

These were originally private helpers inside
``robocasa.demos.demo_open_drawer_contact_curobo`` so they are reproduced
here — with no external dependencies — purely so ``autogen.py`` can live
inside the ``robocasa/demos/autogen`` folder without importing anything
outside it.
"""

from __future__ import annotations


def _parse_config_value(value):
    """Parse a ``--set KEY=VALUE`` raw string through ``yaml.safe_load``.

    Falls back to the raw string when YAML parses it as ``None`` but the
    literal is not an explicit null token (``null`` / ``none`` / ``~``).
    """
    import yaml

    parsed = yaml.safe_load(value)
    return (
        value
        if parsed is None and value.lower() not in ("null", "none", "~")
        else parsed
    )


def _apply_config_overrides(config, overrides):
    """Apply ``--set KEY=VALUE`` CLI overrides onto a config dict (in-place)."""
    for item in overrides:
        key, sep, raw_value = str(item).partition("=")
        if not sep or not key:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        config[key.replace("-", "_")] = _parse_config_value(raw_value)
    return config


def load_yaml_config(path):
    """Load a YAML file and return it as a ``dict``."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data
