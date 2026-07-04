"""Local adapters for the heavyweight reference-demo modules.

``autogen.py`` only talks to the demos through the functions re-exported
here, so the rest of the ``autogen`` folder never imports anything from
``robocasa.demos`` directly.  The imports inside these helpers are lazy —
they only fire when the relevant pipeline branch is actually invoked.
"""

from __future__ import annotations

from typing import Any, Callable


# --- open-drawer branch -----------------------------------------------------


def open_evaluate_open_contacts() -> Callable:
    """Return ``demo_open_drawer_autogen.evaluate_open_contacts``."""
    from robocasa.demos import demo_open_drawer_autogen as _mod

    return _mod.evaluate_open_contacts


def open_drawer_main_and_parse() -> tuple[Callable, Callable]:
    """Return ``(demo_open_drawer_contact_curobo.main, .parse_args)``."""
    from robocasa.demos import demo_open_drawer_contact_curobo as _mod

    return _mod.main, _mod.parse_args


# --- close-drawer branch ----------------------------------------------------


def close_parse_args() -> Callable:
    """Return ``demo_close_drawer_contact_curobo.parse_args``."""
    from robocasa.demos.demo_close_drawer_contact_curobo import parse_args

    return parse_args
