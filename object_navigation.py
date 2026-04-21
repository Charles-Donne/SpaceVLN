"""Convenience entrypoint for OVON object-navigation runs."""

from __future__ import annotations

import contextlib
import io
import sys
import types


def _suppress_gym_notice() -> None:
    gym_notices = types.ModuleType("gym_notices.notices")
    gym_notices.notices = {}
    sys.modules.setdefault("gym_notices.notices", gym_notices)


with contextlib.redirect_stderr(io.StringIO()):
    _suppress_gym_notice()
    from navigation_system.runtime.object_navigation.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
