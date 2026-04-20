"""Convenience entrypoint for OVON object-navigation runs."""

from navigation_system.runtime.object_navigation.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
