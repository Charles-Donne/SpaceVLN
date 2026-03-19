"""
VLM Navigation Runner
=====================
Thin CLI entrypoint for SpaceVLN batch navigation.
"""

from vlnce_baselines.vlm.runner import build_arg_parser, run_navigation_from_args


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_navigation_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
