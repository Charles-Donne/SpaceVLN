"""Thin CLI entrypoint for SpaceVLN result reports."""

from vlnce_baselines.runtime.results_report import (
    build_results_arg_parser,
    run_results_report_from_args,
)


def main() -> int:
    parser = build_results_arg_parser()
    args = parser.parse_args()
    return run_results_report_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
