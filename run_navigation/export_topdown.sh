#!/bin/bash
# Export a Habitat top-down visualization for R2R-CE/RxR-CE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/export_topdown.sh r2rce EPISODE_ID [options]
  bash run_navigation/export_topdown.sh r2rce episode EPISODE_ID [options]
  bash run_navigation/export_topdown.sh rxrce SAMPLE_INDEX [options]
  bash run_navigation/export_topdown.sh rxrce sample SAMPLE_INDEX [options]

Defaults:
  RGB birdseye render, height=3m, transparent background,
  with start/goal/reference-path overlays.

Examples:
  bash run_navigation/export_topdown.sh r2rce 832
  bash run_navigation/export_topdown.sh r2rce 832 --height-above 4 --hfov 100
  bash run_navigation/export_topdown.sh rxrce 794

Options are forwarded to tools/export_habitat_topdown_map.py, e.g.
  --output PATH
  --width 1024 --height 1024
  --height-above 3
  --hfov 105
  --rgb-center start|scene|reference_mid|reference_bounds
  --center-offset-x METERS --center-offset-z METERS
  --fit-reference
  --clean / --no-overlays
  --mode rgb|map
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    usage
    exit 0
fi

FAMILY="${1:-r2rce}"
shift $(( $# >= 1 ? 1 : $# ))

case "$FAMILY" in
    r2rce|rxrce) ;;
    *)
        echo "❌ family must be r2rce or rxrce: $FAMILY" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ "${1:-}" == "episode" || "${1:-}" == "sample" ]]; then
    KIND="$1"
    ID_VALUE="${2:-}"
    shift $(( $# >= 2 ? 2 : $# ))
else
    KIND="episode"
    if [[ "$FAMILY" == "rxrce" ]]; then
        KIND="sample"
    fi
    ID_VALUE="${1:-}"
    shift $(( $# >= 1 ? 1 : $# ))
fi

if [[ -z "$ID_VALUE" || ! "$ID_VALUE" =~ ^[0-9]+$ ]]; then
    echo "❌ missing numeric episode/sample id" >&2
    usage >&2
    exit 1
fi
if [[ "$KIND" == "sample" && "$FAMILY" != "rxrce" ]]; then
    echo "❌ sample mode is only defined for rxrce; use: r2rce episode EPISODE_ID" >&2
    exit 1
fi

PROJECT_ROOT="$(spacevln_project_root)"
PYTHON_BIN="$(spacevln_select_python)"
spacevln_setup_runtime_env "$PYTHON_BIN"

ARGS=(
    --family "$FAMILY"
    --mode rgb
    --height-above 3
    --hfov 105
    --transparent-background
)
OVERLAYS=1
PASSTHROUGH=()
while (( $# > 0 )); do
    case "$1" in
        --clean|--no-overlays)
            OVERLAYS=0
            shift
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done
if [[ "$OVERLAYS" == "1" ]]; then
    ARGS+=(--overlay-path --overlay-goals --overlay-start)
fi
if [[ "$KIND" == "sample" ]]; then
    ARGS+=(--sample-index "$ID_VALUE")
else
    ARGS+=(--episode-id "$ID_VALUE")
fi

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" \
    tools/export_habitat_topdown_map.py \
    "${ARGS[@]}" \
    "${PASSTHROUGH[@]}"
