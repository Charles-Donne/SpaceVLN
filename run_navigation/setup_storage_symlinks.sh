#!/bin/bash
# Create workspace-default symlinks for result/data storage paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(spacevln_project_root)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

RESULT_LINK_PATH="$WORKSPACE_ROOT/result"
DATA_LINK_PATH="$WORKSPACE_ROOT/data"

DISK_ROOT=""
RESULT_TARGET=""
DATA_TARGET=""
LINK_RESULT=1
LINK_DATA=0
FORCE=0
DRY_RUN=0
BACKUP_EXISTING=0
BACKUP_ROOT=""

usage() {
    cat <<'EOF'
Usage:
  bash run_navigation/setup_storage_symlinks.sh [options]

Purpose:
  Keep SpaceVLN paths unified at workspace defaults while placing large data/results
  on another disk via symlinks.

Defaults:
  - result link is enabled
  - data link is disabled (enable explicitly with --with-data or --both)

Options:
  --disk-root DIR          External base directory (e.g. /mnt/ssd/nav_ws)
                           If provided, default targets become:
                             result -> DIR/result
                             data   -> DIR/data
  --result-target DIR      External target directory for workspace result
  --data-target DIR        External target directory for workspace data

  --with-data              Link both result and data
  --data-only              Link data only
  --both                   Alias of --with-data

  --backup-existing        Move existing workspace path into backup before linking
  --backup-root DIR        Backup root (default: <workspace>/.path_backup)
  --force                  Replace existing symlink that points elsewhere
  --dry-run                Print planned operations only

  -h, --help               Show this message

Examples:
  bash run_navigation/setup_storage_symlinks.sh \
    --result-target /mnt/ssd/nav_ws/result \
    --backup-existing

  bash run_navigation/setup_storage_symlinks.sh \
    --disk-root /mnt/ssd/nav_ws \
    --both \
    --backup-existing
EOF
}

abs_path() {
    local raw="$1"
    if [[ "$raw" = /* ]]; then
        printf '%s\n' "$raw"
        return 0
    fi
    printf '%s/%s\n' "$(pwd)" "$raw"
}

ensure_target_dir() {
    local target_dir="$1"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] mkdir -p $target_dir"
        return 0
    fi
    mkdir -p "$target_dir"
}

path_resolves_to() {
    local lhs="$1"
    local rhs="$2"
    local lhs_real=""
    local rhs_real=""
    lhs_real="$(readlink -f "$lhs" 2>/dev/null || true)"
    rhs_real="$(readlink -f "$rhs" 2>/dev/null || true)"
    [[ -n "$lhs_real" && -n "$rhs_real" && "$lhs_real" = "$rhs_real" ]]
}

backup_existing_path() {
    local source_path="$1"

    if [ "$BACKUP_EXISTING" -ne 1 ]; then
        echo "ERROR: $source_path exists. Use --backup-existing to move it before linking." >&2
        return 1
    fi

    local backup_root="$BACKUP_ROOT"
    if [ -z "$backup_root" ]; then
        backup_root="$WORKSPACE_ROOT/.path_backup"
    fi

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local backup_path="$backup_root/$(basename "$source_path")_$timestamp"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] mkdir -p $backup_root"
        echo "[dry-run] mv $source_path $backup_path"
        return 0
    fi

    mkdir -p "$backup_root"
    mv "$source_path" "$backup_path"
    echo "Backed up: $source_path -> $backup_path"
}

create_link() {
    local link_path="$1"
    local target_path_raw="$2"

    if [ -z "$target_path_raw" ]; then
        echo "ERROR: Missing target path for $link_path" >&2
        return 1
    fi

    local target_path
    target_path="$(abs_path "$target_path_raw")"

    ensure_target_dir "$target_path"

    if [ -L "$link_path" ]; then
        if path_resolves_to "$link_path" "$target_path"; then
            echo "Skip: $link_path already points to $target_path"
            return 0
        fi

        if [ "$FORCE" -ne 1 ]; then
            echo "ERROR: $link_path is an existing symlink to another location. Use --force to replace it." >&2
            return 1
        fi

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[dry-run] rm $link_path"
        else
            rm "$link_path"
        fi
    elif [ -e "$link_path" ]; then
        backup_existing_path "$link_path"
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] ln -s $target_path $link_path"
        return 0
    fi

    ln -s "$target_path" "$link_path"
    echo "Linked: $link_path -> $target_path"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --disk-root)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --disk-root requires a path" >&2
                exit 1
            fi
            DISK_ROOT="$2"
            shift 2
            ;;
        --disk-root=*)
            DISK_ROOT="${1#*=}"
            shift
            ;;
        --result-target)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --result-target requires a path" >&2
                exit 1
            fi
            RESULT_TARGET="$2"
            shift 2
            ;;
        --result-target=*)
            RESULT_TARGET="${1#*=}"
            shift
            ;;
        --data-target)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --data-target requires a path" >&2
                exit 1
            fi
            DATA_TARGET="$2"
            shift 2
            ;;
        --data-target=*)
            DATA_TARGET="${1#*=}"
            shift
            ;;
        --with-data|--both)
            LINK_RESULT=1
            LINK_DATA=1
            shift
            ;;
        --data-only)
            LINK_RESULT=0
            LINK_DATA=1
            shift
            ;;
        --backup-existing)
            BACKUP_EXISTING=1
            shift
            ;;
        --backup-root)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --backup-root requires a path" >&2
                exit 1
            fi
            BACKUP_ROOT="$2"
            shift 2
            ;;
        --backup-root=*)
            BACKUP_ROOT="${1#*=}"
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -n "$DISK_ROOT" ]; then
    if [ -z "$RESULT_TARGET" ]; then
        RESULT_TARGET="$DISK_ROOT/result"
    fi
    if [ -z "$DATA_TARGET" ]; then
        DATA_TARGET="$DISK_ROOT/data"
    fi
fi

if [ "$LINK_RESULT" -eq 1 ] && [ -z "$RESULT_TARGET" ]; then
    echo "ERROR: result linking enabled but no target configured. Use --result-target or --disk-root." >&2
    exit 1
fi

if [ "$LINK_DATA" -eq 1 ] && [ -z "$DATA_TARGET" ]; then
    echo "ERROR: data linking enabled but no target configured. Use --data-target or --disk-root." >&2
    exit 1
fi

echo "Workspace root: $WORKSPACE_ROOT"

overall_rc=0

if [ "$LINK_RESULT" -eq 1 ]; then
    if ! create_link "$RESULT_LINK_PATH" "$RESULT_TARGET"; then
        overall_rc=1
    fi
fi

if [ "$LINK_DATA" -eq 1 ]; then
    if ! create_link "$DATA_LINK_PATH" "$DATA_TARGET"; then
        overall_rc=1
    fi
fi

exit "$overall_rc"
