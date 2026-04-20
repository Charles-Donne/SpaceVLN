#!/bin/bash
# Minimal ablation preset resolution helpers used by the canonical VLNCE launcher.

spacevln_ablation_default_config() {
    printf '%s\n' "navigation_system/ablation/configs/default.yaml"
}

spacevln_ablation_normalize_preset() {
    local value="${1:-}"
    value="${value,,}"
    value="${value//-/_}"
    value="${value// /_}"
    printf '%s\n' "$value"
}

spacevln_ablation_resolve_preset_path() {
    local normalized
    normalized="$(spacevln_ablation_normalize_preset "$1")"

    case "$normalized" in
        ""|default)
            printf '%s\n' "$(spacevln_ablation_default_config)"
            ;;
        landmark)
            printf '%s\n' "navigation_system/ablation/configs/no_landmark.yaml"
            ;;
        space_structure)
            printf '%s\n' "navigation_system/ablation/configs/no_space_structure.yaml"
            ;;
        planning_reasoning|thinking_reasoning)
            printf '%s\n' "navigation_system/ablation/configs/no_planning_reasoning.yaml"
            ;;
        action_reasoning)
            printf '%s\n' "navigation_system/ablation/configs/no_action_reasoning.yaml"
            ;;
        planning_action_reasoning|thinking_action_reasoning|all_reasoning)
            printf '%s\n' "navigation_system/ablation/configs/no_planning_action_reasoning.yaml"
            ;;
        both)
            printf '%s\n' "navigation_system/ablation/configs/no_landmark_no_space_structure.yaml"
            ;;
        *)
            return 1
            ;;
    esac
}

spacevln_ablation_resolve_config_path() {
    local raw_value="$1"
    local project_root="$2"
    local candidate="$raw_value"

    if [[ -z "$candidate" ]]; then
        return 1
    fi

    if [[ "$candidate" != /* ]] && [[ -f "$project_root/$candidate" ]]; then
        candidate="$project_root/$candidate"
    fi

    if [[ ! -f "$candidate" ]]; then
        echo "❌ Ablation config does not exist: $raw_value" >&2
        return 1
    fi

    (
        cd "$(dirname "$candidate")" >/dev/null 2>&1 && \
        printf '%s/%s\n' "$(pwd)" "$(basename "$candidate")"
    )
}
