#!/bin/bash
# Shared preset parsing helpers for SpaceVLN ablation entrypoints.

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
        ""|default|full|all)
            printf '%s\n' "$(spacevln_ablation_default_config)"
            ;;
        landmark|no_landmark|without_landmark)
            printf '%s\n' "navigation_system/ablation/configs/no_landmark.yaml"
            ;;
        space|space_structure|no_space_structure|without_space_structure)
            printf '%s\n' "navigation_system/ablation/configs/no_space_structure.yaml"
            ;;
        both|none|no_landmark_no_space_structure|without_both)
            printf '%s\n' "navigation_system/ablation/configs/no_landmark_no_space_structure.yaml"
            ;;
        *)
            return 1
            ;;
    esac
}

spacevln_ablation_is_preset_token() {
    spacevln_ablation_resolve_preset_path "$1" >/dev/null 2>&1
}

spacevln_ablation_print_usage() {
    local script_name="$1"
    cat <<EOF
用法:
  bash run_r2r/${script_name} [preset] [episode_args...]
  bash run_r2r/${script_name} --ablation [preset] [episode_args...]

推荐 preset:
  default            使用隔离目录，但不做消融
  landmark           去掉 landmark 感知输入
  space_structure    去掉 space structure 输入
  both               同时去掉 landmark 和 space structure

示例:
  bash run_r2r/${script_name} landmark 1 100 260 4
  bash run_r2r/${script_name} space_structure 1 100 260 4
  bash run_r2r/${script_name} both 1 1420 260 4
  bash run_r2r/${script_name} --ablation no_landmark 1 100 260 4

兼容旧写法:
  ABLATION_CONFIG=navigation_system/ablation/configs/no_landmark.yaml \\
  bash run_r2r/${script_name} 1 100 260 4

说明:
  episode_args 与原脚本保持一致，例如:
    832                 -> 跑单个 episode 832
    832 300             -> 跑 episode 832，max_steps=300
    1 100 260 4         -> 从 1 开始跑 100 个，max_steps=260，并行 4
    random 20 260 all 4 -> 随机跑 20 个，并行 4
EOF
}

spacevln_ablation_label_from_config_path() {
    local resolved_config="$1"
    local project_root="$2"
    local label=""
    local preset_path=""
    local resolved_preset_path=""

    for label in default landmark space_structure both; do
        case "$label" in
            default)
                preset_path="$(spacevln_ablation_default_config)"
                ;;
            landmark)
                preset_path="navigation_system/ablation/configs/no_landmark.yaml"
                ;;
            space_structure)
                preset_path="navigation_system/ablation/configs/no_space_structure.yaml"
                ;;
            both)
                preset_path="navigation_system/ablation/configs/no_landmark_no_space_structure.yaml"
                ;;
        esac

        resolved_preset_path="$(spacevln_ablation_resolve_config_path "$preset_path" "$project_root")" || continue
        if [[ "$resolved_config" == "$resolved_preset_path" ]]; then
            printf '%s\n' "$label"
            return 0
        fi
    done

    printf '%s\n' "custom"
}

spacevln_ablation_resolve_config_path() {
    local raw_value="$1"
    local project_root="$2"
    local candidate="$raw_value"

    if [[ -z "$candidate" ]]; then
        return 1
    fi

    if [[ "$candidate" != /* ]]; then
        if [[ -f "$project_root/$candidate" ]]; then
            candidate="$project_root/$candidate"
        fi
    fi

    if [[ ! -f "$candidate" ]]; then
        echo "❌ 消融配置不存在: $raw_value" >&2
        return 1
    fi

    (
        cd "$(dirname "$candidate")" >/dev/null 2>&1 && \
        printf '%s/%s\n' "$(pwd)" "$(basename "$candidate")"
    )
}

spacevln_ablation_parse_cli() {
    local project_root="$1"
    local script_name="$2"
    shift 2

    SPACEVLN_ABLATION_RESOLVED_CONFIG=""
    SPACEVLN_ABLATION_SELECTED_PRESET=""
    SPACEVLN_ABLATION_FORWARD_ARGS=()

    local config_source=""
    local raw_value=""
    local first_arg="${1:-}"
    local second_arg="${2:-}"

    case "$first_arg" in
        -h|--help|help)
            spacevln_ablation_print_usage "$script_name"
            return 2
            ;;
        --ablation|--preset|--ablation-preset)
            if [[ -z "$second_arg" ]]; then
                echo "❌ ${first_arg} 后面需要跟一个 preset 或 yaml 路径" >&2
                spacevln_ablation_print_usage "$script_name" >&2
                return 1
            fi
            raw_value="$second_arg"
            SPACEVLN_ABLATION_FORWARD_ARGS=("${@:3}")
            config_source="arg"
            ;;
        *)
            if [[ -n "$first_arg" ]] && spacevln_ablation_is_preset_token "$first_arg"; then
                raw_value="$first_arg"
                SPACEVLN_ABLATION_FORWARD_ARGS=("${@:2}")
                config_source="arg"
            elif [[ -n "$first_arg" ]] && ([[ "$first_arg" == *.yaml ]] || [[ "$first_arg" == *.yml ]] || [[ "$first_arg" == */* ]]); then
                raw_value="$first_arg"
                SPACEVLN_ABLATION_FORWARD_ARGS=("${@:2}")
                config_source="arg"
            elif [[ -n "$first_arg" ]] && [[ "$first_arg" != --* ]] && [[ "$first_arg" != random ]] && [[ "$first_arg" != list ]] && ! [[ "$first_arg" =~ ^[0-9]+$ ]]; then
                echo "❌ 不支持的消融 preset: $first_arg" >&2
                echo "" >&2
                spacevln_ablation_print_usage "$script_name" >&2
                return 1
            else
                SPACEVLN_ABLATION_FORWARD_ARGS=("$@")
                raw_value="${ABLATION_CONFIG:-${SPACEVLN_ABLATION_CONFIG:-$(spacevln_ablation_default_config)}}"
                config_source="env"
            fi
            ;;
    esac

    local preset_path=""
    if preset_path="$(spacevln_ablation_resolve_preset_path "$raw_value" 2>/dev/null)"; then
        SPACEVLN_ABLATION_SELECTED_PRESET="$(spacevln_ablation_normalize_preset "$raw_value")"
        SPACEVLN_ABLATION_RESOLVED_CONFIG="$(spacevln_ablation_resolve_config_path "$preset_path" "$project_root")" || return 1
        return 0
    fi

    if [[ "$config_source" == "arg" ]] && [[ -n "$raw_value" ]] && [[ "$raw_value" != --* ]] && [[ "$raw_value" != random ]] && [[ "$raw_value" != list ]] && ! [[ "$raw_value" =~ ^[0-9]+$ ]]; then
        SPACEVLN_ABLATION_SELECTED_PRESET="custom"
        SPACEVLN_ABLATION_RESOLVED_CONFIG="$(spacevln_ablation_resolve_config_path "$raw_value" "$project_root")" || {
            echo "" >&2
            spacevln_ablation_print_usage "$script_name" >&2
            return 1
        }
        return 0
    fi

    SPACEVLN_ABLATION_RESOLVED_CONFIG="$(spacevln_ablation_resolve_config_path "$raw_value" "$project_root")" || return 1
    SPACEVLN_ABLATION_SELECTED_PRESET="$(spacevln_ablation_label_from_config_path "$SPACEVLN_ABLATION_RESOLVED_CONFIG" "$project_root")"
}

spacevln_ablation_print_selection() {
    local config_path="$1"
    local preset="$2"
    echo "🧪 Ablation preset: ${preset}"
    echo "📄 Ablation config: ${config_path}"
}
