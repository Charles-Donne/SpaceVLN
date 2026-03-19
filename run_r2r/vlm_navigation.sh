#!/bin/bash
# VLM Navigation Controller
# VLM自动导航系统：LLM规划 + VLM执行 + 语义建图
# 
# ⚠️ 步数限制说明：
# 1. 默认：从 habitat_extensions/config/spacevln_task.yaml 读取
#    ENVIRONMENT.MAX_EPISODE_STEPS: 500 (默认)
# 2. 命令行参数可以覆盖配置文件的值
# 
# 用法: 
#   单个episode:           bash run_r2r/vlm_navigation.sh [episode_id] [max_steps]
#   批量运行:              bash run_r2r/vlm_navigation.sh [start_id] [num_episodes] [max_steps]
#   随机运行:              bash run_r2r/vlm_navigation.sh random [num_episodes] [max_steps]
#   指定列表:              bash run_r2r/vlm_navigation.sh list [episode_ids] [max_steps]
# 
# 示例:
#   bash run_r2r/vlm_navigation.sh 832              # 使用配置文件的最大步数
#   bash run_r2r/vlm_navigation.sh 832 300          # 设置最大步数为 300
#   bash run_r2r/vlm_navigation.sh 0 10 200         # 测试 10 个episodes，每个最多 200 步
#   bash run_r2r/vlm_navigation.sh random 20 150    # 随机测试 20 个episodes，最多 150 步
#   bash run_r2r/vlm_navigation.sh list "832,701" 400  # 指定episodes，最多 400 步

set -e
trap 'echo "❌ 错误：脚本在第 $LINENO 行失败"; exit 1' ERR

# 环境变量
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error

# 参数解析
EPISODE_ID=${1:-0}
NUM_EPISODES=${2:-1}
MAX_STEPS=${3:-}  # 可选的第3个参数：最大步数
RANDOM_MODE=""
EPISODE_IDS_MODE=""

# 检查是否为随机模式或列表模式
if [ "$EPISODE_ID" == "random" ]; then
    RANDOM_MODE="--random"
    EPISODE_ID=0
elif [ "$EPISODE_ID" == "list" ]; then
    EPISODE_IDS_MODE="--episode-ids"
    EPISODE_IDS="$NUM_EPISODES"  # 第2个参数是episode ID列表
    NUM_EPISODES=1
    EPISODE_ID=0
fi

# 参数验证
MIN_EPISODE_ID=1
MAX_EPISODE_ID=1800

if ! [[ "$EPISODE_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ episode_id必须是正整数: $EPISODE_ID"
    exit 1
fi

# 验证episode ID范围（仅在非random/list模式下）
if [ -z "$RANDOM_MODE" ] && [ -z "$EPISODE_IDS_MODE" ]; then
    if [ "$EPISODE_ID" -lt "$MIN_EPISODE_ID" ]; then
        echo "❌ episode_id不能小于 $MIN_EPISODE_ID: $EPISODE_ID"
        echo "   建议使用: bash run_r2r/vlm_navigation.sh $MIN_EPISODE_ID ..."
        exit 1
    fi
    
    END_EPISODE_ID=$((EPISODE_ID + NUM_EPISODES - 1))
    if [ "$END_EPISODE_ID" -gt "$MAX_EPISODE_ID" ]; then
        echo "❌ 结束episode ID ($END_EPISODE_ID) 超过最大值 $MAX_EPISODE_ID"
        MAX_NUM=$((MAX_EPISODE_ID - EPISODE_ID + 1))
        echo "   建议使用: bash run_r2r/vlm_navigation.sh $EPISODE_ID $MAX_NUM"
        exit 1
    fi
fi

# 配置路径
CONFIG_FILE="vlnce_baselines/config/experiments/r2r_eval.yaml"
RESULTS_DIR="/root/autodl-tmp/result/spacevln"  # 修改此处可更改结果存储路径
API_CONFIG="vlnce_baselines/config/api/vlm_api_config.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Habitat配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$API_CONFIG" ]; then
    echo "⚠️  统一 API 配置文件不存在: $API_CONFIG"
    echo "   请从 vlnce_baselines/config/api/vlm_api_config.yaml.template 复制并配置"
fi

# 环境检查
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python环境"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "⚠️  未检测到GPU，将使用CPU模式"
fi

# 打印配置信息
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           VLM Navigation Controller                        ║"
echo "║       LLM Planning + VLM Action Execution                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 从配置文件读取默认最大步数
DEFAULT_MAX_STEPS=$(grep -A 2 "ENVIRONMENT:" habitat_extensions/config/spacevln_task.yaml | grep "MAX_EPISODE_STEPS" | awk '{print $2}' || echo "500")

# 如果命令行指定了最大步数，使用命令行参数；否则使用配置文件的值
if [ -z "$MAX_STEPS" ]; then
    DISPLAY_MAX_STEPS="$DEFAULT_MAX_STEPS (配置文件)"
    MAX_STEPS_ARG=""
else
    DISPLAY_MAX_STEPS="$MAX_STEPS (命令行参数)"
    MAX_STEPS_ARG="--max-steps $MAX_STEPS"
fi

if [ -n "$RANDOM_MODE" ]; then
    echo "📋 配置: 随机运行 $NUM_EPISODES 个episodes | 最大步数 $DISPLAY_MAX_STEPS"
elif [ -n "$EPISODE_IDS_MODE" ]; then
    echo "📋 配置: 指定运行 episodes $EPISODE_IDS | 最大步数 $DISPLAY_MAX_STEPS"
else
    if [ "$NUM_EPISODES" -eq 1 ]; then
        echo "📋 配置: Episode $EPISODE_ID | 最大步数 $DISPLAY_MAX_STEPS"
    else
        END_ID=$((EPISODE_ID + NUM_EPISODES - 1))
        echo "📋 配置: Episodes $EPISODE_ID-$END_ID (共$NUM_EPISODES个) | 最大步数 $DISPLAY_MAX_STEPS"
    fi
fi

echo "📁 结果: $RESULTS_DIR/"
echo ""
echo "🤖 模型配置:"
echo "   API: $API_CONFIG"
echo ""
echo "🔄 工作流程:"
echo "   1. 360°环视建图 + 收集4方向图像"
echo "   2. LLM生成子任务规划"
echo "   3. VLM循环执行动作"
echo "   4. 验证子任务完成并重规划"
echo ""
echo "💾 输出目录:"
echo "   rgb/          - RGB观测图像"
echo "   global_map/   - 全局语义地图"
echo "   detection/    - 检测结果"
echo "   vlm/          - VLM相关文件"
echo "     observations/ - 4方向观察图像"
echo "     subtasks/     - 子任务JSON"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "🚀 启动中..."
START_TIME=$(date +%s)

set +e
CUDA_VISIBLE_DEVICES=0 python vlm_navigation.py \
    --exp-config "$CONFIG_FILE" \
    --episode-id "$EPISODE_ID" \
    --num-episodes "$NUM_EPISODES" \
    $RANDOM_MODE \
    $EPISODE_IDS_MODE ${EPISODE_IDS:+"$EPISODE_IDS"} \
    --results-dir "$RESULTS_DIR" \
    --vlm-api-config "$API_CONFIG" \
    --max-subtask-steps 5 \
    $MAX_STEPS_ARG

EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 导航完成 | 用时 ${DURATION}秒"
else
    echo "❌ 异常退出 (code: $EXIT_CODE)"
fi

echo "📁 输出目录: $RESULTS_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    RGB_COUNT=$(find "$OUTPUT_DIR/rgb" -name "*.png" 2>/dev/null | wc -l)
    MAP_COUNT=$(find "$OUTPUT_DIR/global_map" -name "*.png" 2>/dev/null | wc -l)
    VLM_COUNT=$(find "$OUTPUT_DIR/vlm" -name "*.json" 2>/dev/null | wc -l)
    echo "📁 $OUTPUT_DIR/"
    echo "   RGB: $RGB_COUNT | Map: $MAP_COUNT | VLM: $VLM_COUNT"
else
    echo "⚠️  输出目录未创建"
fi

echo "════════════════════════════════════════════════════════════"

exit $EXIT_CODE
