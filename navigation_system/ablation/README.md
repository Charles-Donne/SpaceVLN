# SpaceVLN Ablation Runtime

这个目录只做**原系统减法消融**：

- 不改原始 `navigation_system/vlm/prompts/*`
- 不额外拼接新的说明 prompt
- 仍然走原来的 prompt 模板，只是把部分输入结果不再送进去

## 三种预设

- `navigation_system/ablation/configs/no_landmark.yaml`
  - 去掉 landmark 感知相关输入
- `navigation_system/ablation/configs/no_space_structure.yaml`
  - 去掉 space structure 相关输入
- `navigation_system/ablation/configs/no_landmark_no_space_structure.yaml`
  - landmark 和 space structure 都去掉

## 消融含义

### `landmark` 消融

- thinking 图里不再给 landmark detection box / landmark strip
- verify 时不再给 previous-subtask landmark summary
- action prompt 不再给 `detected_landmarks` / `landmark_map_info`
- action 图改成原始 RGB + obstacle overlay，不再用 landmark detection render

### `space structure` 消融

- verify prompt 不再给 `Space Structure` 文本摘要
- thinking 图里不再给 `Space Waypoint / Current Area / last visited marker`
- thinking 用的 global map 不再画 space-structure overlay

说明：当前主系统里，`space structure` 主要是 thinking 侧输入；action 侧本来就没有单独的 `Space Structure` 文本摘要，因此这个消融主要改 thinking prompt 和 thinking 图像。

## 入口

- `vlm_navigation_ablation.py`
- `run_r2r/vlm_navigation_ablation.sh`
- `vlm_navigation_ablation_qwen_cache.py`
- `run_r2r/vlm_navigation_ablation_qwen_cache.sh`

## 用法

```bash
bash run_r2r/vlm_navigation_ablation.sh landmark 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh space_structure 1 10
```

```bash
bash run_r2r/vlm_navigation_ablation.sh both 1 10
```

也支持显式参数写法：

```bash
bash run_r2r/vlm_navigation_ablation.sh --ablation no_landmark 1 10
```

也兼容旧的环境变量写法：

```bash
ABLATION_CONFIG=navigation_system/ablation/configs/no_landmark.yaml \
bash run_r2r/vlm_navigation_ablation.sh 1 10
```

查看脚本提示：

```bash
bash run_r2r/vlm_navigation_ablation.sh --help
bash run_r2r/vlm_navigation_ablation_qwen_cache.sh --help
```

## 结果隔离

默认结果目录会写到：

- `.../<原模型目录>/ablation/<ablation_name>/`

并自动保存：

- `result_dir/ablation/manifest.json`
- `result_dir/ablation/config.yaml`
