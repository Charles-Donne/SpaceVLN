# SpaceVLN Docker Hub 部署说明

当前仓库已经提交并维护了可直接构建的全量镜像工作流。

这套方案以 `nav_ws/` 为 build context，并把下面这些目录一起打进镜像：

- `data/`
- `GroundingDINO/`
- `vlnce/habitat-lab/`
- `SpaceVLN/`

同时保留下面的约束：

- 真实 API key 不写入镜像；
- 构建时会把 `SpaceVLN/navigation_system/config/vlm/*.yaml.template` 复制成运行时占位配置；
- 运行时默认结果目录是 `/workspace/result`；
- 宿主机默认结果目录是 `nav_ws/result`，不再依赖任何用户本机绝对路径。

## 1. 目录要求

构建上下文必须是当前工作区根目录，也就是同时包含这些目录和文件：

```text
nav_ws/
├── Dockerfile.spacevln
├── .dockerignore
├── data/
├── SpaceVLN/
├── vlnce/
│   └── habitat-lab/
└── GroundingDINO/
```

说明：

- 运行时真正使用的二进制 `habitat-sim` 通过 Conda headless 包安装；
- 如果在线 `conda install` 网络不稳定，可以提前把离线包放到 `vendor/conda/` 目录。当前 Dockerfile 会优先使用这个文件：

```text
vendor/conda/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```

## 2. Mac 上直接构建并推送到 Docker Hub

先登录 Docker Hub：

```bash
docker login
```

然后在 `nav_ws` 根目录执行：

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.spacevln \
  -t <你的DockerHub用户名>/spacevln:v1 \
  --push \
  .
```

说明：

- `--platform linux/amd64`：确保你在 Mac 上构建出的镜像可以直接部署到常见 x86 Linux 服务器。
- `--push`：构建完成后直接推到 Docker Hub。
- 如果你想先本地试跑再推送，把 `--push` 改成 `--load`。
- 如果你用的是 Apple Silicon Mac，首次交叉构建 CUDA 扩展会比较慢，这是正常现象。

## 3. 本地冒烟测试

如果你先用了 `--load`，可以先测试镜像能不能起：

```bash
docker run --rm -it <你的DockerHub用户名>/spacevln:v1 bash
```

进入容器后，检查 Python 环境：

```bash
python -c "import groundingdino, habitat, habitat_baselines, habitat_sim, habitat_extensions; print('ok')"
```

## 4. 在目标服务器上部署

先拉镜像：

```bash
docker pull <你的DockerHub用户名>/spacevln:v1
```

假设目标服务器上已经准备好了：

- NVIDIA 驱动和 NVIDIA Container Toolkit
- 可选：单独挂载一个结果输出目录
- 可选：单独挂载一个自定义 API 配置文件

可以这样启动：

```bash
docker run --rm -it \
  --gpus all \
  -e OPENAI_API_KEY=your_openai_key_if_needed \
  -e DASHSCOPE_API_KEY=your_dashscope_key_if_needed \
  -e OPENROUTER_API_KEY=your_openrouter_key_if_needed \
  -v /srv/spacevln/result:/workspace/result \
  <你的DockerHub用户名>/spacevln:v1 \
  bash -lc "cd /workspace/SpaceVLN && bash run_r2r/vlm_navigation.sh 1 10 260 1"
```

说明：

- 容器内默认工作目录是 `/workspace/SpaceVLN`。
- `SPACEVLN_RESULTS_ROOT` 已经在镜像里默认设成 `/workspace/result`。
- 检测模型路径默认会去找 `/workspace/data/model/...`。
- Habitat 数据默认会去找 `/workspace/data/datasets/...` 和 `/workspace/data/scene_datasets/...`。
- 如果你不挂载 `/workspace/result`，结果就写在容器内部镜像层。

## 5. API 配置建议

不要把真实 key 烧进镜像。

推荐在服务器上单独准备一个配置文件，然后挂载进去。配置里支持环境变量写法，例如：

```yaml
provider: "openai"

openai:
  api_key: "env:OPENAI_API_KEY"
  base_url: "https://your-openai-compatible-endpoint"
  wire_api: "responses"
  llm_model: "gpt-5.4"
  vlm_model: "gpt-5.4-nano"
  reasoning_effort: "none"
```

如果你用 DashScope 或 OpenRouter，也可以写成：

```yaml
dashscope:
  api_key: "env:DASHSCOPE_API_KEY"
```

或：

```yaml
openrouter:
  api_key: "env:OPENROUTER_API_KEY"
```

## 6. 当前镜像内容

当前 `Dockerfile.spacevln` 已经是“全量镜像”模式：

- `data/` 会进入 `/workspace/data`
- `GroundingDINO/` 会进入 `/workspace/GroundingDINO`
- `vlnce/habitat-lab/` 会进入 `/workspace/vlnce/habitat-lab`
- `habitat-sim` 运行时来自 Conda 包，不依赖 `/workspace` 下的源码目录
- `SpaceVLN/` 会进入 `/workspace/SpaceVLN`

因此镜像体积会比较大，但好处是换环境后可以直接运行，不需要再单独搬运数据目录。

## 7. 构建失败后的清理建议

前面如果已经有很多次失败构建，通常不用立刻清理。

- Docker 会复用很多中间层缓存，下一次成功构建反而会更快。
- 只有在磁盘明显变紧张时，才建议做清理。

推荐从轻到重：

```bash
docker image prune
```

```bash
docker builder prune
```

如果你确认本机不用的停止容器、悬空镜像、未使用网络都想一起清：

```bash
docker system prune
```

如果连未被容器引用的卷也一起清：

```bash
docker system prune -a --volumes
```

最后这条会删得比较多，只有在你确认不再需要旧缓存和旧卷时再用。
