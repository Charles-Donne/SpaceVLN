# SpaceVLN Docker Hub 部署说明

> 注意：当前仓库**没有提交** `Dockerfile.spacevln` 与 `.dockerignore`。  
> 本文档保留为历史部署流程参考，方便后续恢复 Docker 构建链路；它不是 fresh clone 后立即可执行的保证文档。

这套打包方案默认做的是“轻镜像”，并且采用下面这套更稳定的组合：

- `habitat-sim`：使用官方 Conda headless 预编译包 `0.1.7`。
- `habitat-lab`：使用当前工作区里的本地源码目录，再按官方依赖方式安装。
- `GroundingDINO`：使用当前工作区里的本地源码目录。
- `SpaceVLN`：使用当前工作区里的本地源码目录。
- 镜像里不包含本地 `data/` 数据集、模型权重和你当前的 API key 配置文件。
- 这样推到 Docker Hub 更安全，也更适合在别的服务器上重复部署。

## 1. 目录要求

如果你要恢复这套流程，构建上下文必须是当前工作区根目录，也就是同时包含这些目录和文件：

```text
nav_ws/
├── Dockerfile.spacevln
├── .dockerignore
├── SpaceVLN/
├── habitat-lab/
└── GroundingDINO/
```

说明：

- 这里不再要求把本地 `habitat-sim` 编译产物一起打包。
- Docker 内部会直接通过 `conda install -c aihabitat -c conda-forge habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7` 安装官方 headless 版本。
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

- 数据目录：`/srv/spacevln/data`
- API 配置文件：`/srv/spacevln/vlm_api_config.yaml`
- 输出目录：`/srv/spacevln/output`
- NVIDIA 驱动和 NVIDIA Container Toolkit

可以这样启动：

```bash
docker run --rm -it \
  --gpus all \
  -e OPENAI_API_KEY=your_openai_key_if_needed \
  -v /srv/spacevln/data:/workspace/data \
  -v /srv/spacevln/vlm_api_config.yaml:/workspace/SpaceVLN/navigation_system/config/vlm/vlm_api_config.yaml:ro \
  -v /srv/spacevln/output:/workspace/output \
  <你的DockerHub用户名>/spacevln:v1 \
  bash run_r2r/vlm_navigation.sh 1 10 260 1
```

说明：

- 容器内默认工作目录是 `/workspace/SpaceVLN`。
- `SPACEVLN_RESULTS_ROOT` 已经在镜像里默认设成 `/workspace/output`。
- 检测模型路径默认会去找 `/workspace/data/model/...`。
- Habitat 数据默认会去找 `/workspace/data/datasets/...` 和 `/workspace/data/scene_datasets/...`。

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

## 6. 如果你想把数据也一起打进镜像

默认不建议，因为你当前工作区的 `data/` 大约有 25G，镜像会非常大，推送 Docker Hub 会很慢。

如果你确实要做“全量镜像”：

1. 从 `.dockerignore` 里删掉 `data/`。
2. 在 `Dockerfile.spacevln` 里增加一行：

```dockerfile
COPY data /workspace/data
```

3. 重新构建并推送。

这种方式更便携，但镜像体积会明显增大，不适合频繁更新。

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
