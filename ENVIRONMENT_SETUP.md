# Seedance2.0_AI 环境准备

这份说明对应当前精简仓库 `seedance2.0_AI_upd`。

## 1. 运行所有 `run_*.sh` 需要的环境

### 系统工具

- `bash`
- `python3.11` 或兼容的 `python3`
- `uv`

说明：

- 仓库默认用 `.venv/bin/python` 跑大多数 `run_*.sh`
- `scripts/install_video_pipeline.sh` 依赖 `uv`
- `ffmpeg` 不需要系统安装，项目通过 `imageio-ffmpeg` 自带二进制解决

### Python 依赖

当前代码里确认需要的第三方包主要是：

- `faster-whisper`
- `rapidocr-onnxruntime`
- `scenedetect`
- `opencv-python-headless`
- `imageio-ffmpeg`
- `Pillow`
- `numpy`
- `jsonschema`

这些已经写在 [`requirements-video-pipeline.txt`](./requirements-video-pipeline.txt)。

### 必备本地配置文件

以下 `run_*.sh` 默认要求对应的 `config/*.local.json` 存在：

- `config/video_pipeline.local.json`
- `config/series_pipeline.local.json`
- `config/openai_agent_flow.local.json`
- `config/art_assets_pipeline.local.json`
- `config/explosive_rewrite_pipeline.local.json`
- `config/nano_banana_assets.local.json`
- `config/genre_package_audit.local.json`
- `config/sync_series_learning.local.json`
- `config/style_transfer_assets.local.json`

### 环境变量 / 密钥

按 workflow 分组如下。

#### 视频学习链

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DASHSCOPE_API_KEY`

三者不一定都要有，但你在配置里选了哪个 provider，就至少要配那个。

#### OpenAI 生产链

- `OPENAI_API_KEY`

如果 art 阶段或其他阶段切到 Gemini，也要配：

- `GEMINI_API_KEY`

#### Nano Banana 出图 / 风格转换

- `GEMINI_API_KEY`

#### Seedance 任务提交

- `ARK_API_KEY`

可选控制项：

- `SEEDANCE_MODEL`
- `SEEDANCE_RATIO`
- `SEEDANCE_RESOLUTION`
- `SEEDANCE_DURATION`
- `SEEDANCE_GENERATE_AUDIO`
- `SEEDANCE_WATERMARK`

#### 上传参考图到 TOS

至少需要一组：

- `TOS_ACCESS_KEY_ID` + `TOS_SECRET_ACCESS_KEY`
- 或 `TOS_AK` + `TOS_SK`

通常还建议补：

- `TOS_ENDPOINT`
- `TOS_REGION`
- `TOS_BUCKET`

#### Model Gate 自动送审 / 回填链

- `MODEL_GATE_ACCESS_KEY`
- `MODEL_GATE_SECRET_KEY`

可选：

- `MODEL_GATE_SIGNATURE_SCRIPT`
- `MODEL_GATE_SIGNATURE_MODE`

## 2. 一键初始化

执行：

```bash
./scripts/bootstrap_repo_env.sh
```

它会做这些事：

1. 创建 `.venv`
2. 安装 Python 依赖
3. 复制缺失的 `config/*.example.json -> config/*.local.json`
4. 复制 `.env.example -> .env`
5. 提示你接下来需要填写哪些 key 和路径

初始化后可以执行：

```bash
./scripts/check_repo_env.sh
```

它会直接告诉你当前机器还缺哪些命令、配置文件和环境变量。

## 3. 推荐初始化顺序

```bash
cd seedance2.0_AI_upd
./scripts/bootstrap_repo_env.sh
```

然后编辑：

- `.env`
- `config/video_pipeline.local.json`
- `config/series_pipeline.local.json`
- `config/openai_agent_flow.local.json`
- 其他你会实际用到的 `config/*.local.json`

## 4. 每条主链最低配置

### `run_video_pipeline.sh`

至少需要：

- `.venv`
- `config/video_pipeline.local.json`
- 选中 provider 的 API key
- 一个真实 `video_path`

### `run_series_pipeline.sh`

至少需要：

- `.venv`
- `config/series_pipeline.local.json`
- 选中 provider 的 API key
- 一个真实 `videos/<剧名>/` 或 `videos/<分类>/<剧名>/`

### `run_openai_agent_flow.sh`

至少需要：

- `.venv`
- `config/openai_agent_flow.local.json`
- `OPENAI_API_KEY`
- 上游已有 `script/<剧名>/...`

### `run_nano_banana_assets.sh`

至少需要：

- `.venv`
- `config/nano_banana_assets.local.json`
- `GEMINI_API_KEY`
- 上游已有脚本和人物/场景提示词

### `run_seedance_style_transfer.sh`

至少需要：

- `.venv`
- `OPENAI_API_KEY`
- 上游已有 `02-seedance-prompts.md/json`
- 仓库中的 `prompt_library/SEARCH_INDEX.json`

### `run_seedance_api_generation.sh`

至少需要：

- `.venv` 或 `python3`
- 上游已有 `02-seedance-prompts.md/json`

### `run_upload_seedance_refs.sh`

至少需要：

- `.venv` 或 `python3`
- TOS 或 Model Gate 对应凭证
- 上游已有参考图和 API 生成脚本

### `run_seedance_batch_submit.sh`

至少需要：

- `.venv` 或 `python3`
- `ARK_API_KEY`
- 已生成 `Pxx__seedance_api.sh`

## 5. 当前检查结论

我已经核过：

- 主要 `run_*.sh` 都是调用本仓库内 Python 脚本
- 没发现必须依赖 `docker`、`node`、`npm`、数据库之类外部运行时
- 视频预处理的核心外部能力由 Python 包解决，不额外要求系统级 `ffmpeg`

当前真正需要你手工补的，主要就是：

- API keys
- `config/*.local.json` 里的本机路径
- 你实际要处理的 `videos/ / script/ / outputs/ / assets/` 数据目录
