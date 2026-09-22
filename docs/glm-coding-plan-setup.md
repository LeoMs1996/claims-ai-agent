# GLM Coding Plan 接入与流式输出修复说明

日期：2026-09-18。本文记录本次将项目从本地推理切换到 GLM 云端模式过程中的全部改动、原因与验证结果。

## 背景

项目默认 `MODEL_PROVIDER=local`，依赖 `127.0.0.1:8000` 的本地推理服务（Qwen2.5 系列）。本机未部署该服务，导致调用时报「意图识别或问答模型暂不可用，请检查本地推理服务」。账号已购买 GLM Coding Plan，因此切换到云端模式。

## 关键结论：Coding Plan 的接口限制

同一个 API Key 在智谱不同端点下的可用性（实测）：

| 端点 | 状态 |
|---|---|
| `https://open.bigmodel.cn/api/paas/v4`（通用 OpenAI 兼容） | ❌ 报 1113「余额不足或无可用资源包」，按量计费，与 Coding Plan 额度不通 |
| `https://open.bigmodel.cn/api/coding/paas/v4`（Coding Plan 专属） | ✅ `glm-5.3` 正常，OpenAI 兼容格式 |
| `https://open.bigmodel.cn/api/anthropic/v1/messages`（Anthropic 兼容） | ✅ 可用，但本项目使用 OpenAI SDK，不走此端点 |

**结论：云端模式必须使用 `coding/paas/v4` 端点。**

## 改动清单

### 1. `config.py` — 关闭 GLM 思考模式（流式修复）

`_chat()` 构造 `ChatOpenAI` 时新增参数：

```python
extra_body={"thinking": {"type": "disabled"}} if settings.model_provider == 'bailian' else None,
```

**原因**：`glm-5.3` 是思考模型，默认先深度思考。思考内容在 `reasoning_content` 字段，而 `app.py` 的 `chat_content()` 只读 `content`，导致：

- 意图分类 `classifier.ainvoke` 静默阻塞 4~7 秒（SSE 流无任何输出）；
- 正文极短的场景（如材料门控固定话术）思考结束后一块吐完，体感为「非流式」。

客服问答与意图分类不需要深度推理，关闭思考换取消除静默期。`local` 模式不注入该参数，行为不变。若某个复杂核赔任务需要深度推理，可单独为其实例传 `thinking: {"type": "enabled"}`。

**实测效果**（`/api/assistant/stream`，问题「什么是保险合同的免赔额」）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 意图分类完成 | 4~7.8s | 2.1s |
| 正文输出 | 思考结束后一次性涌出 | 2.9s 起 354 块持续流式至 8.6s |

### 2. `.env` — 新建并切换云端模式

从 `.env.example` 复制创建（`.env` 已被 gitignore），并修改三处：

| 变量 | 值 | 说明 |
|---|---|---|
| `MODEL_PROVIDER` | `bailian` | 切换云端 GLM |
| `OPENAI_BASE_URL` | `https://open.bigmodel.cn/api/coding/paas/v4` | 必须用 Coding Plan 专属端点，通用端点报余额不足 |
| `CLAIMS_API_KEY` | （空） | 非空时应用自身鉴权（`app.py` 的 `X-API-Key` 校验）会拦截未带请求头的本地调用并报 401「API认证失败」；开发环境置空即跳过校验 |

### 3. `.env.example` — 移除真实密钥

`OPENAI_API_KEY`、`CLAIMS_API_KEY`、`REVIEWER_API_KEY` 三处真实 Key 替换为 `your-xxx-key` 占位符，`OPENAI_BASE_URL` 同步改为 coding 端点。

该文件被 git 跟踪。已用 `git log --all -S` 与 `git grep` 验证：真实 Key 从未进入任何提交、远程仓库无泄露、当前被跟踪文件无残留，因此无需作废 Key。

### 4. `~/.zshrc` — uv 镜像

新增环境变量，所有 uv 安装默认走清华源：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. `.venv` — 依赖安装

`requirements.txt`（约 100+ 锁定包）通过 uv 安装至项目 `.venv`（Python 3.12）。启动命令：

```bash
uv run uvicorn app:app --reload --port 8001
```

`uv run` 自动使用项目 `.venv`，无需 activate，与 conda base 环境无关。

## 遗留事项

- **embedding 仍指向本地**（`EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1`，bge-m3），Milvus（19530）、Redis（6379）同理。知识库检索与会话存储相关功能用到时会再报连接错误，届时需 `docker compose up -d` 起配套服务或改用云端 embedding。
- `pyproject.toml` 缺 `requires-python`，`uv run` 会输出无害警告。
- 修改 `.env` 后需重启 uvicorn：`--reload` 只监控 `.py` 文件。
