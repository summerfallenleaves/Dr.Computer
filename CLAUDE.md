# Dr.Computer — 项目宪法

> 面向 Desktop AI Agent 的 Python Framework。
> 不是"一个会点鼠标的 Demo"，而是"一套可扩展、可替换、可组合的 Computer Use 基础设施"。

---

## 目录

1. [项目定位](#1-项目定位)
2. [核心架构](#2-核心架构)
3. [决策记录（ADR）](#3-决策记录adr)
4. [技术栈](#4-技术栈)
5. [目录结构](#5-目录结构)
6. [**已完成工作详记（Phase 0 + Phase 1）**](#6-已完成工作详记phase-0--phase-1)
7. [**已知遗留问题与技术债**](#7-已知遗留问题与技术债)
8. [**详细开发路线图**](#8-详细开发路线图)
9. [**推荐执行顺序**](#9-推荐执行顺序)
10. [开发规范](#10-开发规范)
11. [与使用者的契约（公共 API）](#11-与使用者的契约公共-api)
12. [当前状态](#12-当前状态)

---

## 1. 项目定位

**Dr.Computer 是一个 Computer Use Framework**，类比 LangGraph 之于 LLM Agent：

- 不绑定某一个模型（OpenAI / Anthropic / Qwen-VL / GLM-4V / Ollama 都可接入）
- 不绑定某一个 Grounder（OmniParser / Qwen-VL 原生 grounding / PaddleOCR 都可接入）
- 不绑定某一个平台（macOS 首发，Windows / Linux 通过实现 Platform 接口扩展）
- 不绑定某一个执行后端（PyAutoGUI / pyobjc / Accessibility API 都可接入）

使用者通过 `pip install dr-computer` 拿到一套 Protocol + 默认实现，能像拼乐高一样组合出自己的 Desktop Agent。

**明确不做的事**：
- 不做 LangChain / LangGraph 的薄封装（核心零依赖该框架）
- 不做"又一个 UI-TARS"（不追求模型权重、不做训练）
- MVP 不做 Browser Agent（DOM 路线由第三方库覆盖）

---

## 2. 核心架构

### 2.1 六大可插拔组件 + AgentLoop

```
┌─────────────────────────────────────────────────────────┐
│                    AgentLoop                            │
│   Observe → Think → Ground → Safety → Act → Memory      │
│                ↓ (循环检测 / 取消 / 事件)                │
└─────┬───────────────────────────────────────────────────┘
      │ 依赖注入
      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Provider │ │Perceiver │ │ Grounder │ │ Executor │ │  Memory  │ │SafetyGuard│
│  (LLM)   │ │ (看屏幕)  │ │ (定位)   │ │ (执行)   │ │ (上下文)  │ │ (拦截)    │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
                                                              │
                                                              ▼
                                                       ┌──────────┐
                                                       │ Verifier │ (Phase 2)
                                                       └──────────┘
```

### 2.2 数据流（单次循环）

```
1. Perceiver.observe()          → Observation (screenshot bytes + 尺寸)
2. Provider.chat(messages, img)  → Intent      (LLM 决策，可能未定位)
3. if Intent 需要坐标且未填:
     Grounder.locate(desc, obs) → GroundedTarget (bbox)
     组装成完整 Action
4. SafetyGuard.check(action)    → allow / deny / ask
5. Executor.execute(action)     → ActionResult
6. Memory.append(step)          → 持久化当前 Step
7. EventEmitter.emit("step")    → 通知 UI / 日志 / 调度器
8. 终止判定（任一满足即停）：
   - Intent.action_type == "done"       → outcome = "done"
   - 连续 N 步同一 Action               → outcome = "loop_detected"
   - 外部 cancel()                       → outcome = "aborted"
   - step >= max_steps                  → outcome = "max_steps"
   - 异常                                → outcome = "failed"
```

### 2.3 Intent vs Action 分层（关键设计）

- **Intent**：Provider 的返回类型，是"LLM 想做什么"，可能只有语义描述（`target_description="登录按钮"`）而无坐标
- **Action**：Loop 解析后的最终动作，必须自包含可执行（坐标已填好）

```python
# 融合 Provider（Qwen-VL 原生 grounding）→ Intent.grounded_target 已填
intent = Intent(action_type="click", grounded_target=GroundedTarget(bbox=(...)))

# 非融合 Provider（GLM-4V）→ Intent.target_description 仅描述
intent = Intent(action_type="click", target_description="登录按钮")
# Loop 调 Grounder.locate() 把 description 转成 GroundedTarget，再生成 Action
```

**好处**：
- Provider 接口无状态（不知道 Grounder 是谁）
- Action 永远是可序列化、可回放的最小单位（Phase 3 Skill Library 的基础）
- 切换融合/非融合 = 换 Provider，不改 Loop

### 2.4 坐标空间约定

| 层 | 坐标空间 |
|---|---------|
| `MacOSScreenshotPerceiver`（默认） | logical pixels（Retina 已降采样） |
| Qwen-VL / GLM-4V 看到的截图 | logical pixels（与 perceiver 一致） |
| 模型返回的 bbox | logical pixels |
| `PyAutoGUIExecutor` 点击坐标 | logical pixels |

**结论**：全程 logical pixels。如需切到 physical pixels（更高的点击精度），构造 perceiver 时传 `retina_physical_pixels=True` 并保证 executor 也用 physical 坐标。

---

## 3. 决策记录（ADR）

### 3.1 初始决策（项目开始时）

| ID | 决策 | 选择 | 理由 |
|----|------|------|------|
| A3 | Planner/Grounder 关系 | 接口分离，允许同一类实现两个 Protocol | 抽象干净 + Qwen-VL 原生 grounding 性能不损失 |
| B2 | Action 驱动方式 | 坐标驱动（target 必须含 bbox） | 便于录制/回放/Skill 化 |
| C3 | 同步模型 | 内部 async + 对外 sync wrapper | 同时满足"用户简单调用"和"可取消/可并发" |
| D1 | Verifier | MVP 不实现，仅定义 Protocol | 先跑通主闭环，验证逻辑 Phase 2 再加 |
| E2 | SafetyGuard | MVP 做黑名单版 | 半天工作量，避免 demo 翻车 |

### 3.2 集成调试中发现并新增的决策（Phase 1 期间）

| ID | 决策 | 选择 | 触发原因 |
|----|------|------|---------|
| A4 | 融合检测机制 | 检查 `Intent.grounded_target` 是否非 None | 不需要 isinstance，纯数据驱动 |
| A5 | bbox 格式兼容 | Parser 同时接受 `[x,y]`（2 元素）和 `[x1,y1,x2,y2]`（4 元素）；2 元素自动展开为 ±12px 的小 bbox | Qwen-VL 实际返回 `[10,10]` 点击点而非 4 元素框 |
| A6 | 循环检测作为终止条件 | 新增 outcome `loop_detected`，连续 N 步同一 action 时触发（默认 N=3） | VLM 在简单点击任务上不会主动 declare done |
| A7 | Loop 必须传历史给 Provider | `_build_messages` 包含最近 5 步的 action 摘要 | 否则模型每步都"失忆"，重复同样动作 |
| F1 | Qwen base_url 选择 | `dashscope.aliyuncs.com`（国际版）而非 `dashscope.aliyun.com`（中国版别名） | 后者在某些网络环境下 DNS 解析间歇性失败 |
| F2 | 关键 prompt 增强 | system prompt 明确说明"已做的动作不要重复；目标已达成立即 done" | 配合 A6/A7 让单步任务能干净退出 |

### 3.3 已否决的方案

- ❌ LangChain / LangGraph 作为核心依赖（API 不稳定、依赖传染、定位冲突）
- ❌ TypeScript 作为主语言（Computer Use 生态全在 Python：PyTorch / OmniParser / PaddleOCR）
- ❌ Browser Agent 路线（DOM 自动化由 Browser Use 等库覆盖，不重叠）
- ❌ 把 API key 写进代码或 examples（统一走 `.env` + `python-dotenv`）

---

## 4. 技术栈

| 维度 | 选择 | 备注 |
|------|------|------|
| 语言 | Python 3.11+（实测 3.12/3.14 均可） | 用 `Literal`、`Self`、`tomllib`、PEP 695 泛型 |
| 包管理 | uv 0.11+ | 比 pip/poetry 快一个数量级 |
| Lint/Format | ruff | 替代 black + isort + flake8 |
| 测试 | pytest + pytest-asyncio | `asyncio_mode = "auto"` |
| 类型检查 | mypy（可选） | 不强制 CI |
| 数据模型 | pydantic v2 | discriminated union 用 `Annotated[Union, Field(discriminator=...)]` |
| HTTP | httpx（被 openai SDK 间接使用） | async 友好 |
| LLM 调用 | openai SDK 1.50+ | Qwen-VL / GLM-4V 都兼容 OpenAI 协议 |
| 环境变量 | python-dotenv 1.0+ | examples 启动时加载 `.env` |
| 截屏（macOS） | pyobjc-core + pyobjc-framework-Quartz | 直接调 CoreGraphics，避免 subprocess |
| 鼠标键盘 | pyautogui 0.9.54+ | 跨平台，足够 MVP |

### 4.1 Phase 1 运行时依赖（最小集）

```
pydantic>=2.5,<3
httpx>=0.27,<1
openai>=1.50,<2
pillow>=10.0,<12
pyautogui>=0.9.54,<1
pyobjc-core>=10.0
pyobjc-framework-Quartz>=10.0
pyobjc-framework-ScreenCaptureKit>=10.0
python-dotenv>=1.0,<2
```

### 4.2 dev 依赖

```
ruff>=0.6
pytest>=8.0
pytest-asyncio>=0.24
mypy>=1.10
```

### 4.3 Phase 2+ 将引入的依赖（未决）

| 用途 | 候选 | 阶段 |
|------|------|------|
| OmniParser HTTP 调用 | 复用 httpx | Phase 1.5 |
| OmniParser 本地服务 | FastAPI + uvicorn + torch + transformers | Phase 1.5（scripts/） |
| SQLite 持久化 | 标准库 `sqlite3`（无需第三方） | Phase 2 |
| OpenAI 原生 vision | 复用 openai SDK | Phase 2 |
| Ollama 本地模型 | ollama-python 或 httpx | Phase 2 |
| macOS Accessibility | pyobjc-framework-ApplicationServices | Phase 2 |
| CLI 入口 | Typer 或 click | Phase 4 |
| HTTP 服务 | FastAPI + uvicorn | Phase 4 |
| 文档站 | mkdocs-material + mkdocstrings | Phase 4 |

---

## 5. 目录结构

```
Dr.Computer/
├── pyproject.toml              ✅
├── README.md                   ⚠️ 待重写（Phase 4）
├── LICENSE                     ✅
├── CLAUDE.md                   ✅（本文档）
├── .env.example                ✅
├── .env                        ✅（gitignored）
├── .gitignore                  ✅（含 .env / .logs / __pycache__）
│
├── docs/                       ⚠️ 空，Phase 4 填充
│   ├── architecture.md
│   └── decisions.md
│
├── src/dr_computer/
│   ├── __init__.py             ✅ 47 个公共 API 导出
│   │
│   ├── core/                   ✅ 零外部依赖（仅 pydantic）
│   │   ├── messages.py         ✅ Message / ContentBlock
│   │   ├── observation.py      ✅ Observation
│   │   ├── grounding.py        ✅ GroundedTarget
│   │   ├── intent.py           ✅ Intent
│   │   ├── actions.py          ✅ 9 种 Action discriminated union
│   │   ├── trajectory.py       ✅ Step / Trajectory / StepStatus
│   │   ├── protocols.py        ✅ 7 Protocol + SafetyDecision/ActionResult/VerifyResult
│   │   ├── loop.py             ✅ AgentLoop + sync wrapper + 循环检测 + 历史传递
│   │   └── safety.py           ✅ DefaultSafetyGuard（黑名单）
│   │
│   ├── providers/
│   │   ├── base.py             ✅ OpenAICompatibleBase
│   │   ├── qwen_vl.py          ✅ 融合 Provider
│   │   ├── glm_4v.py           🔲 Phase 1.5（非融合）
│   │   ├── openai_vision.py    🔲 Phase 2
│   │   └── ollama.py           🔲 Phase 2
│   │
│   ├── perception/
│   │   ├── macos.py            ✅ MacOSScreenshotPerceiver（CoreGraphics）
│   │   └── macos_accessibility.py  🔲 Phase 2
│   │
│   ├── grounding/
│   │   ├── omniparser.py       🔲 Phase 1.5
│   │   └── paddleocr.py        🔲 Phase 2 后备
│   │
│   ├── execution/
│   │   └── pyautogui_exec.py   ✅ PyAutoGUIExecutor
│   │
│   ├── memory/
│   │   ├── in_memory.py        ✅ InMemoryMemory
│   │   └── sqlite.py           🔲 Phase 2
│   │
│   └── utils/
│       ├── bbox.py             ✅ bbox 数学（center/IoU/clamp）
│       ├── image.py            ✅ PNG 编码/缩放/base64
│       └── events.py           ✅ EventEmitter
│   │
│   ├── cli/                    🔲 Phase 4
│   │   └── main.py
│   └── server/                 🔲 Phase 4
│       └── api.py
│
├── examples/
│   ├── 01_open_notes.py        ✅ 端到端 demo（点 Apple 菜单）
│   ├── 02_omniparser.py        🔲 Phase 1.5
│   └── 03_recording.py         🔲 Phase 3
│
├── scripts/
│   ├── dev.sh                  ✅ 预检 + 跑 demo + 存日志
│   └── omniparser_server.py    🔲 Phase 1.5
│
├── .github/workflows/          🔲 Phase 4
│   └── ci.yml
│
└── tests/
    ├── unit/
    │   ├── test_actions.py     ✅ 10 个测试
    │   ├── test_loop.py        ✅ 9 个测试（含循环检测）
    │   └── test_safety.py      ✅ 11 个测试
    └── integration/
        └── test_macos_e2e.py   🔲 Phase 2（需真实 macOS 权限）
```

**图例**：✅ 完成 / ⚠️ 部分完成 / 🔲 未开始

---

## 6. 已完成工作详记（Phase 0 + Phase 1）

### 6.1 Phase 0：项目骨架（已完成）

| 任务 | 状态 | 备注 |
|------|------|------|
| 决策对齐（A3/B2/C3/D1/E2） | ✅ | 详见第 3.1 节 |
| CLAUDE.md 宪法 v1 | ✅ | 本文档为其进化版 |
| `uv init` + `pyproject.toml` | ✅ | hatchling 后端，ruff/pytest/mypy 配置就绪 |
| 目录结构骨架 | ✅ | 8 个子包 + tests/examples/scripts/docs |
| `.gitignore` 覆盖 | ✅ | `.env` / `.venv` / `.logs/` / `__pycache__` |

### 6.2 Phase 1：macOS 端到端最小闭环（已完成，带遗留）

#### 6.2.1 数据模型层（`core/`，零外部依赖）

| 文件 | 行数 | 关键内容 |
|------|------|---------|
| `messages.py` | ~85 | `Message`、`TextBlock`、`ImageBlock`、`ToolCall`、`ToolResult`、`ContentBlock` 联合类型 |
| `observation.py` | ~40 | `Observation`（screenshot + width + height + timestamp + source） |
| `grounding.py` | ~55 | `GroundedTarget`（bbox + label + element_type + confidence），含 `center` 属性，bbox 合法性校验 |
| `intent.py` | ~70 | `Intent`（9 种 action_type），`is_spatial()` / `is_terminal()` 辅助方法 |
| `actions.py` | ~130 | 9 种 Action（click/double_click/right_click/type/hotkey/scroll/wait/drag/done），discriminated union via `Annotated[..., Field(discriminator="type")]` |
| `trajectory.py` | ~90 | `Step`、`Trajectory`、`StepStatus`、`new_task_id()` |
| `protocols.py` | ~200 | 7 个 `Protocol`：Provider/Perceiver/Grounder/Executor/Memory/SafetyGuard/Verifier + 3 个返回类型 |
| `loop.py` | ~400 | `AgentLoop` 含 `arun()`/`run()`、`_run_step()`、`_resolve_action()`、`_resolve_target()`、`_is_stuck_in_loop()`、`_build_messages()`、`cancel()`；`AgentLoopCancelled`、`HumanConfirmationRequired` 异常 |
| `safety.py` | ~150 | `DefaultSafetyGuard` + `SafetyPolicy`（黑名单 hotkey + 文本 regex + 可选 spatial ask） |

#### 6.2.2 默认实现层

| 文件 | 内容 |
|------|------|
| `providers/base.py` | `OpenAICompatibleBase`：封装 openai SDK，处理图片 base64、JSON 容错解析 |
| `providers/qwen_vl.py` | `QwenVLProvider`：融合模式，注入 Qwen-VL 专用 system prompt，解析 2/4 元素 bbox（决策 A5） |
| `perception/macos.py` | `MacOSScreenshotPerceiver`：用 CoreGraphics 直接拿 RGBA → Pillow 编码 PNG，支持 Retina 降采样 |
| `execution/pyautogui_exec.py` | `PyAutoGUIExecutor`：处理 9 种 Action，含 key 别名规范化（`cmd` → `command`） |
| `memory/in_memory.py` | `InMemoryMemory`：进程内 dict + asyncio.Lock（为未来 SQLite 实现留接口形态） |
| `utils/bbox.py` | `center` / `size` / `area` / `is_valid` / `intersection` / `iou` / `clamp` / `union_all` |
| `utils/image.py` | `encode_png` / `decode_png` / `to_base64` / `resize_to_fit` / `resize_bytes` |
| `utils/events.py` | `EventEmitter`：async-first pub/sub，支持 sync + coroutine 订阅者 |

#### 6.2.3 工具与示例

| 文件 | 内容 |
|------|------|
| `examples/01_open_notes.py` | 端到端 demo：加载 .env → 构建 loop → 跑目标。支持 CLI 参数 / 环境变量 / 默认值三级目标解析，缺 key 时清晰报错 |
| `scripts/dev.sh` | bash 调试脚本：5 步预检（Python/deps/.env/权限/网络）+ 跑 demo + tee 到 `.logs/run_*.log` |
| `.env.example` | 模板：`DASHSCOPE_API_KEY` + 可选的 `DR_COMPUTER_QWEN_MODEL` / `DR_COMPUTER_GOAL` |

#### 6.2.4 测试覆盖（30 个测试，全绿）

| 文件 | 测试数 | 覆盖点 |
|------|--------|--------|
| `test_actions.py` | 10 | Action discriminated union 序列化/反序列化、bbox 校验、必需字段 |
| `test_loop.py` | 9 | 融合 click→done、max_steps、循环检测（A6）、SafetyGuard 拦截、无 grounder 报错、事件订阅、取消、sync wrapper、组件构造 |
| `test_safety.py` | 11 | 危险 hotkey（cmd+q）、case/order 无关、危险文本（rm -rf/git push --force/sudo）、strict policy 的 spatial ask、默认值 |

#### 6.2.5 集成调试中发现的 Bug 与修复

这一节记录了 Phase 1 跑通真实端到端时遇到的所有坑，**避免未来重复踩**。

| # | Bug | 现象 | 根因 | 修复 | 对应 ADR |
|---|-----|------|------|------|---------|
| 1 | Qwen base_url 选错 | demo 一启动就 `APIConnectionError: Connection error`，DNS 解析 `dashscope.aliyun.com` 失败 | 默认用了 `aliyun.com`（中国版别名），该域名在某些网络下 DNS 间歇失败 | 改为 `dashscope.aliyuncs.com`（国际版，DNS 更稳） | F1 |
| 2 | Python `.pyc` 缓存掩盖源码改动 | 编辑 `qwen_vl.py` 后跑 demo，`base_url` 仍是旧值；用 `inspect.getsource` 也显示旧值 | `.pyc` 没有按预期失效（疑似 mtime 检测问题），导致运行时仍用旧逻辑 | 开发期间反复清缓存：`find src -name __pycache__ -type d -exec rm -rf {} +`；CI 不受影响 | — |
| 3 | Parser 不认 Qwen-VL 的 bbox 格式 | Qwen-VL 返回 `{"action_type":"click","bbox":[10,10]}`（2 元素点击点），parser 要求 `len(bbox)==4`，结果 `grounded_target=None`，Loop 报"requires grounded_target or target_description" | 不同 VLM 对 grounding 输出格式不一致：Qwen-VL 用 `[x,y]` 点，OmniParser 用 `[x1,y1,x2,y2]` 框 | parser 同时支持 2/4 元素；2 元素自动展开为 ±12px 的小 bbox（点击取 center，等效） | A5 |
| 4 | Loop 不传历史，模型无限循环 | 模型一直返回同样的 click，max_steps 触发，从未 done | `_build_messages` 只发 system+goal，模型不知道自己做过什么 | `_build_messages` 加最近 5 步 action 摘要（按 assistant 消息形式） | A7 |
| 5 | VLM 不会主动 declare done | 即使菜单已打开、模型看到了，仍返回 click 而非 done | VLM 默认倾向"做事"而非"说完成"；prompt 不够明确 | (1) system prompt 强化"已做的不要重复，达成立即 done"；(2) 新增循环检测 outcome `loop_detected` 作为兜底 | A6, F2 |
| 6 | macOS 权限失败时 demo 卡死或报错不清晰 | 截图返回 None 或 1x1，鼠标不动但没有错误信息 | 缺前置检查 | `MacOSScreenshotPerceiver` 在截图小于阈值时抛清晰错误；`scripts/dev.sh` 加预检步骤 | — |
| 7 | bash `set -u` 下空数组报 unbound | `./scripts/dev.sh` 不传参数时报 `ARGS[@]: unbound variable` | `set -u` 对空数组strict | 用 `set --` 把命令位置参数化，避开 `"${ARGS[@]}"` 在空时的边界 case | — |

#### 6.2.6 验收结果

| 验收标准 | 结果 |
|---------|------|
| `uv run ruff check .` 全绿 | ✅ |
| `uv run pytest tests/unit/` 全绿 | ✅ 30 passed |
| `uv run python examples/01_open_notes.py` 端到端跑通 | ✅ 3 步循环检测退出，鼠标真的点到 Apple 菜单 |
| 公共 API `from dr_computer import AgentLoop, ...` 可用 | ✅ 47 个符号导出 |

#### 6.2.7 Phase 1 与原计划的偏差（老实交代）

| 原计划 | 实际 | 原因 |
|--------|------|------|
| 目标"打开备忘录" | 目标"点击 Apple 菜单" | 多步任务会立刻暴露 VLM 不主动 done 的问题；先验证 pipeline 再上多步 |
| outcome 仅有 done/failed/aborted/max_steps | 新增 `loop_detected` | Bug #4/#5 的工程兜底 |
| 仅 13 个源文件 | 实际 24 个 .py | 多了 `__init__.py` 7 个、`base.py`、debug 脚本等基础设施 |

---

## 7. 已知遗留问题与技术债

### 7.1 Phase 1 自身遗留

| 问题 | 影响 | 建议处理阶段 |
|------|------|------------|
| **VLM 不会主动 declare done** | 单步任务靠循环检测兜底；多步任务会失败 | Phase 2：实现 Verifier；Phase 1.5：尝试 few-shot prompt |
| **没有真实多步任务验证** | 不知道"打开备忘录 + 新建笔记"能否跑通 | Phase 1 收尾（见 9.1） |
| **system prompt 比较简单** | 模型对"done"的判断不稳 | Phase 1 收尾 + Phase 2 |
| **bash `--help` 用 awk 实现，依赖注释格式** | 改文件头部注释要小心 | 低优先级，可换成专门的 usage 函数 |
| **`.pyc` 缓存偶尔掩盖源码改动** | 开发体验偶发混乱 | 加 `scripts/clean.sh` 一键清缓存；或设 `PYTHONDONTWRITEBYTECODE=1` |

### 7.2 架构层面的债

| 问题 | 影响 | 建议处理阶段 |
|------|------|------------|
| **Provider 接口未定义错误类型** | 网络错误、JSON 解析错误、模型拒绝都混在 `APIConnectionError` / `ValueError` 里 | Phase 1.5：定义 `ProviderError` 层级 |
| **Memory 没有 recall（跨任务检索）** | Protocol 里有 `recall` 注释但未实现 | Phase 2/3 |
| **EventEmitter 没有错误隔离** | 一个订阅者抛异常会中断后续订阅者 | Phase 2：加 try/except + 错误事件 |
| **`_build_messages` 写死英文 system prompt** | 国际化支持弱 | Phase 2：prompt 模板化 |
| **没有截图归档** | debug 时看不到模型当时看到了什么 | Phase 2：trajectory 含 screenshot bytes，存盘时单独写文件 |
| **PyAutoGUI 的 `scroll` 单位是 click 而非 pixel** | 滚动量不精确 | Phase 2：考虑换 pyobjc 原生事件 |

### 7.3 测试层面的债

| 问题 | 影响 | 建议处理阶段 |
|------|------|------------|
| **没有集成测试** | 真实 macOS 权限/网络/模型的回归没法自动验证 | Phase 2：加 `tests/integration/`，需手动跑 |
| **`QwenVLProvider.parse_intent` 未单测** | 各种 JSON 变体（markdown 包裹、字段缺失）行为可能回归 | Phase 1 收尾 |
| **没有 lint mypy CI** | 类型回归靠人肉 | Phase 4 |

---

## 8. 详细开发路线图

### 8.1 Phase 1 收尾（建议 0.5-1 天）

**目标**：让 Phase 1 真的"可信"，能完成真实多步任务。

| 任务 | 文件 | 验收 |
|------|------|------|
| 改进 system prompt，加 few-shot 示例 | `providers/qwen_vl.py` | 模型在"打开备忘录"任务上能在 5 步内 done |
| 加 `parse_intent` 单元测试（覆盖 markdown 包裹、字段缺失、bbox 各种格式） | `tests/unit/test_qwen_vl.py` | 至少 8 个测试 |
| 加 `scripts/clean.sh` 一键清缓存 | `scripts/clean.sh` | `./scripts/clean.sh` 清空 `__pycache__` + `.logs/*` |
| 真实多步任务测试（打开备忘录 → 新建笔记 → 输入文字） | `examples/01_open_notes.py` 改目标 + 跑通 | 录屏，存到 `.logs/multi_step_demo.mp4` |
| git commit 存档 | — | 提交信息按第 10.5 节规范 |

**风险**：可能发现 Qwen-VL 在多步任务上完全无法 done，需要在 Phase 2 用 Verifier 解决。

---

### 8.2 Phase 1.5：可插拔性验证（3-5 天）

**目标**：证明"换 Provider + 换 Grounder 不改 Loop"，Framework 定位坐实。

#### 8.2.1 OmniParser Grounder

| 任务 | 文件 | 估时 |
|------|------|------|
| 写本地 OmniParser HTTP 服务包装器 | `scripts/omniparser_server.py` | 0.5 天 |
| OmniParser 模型权重下载脚本（gitignored） | `scripts/download_omniparser.py` | 0.2 天 |
| `OmniParserGrounder` 实现 `Grounder` Protocol | `src/dr_computer/grounding/omniparser.py` | 0.5 天 |
| 单元测试（mock HTTP 服务） | `tests/unit/test_omniparser.py` | 0.3 天 |

**OmniParser 服务接口约定**（POST）：
```
POST http://localhost:8000/parse
Content-Type: application/json
{"image": "<base64 PNG>"}
→ {"elements": [{"label": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.95}, ...]}
```

#### 8.2.2 GLM-4V 非融合 Provider

| 任务 | 文件 | 估时 |
|------|------|------|
| 调研 GLM-4V 的 grounding 输出格式（是否原生支持 bbox） | — | 0.2 天 |
| `GLM4VProvider` 实现 `Provider` Protocol（继承 `OpenAICompatibleBase`） | `src/dr_computer/providers/glm_4v.py` | 0.5 天 |
| 单元测试 | `tests/unit/test_glm_4v.py` | 0.2 天 |

#### 8.2.3 对比示例

| 任务 | 文件 | 估时 |
|------|------|------|
| 同一目标用 4 种组合跑，对比效果 | `examples/02_pluggability.py` | 0.5 天 |

四种组合：
1. QwenVLProvider（融合）+ 无 grounder
2. GLM4VProvider + OmniParserGrounder
3. GLM4VProvider + 无 grounder（应该报错"requires grounded_target or target_description"）
4. QwenVLProvider + OmniParserGrounder（融合 + 外部 grounder 兜底）

#### 8.2.4 Phase 1.5 验收

- [ ] `./scripts/dev.sh "点 Apple 菜单"` 用 Qwen-VL 融合模式跑通
- [ ] `python examples/02_pluggability.py` 4 种组合都能识别（即使第 3 种是预期失败）
- [ ] OmniParser 本地服务能启动，响应 < 2 秒
- [ ] `tests/unit/` 全绿，新增至少 15 个测试
- [ ] CLAUDE.md 第 7 节"遗留问题"中 Phase 1 项减少 30%

---

### 8.3 Phase 2：生产级能力（2-3 周）

#### 2.1 Verifier 实现（D1 决策落地）

| 子任务 | 文件 |
|--------|------|
| `ScreenshotDiffVerifier`：截图前后哈希对比 + 简单 LLM 二次判断 | `src/dr_computer/core/verifiers.py` |
| `LLMVerifier`：把 goal + before + after 截图喂给 LLM，问"目标是否达成" | 同上 |
| Loop 集成 Verifier（已有 hook，只需在 `_run_step` 后调） | `src/dr_computer/core/loop.py` |

**关键设计**：Verifier 返回 `satisfied=True` 时，Loop 自动注入 `DoneAction`，不依赖模型自己说 done。**这是修复 Phase 1 遗留 #1 的真正解药**。

#### 2.2 SQLite Memory

| 子任务 | 文件 |
|--------|------|
| `SQLiteMemory` 实现 `Memory` Protocol | `src/dr_computer/memory/sqlite.py` |
| Schema: `trajectories`、`steps`、`observations`（截图单独存文件） | 同上 |
| Migration 支持（轻量，schema_version 表） | 同上 |
| 跨任务 recall（向量检索，可选） | Phase 3 |

#### 2.3 更多 Provider

| Provider | 用途 |
|----------|------|
| `OpenAIVisionProvider`（GPT-4o/5） | 决策能力强，作为对比基准 |
| `OllamaProvider` | 本地模型，隐私场景 |
| `AnthropicProvider` | Claude 的 native computer_use API（如有需要） |

#### 2.4 macOS Accessibility Perceiver

| 子任务 | 文件 |
|--------|------|
| 用 `pyobjc-framework-ApplicationServices` 拿当前 focused window 的 UI 树 | `src/dr_computer/perception/macos_accessibility.py` |
| `Observation` 扩展支持 `ui_tree: list[UIElement] | None` | `core/observation.py` |
| `OmniParserGrounder` / 新 `AccessibilityGrounder` 利用 ui_tree 做更稳定的 grounding | `grounding/` |

#### 2.5 Phase 2 验收

- [ ] `loop = AgentLoop(..., verifier=LLMVerifier(provider))` 能在"打开备忘录"任务上真 done（不靠循环检测）
- [ ] `loop = AgentLoop(..., memory=SQLiteMemory("agent.db"))` 重启进程后能 load 历史 trajectory
- [ ] 至少 3 个 Provider、2 个 Grounder、2 个 Perceiver、2 个 Memory 可选
- [ ] 集成测试 `tests/integration/test_macos_e2e.py` 至少 5 个，可在 CI 上跑（mock 网络）
- [ ] 测试覆盖率达 80%+

---

### 8.4 Phase 3：差异化能力（持续）

#### 3.1 Skill Library

**目的**：把"打开 Safari → 输密码 → 进邮箱"这种常用流程封装成可复用 Skill。

| 子任务 | 文件 |
|--------|------|
| Skill YAML schema 设计 | `docs/skill_schema.md` |
| `SkillLoader`：从 YAML/Python 加载 Skill | `src/dr_computer/skills/loader.py` |
| `SkillRegistry`：注册、查询、组合 Skill | `src/dr_computer/skills/registry.py` |
| `SkillExecutor`：把 Skill 的步骤序列转成 Action 序列 | `src/dr_computer/skills/executor.py` |
| 内置 Skills（open_chrome / commit_git / send_email） | `skills/builtin/*.yaml` |
| Loop 支持"Skill 优先"：如果有匹配 Skill 就直接执行，否则走 LLM | `core/loop.py` |

**Skill YAML 示例**：
```yaml
name: open_safari
description: Open Safari and navigate to a URL
parameters:
  - name: url
    required: true
steps:
  - action: hotkey
    keys: [cmd, space]
  - action: type
    text: "Safari"
  - action: hotkey
    keys: [return]
  - wait: 2
  - action: hotkey
    keys: [cmd, l]
  - action: type
    text: "{{ url }}"
  - action: hotkey
    keys: [return]
```

#### 3.2 Workflow Recorder

**目的**：用户录一次操作，自动生成可复用 Skill。

| 子任务 | 文件 |
|--------|------|
| pyobjc 全局事件监听（鼠标移动/点击/键盘） | `src/dr_computer/recorder/macos_recorder.py` |
| 事件 → Action 序列转换（去抖、合并） | `src/dr_computer/recorder/translator.py` |
| Action 序列 → Skill YAML 生成 | `src/dr_computer/recorder/skill_generator.py` |
| 录制 UI（CLI 优先，Electron 后续） | `scripts/record.sh` |

#### 3.3 多模型协同

**目的**：Planner 用 A 模型、Grounder 用 B 模型、OCR 用本地模型。

| 子任务 | 文件 |
|--------|------|
| `MultiModelProvider`：组合多个 Provider | `src/dr_computer/providers/multi.py` |
| 路由策略：哪个 Intent 类型走哪个模型 | 同上 |

#### 3.4 Electron GUI（长期）

| 子任务 | 文件 |
|--------|------|
| FastAPI WebSocket 流式接口（已经在 Phase 4 雏形） | `src/dr_computer/server/ws.py` |
| Electron 前端 | `desktop/`（新目录） |
| 实时显示截图 + Intent + Action | UI |

---

### 8.5 Phase 4：工程化（与上述并行，1-2 周）

**目的**：让项目"看起来像个开源项目"，可以放心提交 GitHub、发到 PyPI、吸引 contributor。

#### 4.1 CLI 工具

```bash
dr-computer run "打开 Safari"
dr-computer run --provider glm-4v --grounder omniparser "打开 Safari"
dr-computer replay .logs/trajectory_xxx.json
dr-computer record  # 进入录制模式
```

| 子任务 | 文件 |
|--------|------|
| 用 Typer 写 CLI 入口 | `src/dr_computer/cli/main.py` |
| `pyproject.toml` 注册 `[project.scripts]` | `pyproject.toml` |

#### 4.2 FastAPI HTTP 服务

| 端点 | 用途 |
|------|------|
| `POST /run` | 同步跑一个 goal，返回 trajectory |
| `POST /run/stream` | SSE/WebSocket 流式返回每步 |
| `GET /trajectories/{id}` | 查历史 |
| `POST /cancel/{id}` | 取消运行中的任务 |

| 子任务 | 文件 |
|--------|------|
| FastAPI app | `src/dr_computer/server/api.py` |
| 任务管理（asyncio task pool） | `src/dr_computer/server/runner.py` |
| 启动脚本 | `scripts/serve.sh` |

#### 4.3 CI（GitHub Actions）

| 子任务 | 文件 |
|--------|------|
| ruff + pytest on push/PR | `.github/workflows/ci.yml` |
| mypy 检查（非阻塞） | 同上 |
| 覆盖率上报（codecov） | 同上 |
| 自动发 PyPI（tag 触发） | `.github/workflows/release.yml` |

#### 4.4 README 重写

| 子任务 | 内容 |
|--------|------|
| 项目介绍（Framework 定位，对比 UI-TARS/Agent-S） | 顶部 |
| Quickstart（3 行命令跑通） | 中部 |
| 架构图 + 6 Protocol 介绍 | 中部 |
| Provider/Grounder 矩阵 | 中部 |
| 链接到 CLAUDE.md 和 docs/ | 底部 |

#### 4.5 文档站

| 子任务 | 文件 |
|--------|------|
| mkdocs-material 配置 | `mkdocs.yml` |
| API reference（mkdocstrings 自动生成） | `docs/api/` |
| 架构详解 | `docs/architecture.md` |
| ADR 完整版 | `docs/decisions/` |
| GitHub Pages 部署 | `.github/workflows/docs.yml` |

#### 4.6 Phase 4 验收

- [ ] `pip install dr-computer` 后 `dr-computer run "点 Apple 菜单"` 能跑
- [ ] `uvicorn dr_computer.server.api:app` 启动 HTTP 服务，curl 可调
- [ ] GitHub Actions CI 在每个 PR 上跑 ruff + pytest
- [ ] README 配图配文，能吸引 star
- [ ] 文档站发布到 GitHub Pages

---

## 9. 推荐执行顺序

### 9.1 短期（接下来 1-2 周）

按优先级排序：

| 优先级 | 任务 | 理由 | 预估 |
|--------|------|------|------|
| P0 | **git commit 当前所有改动** | 防止丢失，已积累大量未提交工作 | 10 分钟 |
| P0 | **Phase 1 收尾**：改进 prompt + 多步任务验证 | 让 Phase 1 真的"完成"，不留下"理论上跑通"的尴尬 | 0.5-1 天 |
| P1 | **README 重写 + .github/workflows/ci.yml** | 让项目对外可见、CI 守门 | 0.5 天 |
| P1 | **CLI 工具（dr-computer run）** | 大幅降低使用门槛，从"跑脚本"变成"装包用" | 0.5 天 |
| P2 | **Phase 1.5：OmniParser + GLM-4V** | 验证 Framework 可插拔性，是项目灵魂 | 3-5 天 |

### 9.2 中期（接下来 1-2 个月）

| 优先级 | 任务 | 理由 |
|--------|------|------|
| P1 | **Phase 2：Verifier + SQLite Memory** | 修复 Phase 1 的"VLM 不主动 done"根本问题；持久化是生产用的前提 |
| P2 | **Phase 2：OpenAIVisionProvider + OllamaProvider** | 多 Provider 让 Framework 定位更扎实 |
| P2 | **Phase 2：macOS Accessibility Perceiver** | 更稳定的 grounding 来源 |
| P3 | **FastAPI HTTP 服务** | 为 Phase 3 Electron GUI 做准备 |

### 9.3 长期（3 个月+）

| 任务 | 战略价值 |
|------|---------|
| Skill Library | 这是 Dr.Computer 区别于"又一个 Computer Use Agent"的核心 |
| Workflow Recorder | 真正的差异化能力，企业最需要 |
| Electron GUI | 把用户群从开发者扩展到普通用户 |
| 多模型协同 | 性能/成本/隐私的最优组合 |

### 9.4 两条并行线建议

**线 A：项目可信度**（必做）
- git commit → Phase 1 收尾 → README → CI → CLI → 发布 PyPI
- 目标：让别人能在 5 分钟内 `pip install dr-computer` 跑起来
- 节奏：1-2 周

**线 B：技术深度**（核心）
- Phase 1.5（可插拔）→ Phase 2（Verifier + 持久化）→ Phase 3（Skill）
- 目标：让 Framework 定位坐实，不是空头支票
- 节奏：1-2 个月

两条线**并行**：A 让项目对外可见（吸 star、吸 contributor），B 让项目内核扎实（避免"花架子"批评）。

---

## 10. 开发规范

### 10.1 代码风格

- **类型注解强制**：所有公共 API 必须有完整类型
- **docstring**：公共 Protocol / 类 / 函数必须有 docstring，私有可不写
- **注释**：默认不写，仅在"为什么这样"非显然时写一行
- **emoji**：代码和文档默认不使用，除非用户明确要求
- **行宽**：100 字符（ruff 配置）
- **import 排序**：ruff isort 模式

### 10.2 接口稳定性

- `core/protocols.py` 中 7 个 Protocol 是公共契约，签名变更需升版本号
- `core/` 下其他文件是数据模型，可演进但需保证向后兼容
- `providers/` / `perception/` / `grounding/` / `execution/` / `memory/` 下的实现是内部细节
- 公共 API = `dr_computer.__all__` 中导出的 47 个符号

### 10.3 测试要求

- Protocol 不测（无法测抽象）
- 数据模型：序列化/反序列化、discriminated union 解析
- Loop：用 Mock Provider/Grounder/Executor 测端到端逻辑（`test_loop.py` 已覆盖）
- Provider 解析：mock 各种 JSON 变体
- 集成测试：真实 macOS 截屏（需用户授权辅助功能 / 屏幕录制），不进 CI

### 10.4 安全约束

- **绝不**在代码中硬编码 API key（统一环境变量 `DASHSCOPE_API_KEY` 等）
- **绝不**默认开启"自动执行危险动作"（SafetyGuard 默认开启）
- **绝不**在测试中调用真实 LLM（mock 之）
- examples 中可读取用户本地 `.env`，但 `.env` 必须在 `.gitignore`
- API key 一旦泄露（如聊天历史），**立刻去控制台旋转**

### 10.5 Git 规范

- 提交信息：`<type>: <desc>`，type ∈ `feat / fix / docs / refactor / test / chore`
- 不强制 conventional commits 全套，但 type 必须有
- 单次提交尽量小，单一目的
- **不**自动提交（除非用户明确要求）
- **不**force push 到 main

### 10.6 性能预算

| 指标 | 目标 | 现状 |
|------|------|------|
| 单步循环延迟（不含 LLM） | < 50ms | ✅ 截图 ~30ms，pyautogui < 10ms |
| Qwen-VL 单次调用 | < 3 秒 | ✅ 实测 2-3 秒 |
| OmniParser 单次调用 | < 2 秒 | 🔲 未测 |
| 端到端单步 | < 5 秒 | ✅ 实测约 3 秒 |
| 启动到第一次 action | < 10 秒 | ✅ |

### 10.7 错误处理

- 网络错误：Loop 当前是 fail-fast；Phase 2 加 retry 策略
- JSON 解析错误：Provider 已有 fallback（返回 `done` intent + summary 报错）
- 权限错误：Perceiver 抛清晰错误信息
- SafetyGuard 拒绝：Loop 记为 `skipped`，继续下一步

---

## 11. 与使用者的契约（公共 API）

```python
from dr_computer import (
    AgentLoop,
    QwenVLProvider,
    MacOSScreenshotPerceiver,
    PyAutoGUIExecutor,
    InMemoryMemory,
    DefaultSafetyGuard,
)

loop = AgentLoop(
    provider=QwenVLProvider(api_key=..., model="qwen-vl-max"),
    perceiver=MacOSScreenshotPerceiver(),
    grounder=None,                          # 融合模式不需要
    executor=PyAutoGUIExecutor(),
    memory=InMemoryMemory(),
    safety_guard=DefaultSafetyGuard(),
    max_steps=20,
    loop_detection=3,
)

trajectory = loop.run("打开备忘录并新建一条笔记")
print(trajectory.outcome)  # done / loop_detected / max_steps / failed / aborted
```

任何让上面这段代码失效的改动都视为 breaking change，需升版本号并写迁移指南。

---

## 12. 当前状态

**版本**：0.1.0（Alpha）
**阶段**：Phase 1 已完成，Phase 1 收尾 + Phase 1.5 待启动

### 12.1 已就绪

- ✅ 6 个数据模型 + 7 个 Protocol + AgentLoop（含循环检测、历史传递、sync/async 双入口）
- ✅ 默认实现：QwenVLProvider（融合）/ MacOSScreenshotPerceiver / PyAutoGUIExecutor / InMemoryMemory / DefaultSafetyGuard
- ✅ 工具：bbox / image / events
- ✅ 调试：dev.sh + .env + .env.example + .logs/
- ✅ 测试：30 个单元测试全绿
- ✅ 端到端：demo 真的能截屏 → 调 Qwen-VL → 移动鼠标 → 点击

### 12.2 立即可做

- 🔲 `git commit`（P0，10 分钟）
- 🔲 Phase 1 收尾（P0，0.5-1 天）
- 🔲 README + CI（P1，0.5 天）
- 🔲 CLI 工具（P1，0.5 天）

### 12.3 下一步战略

- 🔲 Phase 1.5：OmniParser + GLM-4V（验证 Framework 可插拔性）
- 🔲 Phase 2：Verifier + SQLite + 更多 Provider（生产级能力）

---

**本文档维护规则**：
- 每完成一个 Phase 必须更新第 12 节
- 每个新决策必须追加到第 3.2 节
- 每个 bug 修复涉及架构选择的，必须记到第 6.2.5 节
- 文档版本与 `pyproject.toml` 的 `version` 字段保持一致
